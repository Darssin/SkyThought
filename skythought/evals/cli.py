import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import click
import typer
from typing_extensions import Annotated

from skythought.evals.common.entities import (
    Backend,
    BackendParameters,
    SamplingParameters,
)
from skythought.evals.inference_and_check import (
    create_vllm_engine,
    generate_and_save,
    generate_and_score,
    score_results,
)
from skythought.evals.models import ModelConfig, get_system_prompt_keys
from skythought.evals.tasks import TASK_HANDLER_MAP, TASK_NAMES_TO_YAML, TaskConfig
from skythought.evals.util.cli_util import (
    comma_separated_to_list,
    get_deterministic_hash,
    parse_multi_args,
)
from skythought.evals.util.common import set_seed
from skythought.evals.util.results import SummaryResults

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(pretty_exceptions_enable=False)

SAMPLING_PARAMS_DEFAULT = "temperature=0,top_p=1,max_tokens=32768"


def _parse_sampling_params_by_task(raw: str) -> Dict[str, str]:
    """Parses per-task sampling config from a JSON string or JSON file path.

    Expected format:
      {"math500": "temperature=0.6,top_p=0.95,max_tokens=8192", ...}
    or
      {"math500": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192}, ...}
    """
    if not raw:
        return {}

    if os.path.exists(raw):
        with open(raw, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError("`sampling_params_by_task` must be a JSON object.")

    parsed: Dict[str, str] = {}
    for task, val in data.items():
        if not isinstance(task, str):
            raise ValueError("Keys of `sampling_params_by_task` must be task names.")
        if isinstance(val, str):
            parsed[task] = val
        elif isinstance(val, dict):
            parsed[task] = ",".join([f"{k}={v}" for k, v in val.items()])
        else:
            raise ValueError(
                f"Invalid sampling config for task `{task}`. Expect string or object."
            )
    return parsed


def _attach_task_log_handler(log_root: Path, task: str) -> logging.FileHandler:
    task_log_dir = log_root / task
    task_log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = task_log_dir / f"run_{ts}.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    logger.info(f"Task log file: {log_path}")
    return handler


def _detach_task_log_handler(handler: logging.FileHandler) -> None:
    root_logger = logging.getLogger()
    root_logger.removeHandler(handler)
    handler.close()


def _save_multi_task_summary(
    *,
    result_dir: Path,
    model: str,
    backend: Backend,
    rows: List[dict],
) -> None:
    payload = {
        "model": model,
        "backend": backend,
        "result_dir": str(result_dir),
        "tasks": [
            {
                "task": row.get("task"),
                "accuracy": row.get("accuracy"),
                "pass_at_k": row.get("pass_at_k"),
            }
            for row in rows
        ],
    }
    json_path = result_dir / "summary_multi_tasks.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_path = result_dir / "summary_multi_tasks.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("task,accuracy,pass_at_k\n")
        for row in rows:
            pass_at_k = (
                json.dumps(row.get("pass_at_k"), ensure_ascii=False)
                if row.get("pass_at_k") is not None
                else ""
            )
            f.write(f'{row.get("task")},{row.get("accuracy")},{pass_at_k}\n')
    logger.info(f"Saved multi-task summary JSON to {json_path}")
    logger.info(f"Saved multi-task summary CSV to {csv_path}")


def get_run_config(
    task: str,
    task_config: TaskConfig,
    model_config: ModelConfig,
    backend: Backend,
    backend_args_as_dict: dict,
    sampling_params_as_dict: dict,
    start: int,
    end: int,
) -> dict:
    return {
        "task": {
            "name": task,
            "config": task_config.model_dump(),
            "start": start,
            "end": end,
        },
        "model": {
            "name": model_config.model_id,
            "config": model_config.model_dump(),
        },
        "backend": {
            "name": backend,
            "backend_args": backend_args_as_dict,
        },
        "sampling_params": sampling_params_as_dict,
    }


def parse_common_args(
    *,
    task: str,
    model: str,
    task_args: str,
    backend: Backend,
    backend_args: str,
    sampling_params: str,
    n: int,
    batch_size: int,
    system_prompt: str,
    assistant_prefill: str,
) -> Tuple[
    str,
    dict,
    str,
    Backend,
    dict,
    BackendParameters,
    SamplingParameters,
    dict,
    int,
    int,
    str,
]:
    # For strings passed via CLI, recover escape characters properly. This is hacky but works and convenient for short strings
    system_prompt = (
        system_prompt.encode("utf-8").decode("unicode_escape")
        if system_prompt
        else None
    )
    assistant_prefill = (
        assistant_prefill.encode("utf-8").decode("unicode_escape")
        if assistant_prefill
        else None
    )

    # TODO (sumanthrh): We should ideally read from ctx and get user-provided params
    if batch_size != 64 and backend != Backend.VLLM:
        raise ValueError("Batch size is only supported for the vllm backend.")

    # Enable hf_transfer if not overridden by the user
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", None) is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    if task not in TASK_NAMES_TO_YAML:
        raise ValueError(
            f"Task {task} not found. Should be one of {TASK_NAMES_TO_YAML.keys()}"
        )
    task_args_as_dict = parse_multi_args(task_args)
    user_provided_sampling_params_as_dict = parse_multi_args(sampling_params)
    sampling_params_as_dict = parse_multi_args(SAMPLING_PARAMS_DEFAULT)
    sampling_params_as_dict.update(user_provided_sampling_params_as_dict)

    backend_args_as_dict = parse_multi_args(backend_args)

    if n is not None:
        sampling_params_as_dict["n"] = n

    sampling_params: SamplingParameters = SamplingParameters.from_dict(
        backend, sampling_params_as_dict
    )
    backend_params: BackendParameters = BackendParameters.from_dict(
        backend, backend_args_as_dict
    )

    if sampling_params.params.top_p < 1 and model.startswith("openai/o1"):
        print(
            "OpenAI o1 models do not support `top_p` sampling. Resetting `top_p` to 1"
        )
        sampling_params.params.top_p = 1
        sampling_params_as_dict["top_p"] = 1

    if sampling_params.params.temperature == 0 and sampling_params.params.n > 1:
        sampling_params.params.n = 1
        sampling_params_as_dict["n"] = 1
        logger.warning(
            "Warning: Temperature 0 does not support multiple samples. Setting n=1."
        )

    return (
        task,
        task_args_as_dict,
        model,
        backend,
        backend_args_as_dict,
        backend_params,
        sampling_params_as_dict,
        sampling_params,
        n,
        batch_size,
        system_prompt,
        assistant_prefill,
    )


def get_output_dir(
    result_dir,
    *,
    model_id: str,
    task: str,
    start: int,
    end: int,
    run_config: dict,
) -> Path:
    parameter_hash = get_deterministic_hash(run_config)

    return Path(result_dir) / f"{model_id.replace('/', '_')}_{task}_{parameter_hash}"


@app.command("evaluate", help="Evaluate a model on a task")
def evaluate(
    ctx: typer.Context,
    task: Annotated[
        str,
        typer.Option(
            ...,
            help="Task to process.",
            click_type=click.Choice(list(TASK_NAMES_TO_YAML.keys())),
            case_sensitive=False,
        ),
    ],
    model: Annotated[str, typer.Option(..., help="The model to run")],
    backend: Annotated[
        Backend,
        typer.Option(
            help="Backend to use for inference.",
            case_sensitive=False,
        ),
    ] = Backend.VLLM,
    backend_args: Annotated[
        str,
        typer.Option(
            help="Backend parameters to use for inference.",
            case_sensitive=False,
        ),
    ] = "",
    sampling_params: Annotated[
        str,
        typer.Option(
            help="Sampling parameters to use for inference.",
            case_sensitive=False,
        ),
    ] = SAMPLING_PARAMS_DEFAULT,
    result_dir: Annotated[
        str,
        typer.Option(
            help="Result directory to save outputs.",
        ),
    ] = "./",
    system_prompt_name: Annotated[
        str,
        typer.Option(
            help="System prompt template to use, overriding any pre-configured system prompt for this model.",
            click_type=click.Choice(get_system_prompt_keys()),
        ),
    ] = None,
    system_prompt: Annotated[
        str,
        typer.Option(
            help="System prompt to use, overriding any pre-configured system prompt for this model."
        ),
    ] = None,
    n: Annotated[
        int, typer.Option(help="Number of samples generated per problem.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 41,
    assistant_prefill: Annotated[
        str,
        typer.Option(
            help=r'Assistant prefill for the model response, overriding any pre-configured assistant prefill for this model. Ex: "<think>\n"'
        ),
    ] = None,
    as_test: Annotated[
        bool, typer.Option(help="Perform a test run on 10 samples of the dataset.")
    ] = False,
    overwrite: Annotated[
        bool, typer.Option(help="Overwrite existing results.")
    ] = False,
    batch_size: Annotated[
        int,
        typer.Option(
            help="Batch size for inference. only applicable for the vllm backend."
        ),
    ] = 64,
):
    set_seed(seed)

    (
        task,
        _,
        model,
        backend,
        backend_args_as_dict,
        backend_params,
        sampling_params_as_dict,
        sampling_params,
        n,
        batch_size,
        system_prompt,
        assistant_prefill,
    ) = parse_common_args(
        task=task,
        model=model,
        # `evaluate` does not allow customization of `task_args`
        task_args="",
        backend=backend,
        backend_args=backend_args,
        sampling_params=sampling_params,
        n=n,
        batch_size=batch_size,
        system_prompt=system_prompt,
        assistant_prefill=assistant_prefill,
    )
    # ensure parsing was correct
    assert isinstance(sampling_params, SamplingParameters)
    logger.info(
        f"Temperature: {sampling_params.params.temperature}, top_p: {sampling_params.params.top_p}, max_tokens: {sampling_params.params.max_tokens}"
    )

    start = 0
    end = -1
    if as_test:
        start = 0
        end = 10
        sampling_params.params.max_tokens = 2048
        logger.info("Running test run with 10 samples and max tokens set to 2048.")

    task_config = TaskConfig.from_yaml(TASK_NAMES_TO_YAML[task])
    handler_name = task_config.handler
    handler_cls = TASK_HANDLER_MAP[handler_name]
    handler = handler_cls(task_config)

    model_config = ModelConfig.from_model_id(
        model, system_prompt_name, system_prompt, assistant_prefill
    )

    run_config_dict = get_run_config(
        task,
        task_config,
        model_config,
        backend,
        backend_args_as_dict,
        sampling_params_as_dict,
        start,
        end,
    )

    output_dir = get_output_dir(
        result_dir,
        model_id=model,
        task=task,
        start=start,
        end=end,
        run_config=run_config_dict,
    )
    if not overwrite and output_dir.exists() and len(os.listdir(output_dir)) != 0:
        raise ValueError(
            f"Output directory {output_dir} already exists. pass `--overwrite` to overwrite."
        )
    # create result dir if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generate_and_score(
        handler,
        model_config,
        backend,
        backend_params,
        sampling_params,
        output_dir,
        start,
        end,
        run_config_dict,
        batch_size=batch_size,
    )


@app.command("evaluate-multi", help="Evaluate a model on multiple tasks")
def evaluate_multi(
    tasks: Annotated[
        str,
        typer.Option(
            ...,
            help="Comma-separated list of tasks to process.",
        ),
    ],
    model: Annotated[str, typer.Option(..., help="The model to run")],
    backend: Annotated[
        Backend,
        typer.Option(
            help="Backend to use for inference.",
            case_sensitive=False,
        ),
    ] = Backend.VLLM,
    backend_args: Annotated[
        str,
        typer.Option(
            help="Backend parameters to use for inference.",
            case_sensitive=False,
        ),
    ] = "",
    sampling_params: Annotated[
        str,
        typer.Option(
            help="Sampling parameters to use for inference.",
            case_sensitive=False,
        ),
    ] = SAMPLING_PARAMS_DEFAULT,
    sampling_params_by_task: Annotated[
        str,
        typer.Option(
            help=(
                "Task-specific sampling params as JSON string or JSON file path. "
                'Example: {"math500":"temperature=0.6,top_p=0.95,max_tokens=8192"}'
            ),
        ),
    ] = "",
    result_dir: Annotated[
        str,
        typer.Option(
            help="Result directory to save outputs.",
        ),
    ] = "./",
    log_root: Annotated[
        str,
        typer.Option(
            help=(
                "Parent directory for task logs. "
                "Each task writes logs to a dedicated subdirectory."
            ),
        ),
    ] = None,
    system_prompt_name: Annotated[
        str,
        typer.Option(
            help="System prompt template to use, overriding any pre-configured system prompt for this model.",
            click_type=click.Choice(get_system_prompt_keys()),
        ),
    ] = None,
    system_prompt: Annotated[
        str,
        typer.Option(
            help="System prompt to use, overriding any pre-configured system prompt for this model."
        ),
    ] = None,
    n: Annotated[
        int, typer.Option(help="Number of samples generated per problem.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 41,
    assistant_prefill: Annotated[
        str,
        typer.Option(
            help=r'Assistant prefill for the model response, overriding any pre-configured assistant prefill for this model. Ex: "<think>\n"'
        ),
    ] = None,
    as_test: Annotated[
        bool, typer.Option(help="Perform a test run on 10 samples of the dataset.")
    ] = False,
    overwrite: Annotated[
        bool, typer.Option(help="Overwrite existing results.")
    ] = False,
    batch_size: Annotated[
        int,
        typer.Option(
            help="Batch size for inference. only applicable for the vllm backend."
        ),
    ] = 64,
):
    set_seed(seed)

    task_list = [t for t in comma_separated_to_list(tasks) if t]
    if not task_list:
        raise ValueError("At least one task must be provided via `--tasks`.")

    for t in task_list:
        if t not in TASK_NAMES_TO_YAML:
            raise ValueError(
                f"Task {t} not found. Should be one of {TASK_NAMES_TO_YAML.keys()}"
            )

    (
        _,
        _,
        model,
        backend,
        backend_args_as_dict,
        backend_params,
        sampling_params_as_dict,
        sampling_params,
        _,
        batch_size,
        system_prompt,
        assistant_prefill,
    ) = parse_common_args(
        task=task_list[0],
        model=model,
        task_args="",
        backend=backend,
        backend_args=backend_args,
        sampling_params=sampling_params,
        n=n,
        batch_size=batch_size,
        system_prompt=system_prompt,
        assistant_prefill=assistant_prefill,
    )
    assert isinstance(sampling_params, SamplingParameters)
    logger.info(
        f"Temperature: {sampling_params.params.temperature}, top_p: {sampling_params.params.top_p}, max_tokens: {sampling_params.params.max_tokens}"
    )

    start = 0
    end = -1
    if as_test:
        start = 0
        end = 10
        sampling_params.params.max_tokens = 2048
        logger.info("Running test run with 10 samples and max tokens set to 2048.")

    model_config = ModelConfig.from_model_id(
        model, system_prompt_name, system_prompt, assistant_prefill
    )
    per_task_sampling = _parse_sampling_params_by_task(sampling_params_by_task)
    if log_root is None:
        log_root = str(Path(result_dir) / "logs")
    log_root_path = Path(log_root)
    log_root_path.mkdir(parents=True, exist_ok=True)

    vllm_engine = None
    if backend == Backend.VLLM:
        logger.info("Initializing shared vLLM engine for multi-task evaluation.")
        vllm_engine = create_vllm_engine(backend_params, model_config)

    task_rows = []
    for task in task_list:
        logger.info(f"Evaluating task: {task}")
        task_log_handler = _attach_task_log_handler(log_root_path, task)
        task_log_dir = str(log_root_path / task)
        task_sampling_params_str = per_task_sampling.get(task, sampling_params)
        (
            _,
            _,
            _,
            _,
            _,
            _,
            task_sampling_params_as_dict,
            task_sampling_params,
            _,
            _,
            _,
            _,
        ) = parse_common_args(
            task=task,
            model=model,
            task_args="",
            backend=backend,
            backend_args=backend_args,
            sampling_params=task_sampling_params_str,
            n=n,
            batch_size=batch_size,
            system_prompt=system_prompt,
            assistant_prefill=assistant_prefill,
        )
        if as_test:
            task_sampling_params.params.max_tokens = 2048
            task_sampling_params_as_dict["max_tokens"] = 2048
        logger.info(
            f"Task `{task}` sampling params: {task_sampling_params_as_dict}"
        )

        task_config = TaskConfig.from_yaml(TASK_NAMES_TO_YAML[task])
        handler_name = task_config.handler
        handler_cls = TASK_HANDLER_MAP[handler_name]
        handler = handler_cls(task_config)

        run_config_dict = get_run_config(
            task,
            task_config,
            model_config,
            backend,
            backend_args_as_dict,
            task_sampling_params_as_dict,
            start,
            end,
        )

        output_dir = get_output_dir(
            result_dir,
            model_id=model,
            task=task,
            start=start,
            end=end,
            run_config=run_config_dict,
        )
        if not overwrite and output_dir.exists() and len(os.listdir(output_dir)) != 0:
            raise ValueError(
                f"Output directory {output_dir} already exists. pass `--overwrite` to overwrite."
            )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            generate_and_score(
                handler,
                model_config,
                backend,
                backend_params,
                task_sampling_params,
                output_dir,
                start,
                end,
                run_config_dict,
                batch_size=batch_size,
                vllm_engine=vllm_engine,
            )
            summary_path = output_dir / "summary.json"
            results_path = output_dir / "results.json"
            accuracy = None
            pass_at_k = None
            avg_prompt_tokens = None
            avg_completion_tokens = None
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
                accuracy = summary_data.get("accuracy")
                pass_at_k = summary_data.get("pass_at_k")
                avg_prompt_tokens = summary_data.get("avg_prompt_tokens")
                avg_completion_tokens = summary_data.get("avg_completion_tokens")
            task_rows.append(
                {
                    "task": task,
                    "status": "success",
                    "accuracy": accuracy,
                    "pass_at_k": pass_at_k,
                    "avg_prompt_tokens": avg_prompt_tokens,
                    "avg_completion_tokens": avg_completion_tokens,
                    "output_dir": str(output_dir),
                    "summary_path": str(summary_path),
                    "results_path": str(results_path),
                    "log_dir": task_log_dir,
                }
            )
        except Exception as e:
            task_rows.append(
                {
                    "task": task,
                    "status": "failed",
                    "accuracy": None,
                    "pass_at_k": None,
                    "avg_prompt_tokens": None,
                    "avg_completion_tokens": None,
                    "output_dir": str(output_dir),
                    "summary_path": str(output_dir / "summary.json"),
                    "results_path": str(output_dir / "results.json"),
                    "log_dir": task_log_dir,
                    "error": repr(e),
                }
            )
            raise
        finally:
            _detach_task_log_handler(task_log_handler)

    _save_multi_task_summary(
        result_dir=Path(result_dir),
        model=model,
        backend=backend,
        rows=task_rows,
    )


@app.command("generate", help="Generate model response for a task and save results")
def generate(
    task: Annotated[
        str,
        typer.Option(
            ...,
            help="Task to process.",
            click_type=click.Choice(list(TASK_NAMES_TO_YAML.keys())),
            case_sensitive=False,
        ),
    ],
    model: Annotated[str, typer.Option(..., help="The model to run")],
    task_args: Annotated[
        str,
        typer.Option(
            help="Task arguments to use for inference.",
        ),
    ] = "",
    backend: Annotated[
        Backend,
        typer.Option(
            help="Backend to use for inference.",
            case_sensitive=False,
        ),
    ] = Backend.VLLM,
    backend_args: Annotated[
        str,
        typer.Option(
            help="Backend parameters to use for inference.",
            case_sensitive=False,
        ),
    ] = "",
    sampling_params: Annotated[
        str,
        typer.Option(
            help="Sampling parameters to use for inference.",
            case_sensitive=False,
        ),
    ] = "temperature=0,top_p=1,max_tokens=32768",
    result_dir: Annotated[
        str,
        typer.Option(
            help="Result directory to save outputs.",
        ),
    ] = None,
    system_prompt_name: Annotated[
        str,
        typer.Option(
            help="System prompt template to use, overriding any pre-configured system prompt for this model.",
            click_type=click.Choice(get_system_prompt_keys()),
        ),
    ] = None,
    system_prompt: Annotated[
        str,
        typer.Option(
            help="System prompt to use, overriding any pre-configured system prompt for this model."
        ),
    ] = None,
    n: Annotated[
        int, typer.Option(help="Number of samples generated per problem.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 41,
    assistant_prefill: Annotated[
        str,
        typer.Option(help=r'Assistant prefill for the model response. Ex: "<think>\n"'),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option(help="Overwrite existing results.")
    ] = False,
    batch_size: Annotated[
        int,
        typer.Option(
            help="Batch size for inference. only applicable for the vllm backend."
        ),
    ] = 64,
    start: Annotated[int, typer.Option(help="Start index for the dataset.")] = 0,
    end: Annotated[
        int,
        typer.Option(
            help="End index for the dataset (non-inclusive). If a negative value is provided, we use all the samples."
        ),
    ] = -1,
    resume_from: Annotated[
        str, typer.Option(help="Resume from a previous run.")
    ] = None,
):
    set_seed(seed)

    (
        task,
        task_args_as_dict,
        model,
        backend,
        backend_args_as_dict,
        backend_params,
        sampling_params_as_dict,
        sampling_params,
        n,
        batch_size,
        system_prompt,
        assistant_prefill,
    ) = parse_common_args(
        task=task,
        model=model,
        task_args=task_args,
        backend=backend,
        backend_args=backend_args,
        sampling_params=sampling_params,
        n=n,
        batch_size=batch_size,
        system_prompt=system_prompt,
        assistant_prefill=assistant_prefill,
    )

    assert isinstance(sampling_params, SamplingParameters)
    logger.info(
        f"Temperature: {sampling_params.params.temperature}, top_p: {sampling_params.params.top_p}, max_tokens: {sampling_params.params.max_tokens}"
    )

    task_config = TaskConfig.from_yaml(TASK_NAMES_TO_YAML[task])
    # update with user-provided args
    task_config.update(**task_args_as_dict)

    handler_name = task_config.handler
    handler_cls = TASK_HANDLER_MAP[handler_name]
    handler = handler_cls(task_config)

    model_config = ModelConfig.from_model_id(
        model, system_prompt_name, system_prompt, assistant_prefill
    )

    output_dir = None
    if resume_from is not None:
        resume_from = Path(resume_from)
        if not resume_from.exists():
            raise ValueError(f"Output directory {resume_from} does not exist.")

    assert (resume_from is None) ^ (
        result_dir is None
    ), "One of `resume_from` or `result_dir` must be true."

    run_config_dict = get_run_config(
        task,
        task_config,
        model_config,
        backend,
        backend_args_as_dict,
        sampling_params_as_dict,
        start,
        end,
    )

    if result_dir is not None:
        output_dir = get_output_dir(
            result_dir,
            model_id=model,
            task=task,
            start=start,
            end=end,
            run_config=run_config_dict,
        )
        if not overwrite and output_dir.exists() and len(os.listdir(output_dir)) != 0:
            raise ValueError(
                f"Output directory {output_dir} already exists. pass `--overwrite` to overwrite."
            )
        # create result dir if not exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    generate_and_save(
        handler,
        model_config,
        backend,
        backend_params,
        sampling_params,
        output_dir,
        start,
        end,
        run_config_dict,
        resume_from=resume_from,
        batch_size=batch_size,
    )


@app.command("score", help="Score a model on a task")
def score(
    run_dir: Annotated[
        str, typer.Option(..., help="The directory of saved results to score")
    ],
    task: Annotated[
        str,
        typer.Option(
            ...,
            help="Task to process.",
            click_type=click.Choice(list(TASK_NAMES_TO_YAML.keys())),
            case_sensitive=False,
        ),
    ],
    ids: Annotated[
        str,
        typer.Option(
            help="Comma-separated list of indices in the results JSON to re-score."
            "If provided, only the scores for these samples are computed/re-computed. If None, we compute scores for all samples",
        ),
    ] = None,
):
    if not os.path.exists(run_dir):
        raise ValueError(f"Run directory {run_dir} does not exist.")

    run_dir = Path(run_dir)

    if ids:
        ids: List[str] = comma_separated_to_list(ids)
        # make them unique
        ids = list(set(ids))

    if task not in TASK_NAMES_TO_YAML:
        raise ValueError(
            f"Task {task} not found. Should be one of {TASK_NAMES_TO_YAML.keys()}"
        )

    task_config = TaskConfig.from_yaml(TASK_NAMES_TO_YAML[task])
    handler_name = task_config.handler
    handler_cls = TASK_HANDLER_MAP[handler_name]
    handler = handler_cls(task_config)

    # get run_config from run_dir
    summary_file = run_dir / "summary.json"
    if not summary_file.exists():
        raise ValueError(f"Run summary file {summary_file} does not exist.")

    with open(summary_file, "r") as f:
        run_summary = json.load(f)

    run_summary = SummaryResults(**run_summary)

    score_results(handler, run_dir, run_summary, ids)


def main():
    app()
