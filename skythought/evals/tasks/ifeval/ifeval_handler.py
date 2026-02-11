import json

from skythought.evals.scoring.ifeval.instructions_main import (
    InputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)

from ..base import TaskHandler


class IFEvalTaskHandler(TaskHandler):
    def generate_prompt(self, problem):
        return problem[self.task_config.question_key]

    def check_correctness(self, problem, generation):
        inp = InputExample(
            key=problem["key"],
            instruction_id_list=problem["instruction_id_list"],
            prompt=problem["prompt"],
            kwargs=problem["kwargs"],
        )
        out_strict = test_instruction_following_strict(inp, generation)
        return bool(out_strict.follow_all_instructions)

    def update_results(self, problem, response):
        response_entry = {
            "content": response,
            "correctness": None,
            "reason": None,
        }
        inp = InputExample(
            key=problem["key"],
            instruction_id_list=problem["instruction_id_list"],
            prompt=problem["prompt"],
            kwargs=problem["kwargs"],
        )

        out_strict = test_instruction_following_strict(inp, response)
        out_loose = test_instruction_following_loose(inp, response)
        response_entry["correctness"] = bool(out_strict.follow_all_instructions)
        response_entry["reason"] = json.dumps(
            {
                "prompt_level_strict_acc": out_strict.follow_all_instructions,
                "inst_level_strict_acc": out_strict.follow_instruction_list,
                "prompt_level_loose_acc": out_loose.follow_all_instructions,
                "inst_level_loose_acc": out_loose.follow_instruction_list,
            },
            ensure_ascii=False,
        )
        return response_entry

    def load_and_filter_dataset(
        self, start, end, split=None, subset=None, difficulty=None
    ):
        dataset = self.load_dataset(subset=subset, split=split).to_pandas()
        return dataset.iloc[start:end] if end > 0 else dataset.iloc[start:]

