"""Problem definition: prompt templates and sandboxed evaluation of generated heuristics."""
import logging
import os
import re
import subprocess
import sys
from typing import Any

from utils.utils import block_until_running, file_to_string, filter_traceback

# Name of the returned variable, keyed by the prefix of the heuristic's function name
FUNC_OUTPUTS = {
    'select_next_node': 'next_node',
    'priority': 'priority',
    'heuristics': 'heuristics_matrix',
    'crossover': 'offsprings',
    'utility': 'utility_value',
}


class Prompts:
    def __init__(self, problem_cfg, root_dir: str):
        self.cfg = problem_cfg
        prompt_path_suffix = "_black_box" if problem_cfg.problem_type == "black_box" else ""
        problem_prompt_path = f'{root_dir}/prompts/{problem_cfg.problem_name}{prompt_path_suffix}'
        self.func_signature = file_to_string(f'{problem_prompt_path}/func_signature.txt').format(version=2).strip()
        self.func_desc = file_to_string(f'{problem_prompt_path}/func_desc.txt')
        self.seed_func = file_to_string(f'{problem_prompt_path}/seed_func.txt')

        match = re.match(r'^def +(.+?)\((.*)\) *-> *(.*?) *:', self.func_signature)
        assert match is not None
        self.prompt_func_name = match.group(1)
        self.prompt_func_inputs = [txt.split(":")[0].strip() for txt in match.group(2).split(",")]
        output = next((out for prefix, out in FUNC_OUTPUTS.items() if self.prompt_func_name.startswith(prefix)),
                      'result')
        self.prompt_func_outputs = [output]

    def get_task(self):
        return self.cfg.description

    def get_func_name(self):
        return self.prompt_func_name

    def get_func_inputs(self):
        return self.prompt_func_inputs

    def get_func_outputs(self):
        return self.prompt_func_outputs

    def get_inout_inf(self):
        return self.func_desc

    def get_other_inf(self):
        return ""

    def get_seed_func(self):
        return self.seed_func


class Problem:
    def __init__(self, cfg, root_dir):
        self.config = cfg
        self.root_dir = root_dir

        self.problem = cfg.problem.problem_name
        self.problem_description = cfg.problem.description
        self.problem_size = cfg.problem.problem_size
        self.obj_type = cfg.problem.obj_type
        self.problem_type = cfg.problem.problem_type
        # Candidate heuristics are written here and imported by the problem's eval script
        self.output_file = f"{root_dir}/problems/{self.problem}/gpt.py"
        eval_script = "eval_black_box.py" if self.problem_type == "black_box" else "eval.py"
        self.eval_script = f'{root_dir}/problems/{self.problem}/{eval_script}'

        self.prompts = Prompts(cfg.problem, root_dir)

    def response_to_individual(self, code, response_id) -> dict:
        outdir = './evaluations/'
        os.makedirs(outdir, exist_ok=True)
        runid = hash(code)
        file_name = outdir + f"problem_eval{runid}.txt"
        with open(file_name, 'w', encoding='utf-8') as file:
            file.writelines(code + '\n')

        return {
            "stdout_filepath": outdir + f"problem_eval{runid}_stdout.txt",
            "code_path": outdir + f"problem_eval{runid}_code.py",
            "code": code,
            "response_id": response_id,
        }

    @staticmethod
    def mark_invalid_individual(individual: dict, traceback_msg: str) -> dict:
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["traceback_msg"] = traceback_msg
        return individual

    def batch_evaluate(self, codes: list[str], iteration: int) -> str | list[Any]:
        """
        Run each code through the problem's eval script on the training instances.
        Returns a list of objectives (negated for maximisation problems, inf on failure),
        or the string 'timeout' if an evaluation exceeds ``cfg.timeout`` seconds.
        """
        population = [self.response_to_individual(code, index) for index, code in enumerate(codes)]
        for response_id, individual in enumerate(population):
            runid = hash(individual["code"])
            if individual["code"] is None:
                self.mark_invalid_individual(individual, "Invalid response!")
                continue

            logging.info(f"Iteration {iteration}: Running Code {runid}")
            try:
                with open(self.output_file, 'w', encoding='utf-8') as file:
                    file.writelines(individual["code"] + '\n')

                with open(individual["stdout_filepath"], 'w') as f:
                    process = subprocess.Popen(
                        [sys.executable, '-u', self.eval_script, f'{self.problem_size}', self.root_dir, "train"],
                        stdout=f, stderr=f)
                block_until_running(individual["stdout_filepath"], log_status=True)
            except Exception as e:
                logging.info(f"Error for response_id {response_id}: {e}")
                self.mark_invalid_individual(individual, str(e))
                continue

            try:
                process.communicate(timeout=self.config.timeout)
            except subprocess.TimeoutExpired as e:
                logging.info(f"Error for response_id {response_id}: {e}")
                self.mark_invalid_individual(individual, str(e))
                process.kill()
                return 'timeout'

            with open(individual["stdout_filepath"], "r", encoding="utf-8", errors="ignore") as f:
                stdout_str = f.read()
            traceback_msg = filter_traceback(stdout_str)

            if traceback_msg == '':
                try:
                    # The eval script prints the objective on its last line
                    individual["obj"] = float(stdout_str.split('\n')[-2])
                    assert individual["obj"] > 0, "Objective value <= 0 is not supported."
                    individual["obj"] = -individual["obj"] if self.obj_type == "max" else individual["obj"]
                    individual["exec_success"] = True
                except Exception:
                    self.mark_invalid_individual(individual, "Invalid std out / objective value!")
            else:
                self.mark_invalid_individual(individual, traceback_msg)

            logging.info(f"Iteration {iteration}, response_id {response_id}: Objective value: {individual['obj']}")
        return [indiv["obj"] for indiv in population]
