import logging
import os
import subprocess
import sys
from pathlib import Path

import hydra

from genesis import Genesis

ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")
    logging.info(f"Using LLM: {cfg.model}")
    logging.info(f"Problem: {cfg.problem.problem_name} (size {cfg.problem.problem_size})")

    best_code_overall, best_code_path_overall = Genesis(cfg, ROOT_DIR, workspace_dir).run()
    logging.info(f"Best Code Overall: {best_code_overall}")
    logging.info(f"Best Code Path Overall: {best_code_path_overall}")

    # Install the best heuristic and run the validation script on it
    problem_dir = Path(ROOT_DIR) / "problems" / cfg.problem.problem_name
    with open(problem_dir / "gpt.py", 'w') as file:
        file.writelines(best_code_overall + '\n')

    test_script = problem_dir / "eval.py"
    test_script_stdout = "best_code_overall_val_stdout.txt"
    logging.info(f"Running validation script: {test_script}")
    with open(test_script_stdout, 'w') as stdout:
        subprocess.run([sys.executable, str(test_script), "-1", ROOT_DIR, "val"],
                       stdout=stdout, stderr=subprocess.STDOUT, check=False)
    logging.info(f"Validation script finished. Results saved in {test_script_stdout}.")

    with open(test_script_stdout, 'r') as file:
        for line in file:
            logging.info(line.strip())


if __name__ == "__main__":
    main()
