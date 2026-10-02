import inspect
import logging


def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()


def filter_traceback(s):
    """Return the traceback contained in ``s`` (empty string if there is none)."""
    lines = s.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            filtered_lines = []
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''


def block_until_running(stdout_filepath, log_status=False, iter_num=-1, response_id=-1):
    """Wait until the evaluation subprocess has written its first output."""
    while True:
        log = file_to_string(stdout_filepath)
        if len(log) > 0:
            if log_status and "Traceback" in log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            else:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successful!")
            break


def get_heuristic_name(module, possible_names: list[str]):
    """Used by the problem eval scripts to locate the heuristic function in gpt.py."""
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name
