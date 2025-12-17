# tools/code_runner.py
# This file contains the tool for running code in a specified project directory.

import logging
import subprocess
import os
import threading
from config.settings import Settings

def _run_code_logic(model_name: str, timeout: int, gpu_id: int) -> dict:
    """
    The core logic for running code. This function is not decorated as a tool
    and can be safely imported and called with keyword arguments by the orchestrator.
    """
    # Get the logger specific to this thread
    logger = logging.getLogger(threading.current_thread().name)

    settings = Settings()
    domain_config = settings.domain_config
    task = settings.task
    
    project_dir = domain_config.get('work_dir')
    run_command_template = domain_config.get('run_command')

    if not project_dir or not run_command_template:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": "Error: 'work_dir' or 'run_command' not configured for the current domain."
        }

    command_to_run = run_command_template.format(task=task, model_name=model_name, gpu_id=gpu_id)
    logger.info(f"Executing command in '{project_dir}': {command_to_run}")
    
    result = {
        "exit_code": None,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
    }

    try:
        # We need to ensure the project directory exists.
        if not os.path.isdir(project_dir):
             raise FileNotFoundError(f"The project directory '{project_dir}' does not exist.")

        timed_out = False

        proc = subprocess.Popen(
            command_to_run.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            cwd=project_dir  # Use cwd to run in the correct directory without changing the global CWD
        )

        def kill_proc(p):
            nonlocal timed_out
            timed_out = True
            try:
                p.kill()
            finally:
                timeout_msg = (
                    f"Error: Process timed out after {timeout} seconds. "
                    "Consider checking the code for infinite loops or reducing the number of epochs."
                )
                # Defer attaching the message until after communicate() so it is not overwritten.
                result["stderr"] = (result["stderr"] or "") + timeout_msg

        timer = threading.Timer(timeout, kill_proc, [proc])
        try:
            timer.start()
            stdout, stderr = proc.communicate()
            result["stdout"] = (stdout or "")[-1000:]
            stderr_tail = (stderr or "")[-1000:]
            if timed_out:
                combined_err = (stderr_tail + ("\n" if stderr_tail else "") + result["stderr"]).strip()
                result["stderr"] = combined_err[-2000:]
                logger.warning(f"Command timed out after {timeout}s: {command_to_run}")
            else:
                result["stderr"] = stderr_tail
            result["exit_code"] = proc.returncode if proc.returncode is not None else -1
            result["timed_out"] = timed_out
        finally:
            timer.cancel()

    except FileNotFoundError as e:
        result["exit_code"] = -1
        result["stderr"] = f"Error: {e}"
    except Exception as e:
        result["exit_code"] = -1
        result["stderr"] = f"An unexpected error occurred: {e}"

    return result

# This is the clean function for the orchestrator (paper_agent.py) to call.
def run_code(model_name: str, timeout: int, gpu_id: int) -> dict:
    """

    A simple wrapper that calls the core code running logic.
    This function is NOT a LangChain tool and is safe to be called with keyword arguments.
    """
    return _run_code_logic(model_name=model_name, timeout=timeout, gpu_id=gpu_id)
