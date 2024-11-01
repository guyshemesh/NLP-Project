import importlib
import os

import bigbench.api.util as util
from bigbench.api import json_task


def get_task(dataset, number_json_examples, number_json_shots):
    print("start get_task()")
    task_module_name = f"bigbench.benchmark_tasks.{dataset}"
    task_module = importlib.import_module(task_module_name)
    module_path = list(task_module.__path__)[0]
    json_path = os.path.join(module_path, "task.json")
    task_submodule_name = f"{task_module_name}.task"
    if os.path.exists(json_path):
        task = json_task.JsonTask(json_path, shot_list=[number_json_shots], verbose=False, max_examples=number_json_examples)
    else:
        task = util.load_programmatic_task(task_submodule_name)
    print("end get_task()")
    return task


def evaluate_model_task(model, task):
    print("start evaluate")
    score_data = task.evaluate_model(model)
    print("end evaluate")
    return score_data