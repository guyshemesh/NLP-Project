from dataclasses import dataclass
from typing import List


@dataclass
class ResultEntry:
    base_model_name: str
    model_name: str
    task: str
    scores: List


def print_results(results):
    for result in results:
        print(f"{result.base_model_name}, {result.model_name} on the task: {result.task} gave: {result.scores}")
