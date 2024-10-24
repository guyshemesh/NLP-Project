import bigbench_tasks
import pythia_model
import llama3_model
from bootstrapers import *

from transformers import AutoTokenizer, GPTNeoXForCausalLM, LlamaForCausalLM
import pandas as pd

def load_model(model_name, model_full_path, tokenizer_full_dir):
    print("start load_model()")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_full_dir)
    if 'pythia' in model_name:
        model = GPTNeoXForCausalLM.from_pretrained(model_full_path)
        print("end load_model() and start PythiaModel")
        return pythia_model.PythiaModel(model_name, model, tokenizer)
    else:
        model = LlamaForCausalLM.from_pretrained(model_full_path)
        print("end load_model() and start LlamaModel")
        return llama3_model.LlamaModel(model_name, model, tokenizer)



def run_pipeline(model_dirs, tokenizer_dirs, model_names, task_names, number_of_shots):
    results_df = pd.DataFrame(columns=['base_model', 'model_name', 'task_name', 'score'])
    for task_name in task_names:
        for base_model, model_dir in model_dirs.items():
            for model_name in model_names[base_model]:
                if base_model == 'llama3':  # TODO change this to the models you want this
                    print(f"Model: {model_name}")
                    model_full_path = f"{model_dir}/{model_name}"
                    tokenizer_full_dir = tokenizer_dirs[base_model]
                    task = bigbench_tasks.get_task(dataset=task_name, number_json_examples=None,
                                                   number_json_shots=number_of_shots)  # TODO delete 100 and write None/big number instead!
                    model = load_model(model_name, model_full_path, tokenizer_full_dir)
                    scores_data = bigbench_tasks.evaluate_model_task(model, task)
                    # TODO update next line multiple_choice_grade to the metric/metrics each task has. maybe even(check the spelling): score_data.perfered_metric
                    scores_and_subtasks = [(score_data.score_dict['multiple_choice_grade'], score_data.subtask_description) for score_data in scores_data]
                    for score, sub_task in scores_and_subtasks:
                        new_row = pd.DataFrame([{'base_model': base_model, 'model_name': model_name, 'task_name': task_name + " " + sub_task,
                                                 'score': score}])
                        results_df = pd.concat([results_df, new_row], ignore_index=True)
                    results_df.to_csv('output.csv', index=False)
    return results_df


def main():
    print("start main()")
    model_dirs = bootstrap_model_dirs()
    tokenizer_dirs = bootstrap_tokenizer_dirs()
    model_names = bootstrap_model_names()
    task_names = bootstrap_task_names()
    number_of_shots = bootstrap_number_of_shots()
    results_df = run_pipeline(model_dirs, tokenizer_dirs, model_names, task_names, number_of_shots)
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)  # Show all rows
    print(results_df)
    print("end main()")


if __name__ == "__main__":
    main()
