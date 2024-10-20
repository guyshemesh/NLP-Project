import bigbench_tasks
import pythia_model
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from bootstrapers import bootstrap_model_dirs, bootstrap_tokenizer_dirs, bootstrap_model_names, bootstrap_task_names, \
    bootstrap_number_of_shots
import pandas as pd

def load_pythia_model(model_name, model_full_path, tokenizer_full_dir):
    print("start load_pythia_model()")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_full_dir)
    model = GPTNeoXForCausalLM.from_pretrained(model_full_path)
    print("end load_pythia_model() and start PythiaModel")
    return pythia_model.PythiaModel(model_name, model, tokenizer)


def run_pipeline(model_dirs, tokenizer_dirs, model_names, task_names, number_of_shots):
    results_df = pd.DataFrame(columns=['base_model', 'model_name', 'task_name', 'score_data'])
    for task_name in task_names:
        for base_model, model_dir in model_dirs.items():
            for model_name in model_names[base_model]:
                if base_model == 'pythia_small' and model_name == 'original_pythia':  # TODO delete this
                    print(f"Model: {model_name}")
                    model_full_path = f"{model_dir}/{model_name}"
                    tokenizer_full_dir = tokenizer_dirs[base_model]
                    task = bigbench_tasks.get_task(dataset=task_name, number_json_examples=100,
                                                   number_json_shots=number_of_shots)  # TODO delete 100 and write None/big number instead!
                    model = load_pythia_model(model_name, model_full_path, tokenizer_full_dir)
                    score_data = bigbench_tasks.evaluate_model_task(model, task)
                    new_row = pd.DataFrame([{'base_model': base_model, 'model_name': model_name, 'task_name': task_name, 'score_data': score_data}])
                    results_df = pd.concat([results_df, new_row], ignore_index=True)
                    print(score_data)
                    # return  # TODO delete this
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
