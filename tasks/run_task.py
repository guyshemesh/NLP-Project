import bigbench_tasks


def load_model():
    model_directory = "pythia/models/pythia-6.9b-v0/original_pythia/"
    tokenizer_directory = "pythia/tokenizers/pythia-6.9b-v0/pythia_tokenizer/"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory)
    model = AutoModelForCausalLM.from_pretrained(model_directory, torch_dtype=torch.float16)
    return model


def main():
    task = bigbench_tasks.get_task(dataset="elementary_math_qa", number_json_examples=None, number_json_shots=2)
    model = load_model()
    score_data = bigbench_tasks.evaluate_model_task(model, task)
    print(score_data)


if __name__ == "__main__":
    main()
