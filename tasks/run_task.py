import bigbench_tasks
import pythia_model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_pythia_model():
    print("start load_pythia_model()")
    model_directory = "/home/joberant/NLP_2324b/adishani1/Pythia/models/pythia-6.9b-v0/original_pythia/"
    tokenizer_directory = "/home/joberant/NLP_2324b/adishani1/Pythia/tokenizers/pythia-6.9b-v0/pythia_tokenizer/"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory)
    model = AutoModelForCausalLM.from_pretrained(model_directory, torch_dtype=torch.float16)
    print("end load_pythia_model() and start PythiaModel")
    return pythia_model.PythiaModel("original_pythia", model, tokenizer, "Original Pythia")


def main():
    print("start main()")
    task = bigbench_tasks.get_task(dataset="elementary_math_qa", number_json_examples=None, number_json_shots=2)
    model = load_pythia_model()
    score_data = bigbench_tasks.evaluate_model_task(model, task)
    print("end main()")
    print(score_data)


if __name__ == "__main__":
    main()
