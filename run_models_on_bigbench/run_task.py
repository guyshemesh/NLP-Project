import bigbench_tasks
import pythia_model
from transformers import AutoTokenizer, GPTNeoXForCausalLM


def load_pythia_model():
    print("start load_pythia_model()")
    tokenizer = AutoTokenizer.from_pretrained("/home/joberant/NLP_2324b/neoeyal/playground/pythia/tokenizers/pythia-70m-deduped/pythia_tokenizer")
    model = GPTNeoXForCausalLM.from_pretrained("/home/joberant/NLP_2324b/neoeyal/playground/pythia/models/pythia-70m-deduped/pythia_1x4")
    print("end load_pythia_model() and start PythiaModel")
    return pythia_model.PythiaModel("original_pythia", model, tokenizer, "Original Pythia")


def main():
    print("start main()")
    # TODO delete 100 and write None instead!
    task = bigbench_tasks.get_task(dataset="analogical_reasoning", number_json_examples=100, number_json_shots=2)
    model = load_pythia_model()
    score_data = bigbench_tasks.evaluate_model_task(model, task)
    print("end main()")
    print(score_data)


if __name__ == "__main__":
    main()
