from transformers import GPTNeoXForCausalLM, AutoTokenizer, GPTNeoXConfig, GPTNeoXModel, GPTNeoXLayer, \
    AutoModelForCausalLM
import torch


def generate_response(model, tokenizer, inputs):
    """
    given a model, prompt it for output
    :return:
    """
    tokens = model.generate(**inputs)
    return tokenizer.decode(tokens[0])


def main():
    print("------------------------------------start prompting model------------------------------------")
    original_model = GPTNeoXForCausalLM.from_pretrained("./models/original_pythia_small")
    custom_model = GPTNeoXForCausalLM.from_pretrained("./models/pythia_small_1x4")
    tokenizer = AutoTokenizer.from_pretrained("./tokenizers/pythia_small_tokenizer")
    inputs = tokenizer("Once upon a time", return_tensors="pt")
    original_response = generate_response(original_model, tokenizer, inputs)
    new_response = generate_response(custom_model, tokenizer, inputs)
    print(f"original response: {original_response}")
    print(f"new response: {new_response}")
    print("------------------------------------done prompting model------------------------------------")


if __name__ == "__main__":
    main()
