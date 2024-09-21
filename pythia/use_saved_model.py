from transformers import GPTNeoXForCausalLM, AutoTokenizer, GPTNeoXConfig, GPTNeoXModel, GPTNeoXLayer, \
    AutoModelForCausalLM
import torch
import argparse

def generate_response(model, tokenizer, inputs):
    """
    given a model, prompt it for output
    :return:
    """
    tokens = model.generate(**inputs)
    return tokenizer.decode(tokens[0])

"""
argument should be --original_model_name then:
pythia-6.9b-v0 or pythia-70m-deduped
"""
def main():
    print("------------------------------------start prompting model------------------------------------")
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_model_name")
    original_model_name = parser.parse_args().original_model_name

    original_model = GPTNeoXForCausalLM.from_pretrained(f"./models/{original_model_name}/original_pythia")
    custom_model = GPTNeoXForCausalLM.from_pretrained(f"./models/{original_model_name}/pythia_4x1")
    tokenizer = AutoTokenizer.from_pretrained(f"./tokenizers/{original_model_name}/pythia_tokenizer")
    inputs = tokenizer("Once upon a time", return_tensors="pt")
    original_response = generate_response(original_model, tokenizer, inputs)
    new_response = generate_response(custom_model, tokenizer, inputs)
    print(f"original response: {original_response}")
    print(f"new response: {new_response}")
    print("------------------------------------done prompting model------------------------------------")


if __name__ == "__main__":
    main()
