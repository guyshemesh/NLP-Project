import os

from transformers import GPTNeoXForCausalLM, AutoTokenizer, GPTNeoXConfig, GPTNeoXModel, GPTNeoXLayer, \
    AutoModelForCausalLM
import torch
import argparse


class CustomGPTNeoXForCausalLM(GPTNeoXForCausalLM):
    """
    class of our custom llama model with num_layers layers
    """

    def __init__(self, config, num_layers):
        super().__init__(config)
        self.gpt_neox.layers = torch.nn.ModuleList(
            [GPTNeoXLayer(config) for _ in range(num_layers)]
        )


def transfer_weights(original_model, new_model, layer_index_to_duplicate, number_of_times_to_duplicate):
    """
    transfers weights from original_model to new_model in place
    such that the new model will be like old_model but with
    layers[layer_index_to_duplicate] duplicated number_of_times_to_duplicate times
    """
    new_model.gpt_neox.embed_in.load_state_dict(original_model.gpt_neox.embed_in.state_dict())
    new_model.gpt_neox.emb_dropout.load_state_dict(original_model.gpt_neox.emb_dropout.state_dict())
    original_layers = original_model.gpt_neox.layers
    new_layers = new_model.gpt_neox.layers
    assert len(original_layers) >= layer_index_to_duplicate + 1
    for i in range(len(original_layers)):
        # adding the layer_index_to_duplicate layer, number_of_times_to_duplicate times
        if i == layer_index_to_duplicate:
            for j in range(number_of_times_to_duplicate + 1):
                new_layers[i + j].load_state_dict(original_layers[i].state_dict())
        # duplicating layers before layer_index_to_duplicate
        elif i < layer_index_to_duplicate:
            new_layers[i].load_state_dict(original_layers[i].state_dict())
        # duplicating layers after layer_index_to_duplicate
        elif i > layer_index_to_duplicate:
            new_layers[i + number_of_times_to_duplicate].load_state_dict(original_layers[i].state_dict())
    new_model.gpt_neox.final_layer_norm.load_state_dict(original_model.gpt_neox.final_layer_norm.state_dict())
    new_model.embed_out.load_state_dict(original_model.embed_out.state_dict())


def are_state_dicts_equal(state_dict1, state_dict2):
    """
    returns true iff the 2 state_dicts are identical
    """
    if state_dict1.keys() != state_dict2.keys():
        return False
    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False
    return True


def check_transfer(original_model, new_model, layer_index_to_duplicate, number_of_times_to_duplicate):
    """
    :return: true iff the transfer of the wights worked
    """
    if not are_state_dicts_equal(new_model.gpt_neox.embed_in.state_dict(),
                                 original_model.gpt_neox.embed_in.state_dict()):
        return False
    if not are_state_dicts_equal(new_model.gpt_neox.emb_dropout.state_dict(),
                                 original_model.gpt_neox.emb_dropout.state_dict()):
        return False
    original_layers = original_model.gpt_neox.layers
    new_layers = new_model.gpt_neox.layers
    for i in range(len(original_layers)):
        # checking duplicated layers
        if i == layer_index_to_duplicate:
            for j in range(number_of_times_to_duplicate + 1):
                if not are_state_dicts_equal(new_layers[i + j].state_dict(), original_layers[i].state_dict()):
                    return False
        # Check layers before the duplicated layer
        elif i < layer_index_to_duplicate:
            if not are_state_dicts_equal(new_layers[i].state_dict(), original_layers[i].state_dict()):
                return False
        # Check layers after the duplicated layer, accounting for the duplicates
        elif i > layer_index_to_duplicate:
            if not are_state_dicts_equal(new_layers[i + number_of_times_to_duplicate].state_dict(),
                                         original_layers[i].state_dict()):
                return False
    if not are_state_dicts_equal(new_model.gpt_neox.final_layer_norm.state_dict(),
                                 original_model.gpt_neox.final_layer_norm.state_dict()):
        return False
    if not are_state_dicts_equal(new_model.embed_out.state_dict(), original_model.embed_out.state_dict()):
        return False
    return True


def make_and_save_new_model(original_model, config, original_model_name, layer_index_to_duplicate,
                            number_of_times_to_duplicate):
    """
    :return: new model with one of the old models layers duplicated
    """

    original_numer_of_layers = config.num_hidden_layers
    new_model = CustomGPTNeoXForCausalLM(config, original_numer_of_layers + number_of_times_to_duplicate)
    # print(f"new model architecture: {new_model}")
    transfer_weights(original_model, new_model, layer_index_to_duplicate, number_of_times_to_duplicate)
    # uncomment if need to check
    # assert (check_transfer(original_model, new_model, layer_index_to_duplicate, number_of_times_to_duplicate))
    # Save the new model
    save_directory = f"./models/{original_model_name}/pythia_{layer_index_to_duplicate}x{number_of_times_to_duplicate}"
    new_model.save_pretrained(save_directory)
    return new_model


"""
argument should be --original_model_name then:
pythia-6.9b-v0 or pythia-70m-deduped
"""
def main():
    print("------------------------------------start making custom model------------------------------------")
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_model_name")
    original_model_name = parser.parse_args().original_model_name
    original_model_full_name = f"EleutherAI/{original_model_name}"
    original_model = GPTNeoXForCausalLM.from_pretrained(
        original_model_full_name,
        cache_dir=original_model_name
    )
    print(f"original model architecture: {original_model}")
    save_directory = f"./models/{original_model_name}/original_pythia"
    original_model.save_pretrained(save_directory)

    tokenizer = tokenizer = AutoTokenizer.from_pretrained(
        original_model_full_name,
        cache_dir=f"./{original_model_name}"
    )
    # Save the tokenizer
    save_directory = f"./tokenizers/{original_model_name}/pythia_tokenizer"
    tokenizer.save_pretrained(save_directory)
    config = GPTNeoXConfig.from_pretrained(original_model_full_name)
    original_model_number_of_layers = len(original_model.gpt_neox.layers)
    times_to_duplicate_list = [1, 2, 4, 8, 16]
    for layer_index_to_duplicate in range(original_model_number_of_layers):
        for times_to_duplicate in times_to_duplicate_list:
            save_directory = f"./models/{original_model_name}/pythia_{layer_index_to_duplicate}x{times_to_duplicate}"
            # if we need a few runs to finish the job
            if not os.path.exists(save_directory):
                make_and_save_new_model(original_model, config, original_model_name, layer_index_to_duplicate,
                                        times_to_duplicate)
    print("------------------------------------done making custom model------------------------------------")


if __name__ == "__main__":
    main()
