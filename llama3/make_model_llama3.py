import os
from transformers import LlamaConfig, LlamaForCausalLM
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline


# works, but copies first layer 1 time
def transfer_wights_and_save(new_model, original_model):
    # Load the state_dict of the original model
    original_state_dict = original_model.state_dict()

    print(f"keys: {original_state_dict.keys()}")
    # Create a new state_dict for the new model
    new_state_dict = new_model.state_dict()

    # Copy weights from the original model to the new model
    for key in original_state_dict.keys():
        print(f"copying key: {key}")
        if 'layers.0' in key:
            # Copy the first layer twice for the new model (layer 0 and layer 1 in the new model)
            new_state_dict[key.replace('layers.0', 'layers.0')] = original_state_dict[key]
            new_state_dict[key.replace('layers.0', 'layers.1')] = original_state_dict[key]
        elif 'layers.' in key:
            layer_num = int(key.split('.')[2])
            if layer_num > 0:
                # Shift layer indices by 1 to accommodate the extra layer at the beginning
                new_state_dict[key.replace(f'layers.{layer_num}', f'layers.{layer_num + 1}')] = original_state_dict[key]
        else:
            # For all other non-layer parameters (e.g., embeddings, layer norms, etc.)
            new_state_dict[key] = original_state_dict[key]
    # Load the modified state_dict into the new model
    new_model.load_state_dict(new_state_dict)

    # Save the new model
    save_directory = f"./models/llama3/modified_llama3_with_33_layers"
    new_model.save_pretrained(save_directory)

    print(f"new state: {new_state_dict}")
    print(f"old state: {original_state_dict}")
    return new_model


# need to check if works, copies layer_index_to_duplicate layer times_to_duplicate times
def make_and_save_new_model(original_model, config, original_model_name, layer_index_to_duplicate, times_to_duplicate):
    print("starting")
    # Load the state_dict of the original model
    original_state_dict = original_model.state_dict()

    new_config = config
    new_config.num_hidden_layers = config.num_hidden_layers + times_to_duplicate
    new_model = LlamaForCausalLM(new_config)

    # Create a new state_dict for the new model
    new_state_dict = new_model.state_dict()

    # Copy weights from the original model to the new model
    for key in original_state_dict.keys():
        print(f"copying key: {key}")
        if 'layers.' in key:
            layer_num = int(key.split('.')[2])
            if layer_index_to_duplicate == layer_num:
                for i in range(times_to_duplicate + 1):
                    # duplicate layer_index_to_duplicate times_to_duplicate times into correct place in new model
                    new_key = key.replace(f'layers.{layer_index_to_duplicate}', f'layers.{layer_index_to_duplicate + i}')
                    new_state_dict[new_key] = original_state_dict[key]
            else:
                if layer_num > layer_index_to_duplicate:
                    # Shift layer indices by times_to_duplicate to accommodate the extra layer at the beginning
                    new_key = key.replace(f'layers.{layer_num}', f'layers.{layer_num + times_to_duplicate}')
                    new_state_dict[new_key] = original_state_dict[key]
                if layer_num < layer_index_to_duplicate:
                    new_state_dict[key.replace(f'layers.{layer_num}', f'layers.{layer_num}')] = original_state_dict[key]
        else:
            # For all other non-layer parameters (e.g., embeddings, layer norms, etc.)
            new_state_dict[key] = original_state_dict[key]
    # Load the modified state_dict into the new model
    new_model.load_state_dict(new_state_dict)

    # Save the new model
    save_directory = f"./models/{original_model_name}/llama3_{layer_index_to_duplicate}x{times_to_duplicate}"
    new_model.save_pretrained(save_directory)



def generate_text(model, tokenizer):
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    prompt = "I have tomatoes, basil, and cheese at home. What can I cook for dinner?"
    sequences = text_generator(prompt, max_length=100, num_return_sequences=1)
    for seq in sequences:
        print(seq['generated_text'])


def main():
    print("------------------------------------start making custom model------------------------------------")

    login(token='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')
    original_model_name = "Meta-Llama-3-8B-Instruct"
    original_model_name_full_name = f"meta-llama/{original_model_name}"

    # Print the model's configuration
    config = LlamaConfig.from_pretrained(original_model_name_full_name)
    print(f"Model Configuration: {config}")

    tokenizer = AutoTokenizer.from_pretrained(original_model_name_full_name)
    save_directory = f"./tokenizers/{original_model_name}/llama3_tokenizer"
    tokenizer.save_pretrained(save_directory)

    original_model = LlamaForCausalLM.from_pretrained(original_model_name_full_name)
    save_directory = f"./models/{original_model_name}/original_llama3"
    original_model.save_pretrained(save_directory)

    # generate_text(original_model, tokenizer)
    new_config = config
    new_config.num_hidden_layers = config.num_hidden_layers + 1
    new_model = LlamaForCausalLM(new_config)
    print(f"new model config is {new_model.config}")

    # simple copy, first layer 1 time- works
    new_model = transfer_wights_and_save(new_model, original_model)
    generate_text(new_model, tokenizer)

    # copies we want- need to check if works. probably doesnt and needs to be run and see if errors,
    # even if runs smoothly need to check if the new model is build how we want
    times_to_duplicate_list = [1, 2, 4, 8, 16]
    for layer_index_to_duplicate in range(config.num_hidden_layers):
        for times_to_duplicate in times_to_duplicate_list:
            save_directory = f"./models/{original_model_name}/llama3_{layer_index_to_duplicate}x{times_to_duplicate}"
            # if we need a few runs to finish the job
            if not os.path.exists(save_directory):
                make_and_save_new_model(original_model, config, original_model_name, layer_index_to_duplicate, times_to_duplicate)
    print("------------------------------------done making custom model------------------------------------")

if __name__ == "__main__":
    main()
