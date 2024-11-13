def bootstrap_model_names():
    duplication_factors = [1, 2, 4, 8, 16]

    pythia_small_num_layers = 6
    pythia_small_models_names = ['original_pythia']
    for duplication_factor in duplication_factors:
        for layer_to_dup in range(pythia_small_num_layers):
            pythia_small_models_names.append(f"pythia_{layer_to_dup}x{duplication_factor}")

    pythia_big_num_layers = 32
    pythia_big_models_names = ['original_pythia']
    for duplication_factor in duplication_factors:
        for layer_to_dup in range(pythia_big_num_layers):
            pythia_big_models_names.append(f"pythia_{layer_to_dup}x{duplication_factor}")

    llama3_num_layers = 32
    llama3_models_names = ['original_llama3']
    for duplication_factor in duplication_factors:
        for layer_to_dup in range(llama3_num_layers):
            llama3_models_names.append(f"llama3_{layer_to_dup}x{duplication_factor}")

    model_names = {
        'pythia_small': pythia_small_models_names,
        'pythia_big': pythia_big_models_names,
        'llama3': llama3_models_names
    }
    return model_names


def bootstrap_model_dirs():
    model_dirs = {
        'pythia_small': '/home/joberant/NLP_2324b/neoeyal/playground/pythia/models/pythia-70m-deduped',
        'pythia_big': '/home/joberant/NLP_2324b/neoeyal/playground/pythia/models/pythia-6.9b-v0',
        'llama3': '/home/joberant/NLP_2324b/neoeyal/playground/llama3/models/Meta-Llama-3-8B-Instruct'
    }
    return model_dirs


def bootstrap_tokenizer_dirs():
    tokenizer_dirs = {
        'pythia_small': '/home/joberant/NLP_2324b/neoeyal/playground/pythia/tokenizers/pythia-70m-deduped/pythia_tokenizer',
        'pythia_big': '/home/joberant/NLP_2324b/neoeyal/playground/pythia/tokenizers/pythia-6.9b-v0/pythia_tokenizer',
        'llama3': '/home/joberant/NLP_2324b/neoeyal/playground/llama3/tokenizers/Meta-Llama-3-8B-Instruct/llama3_tokenizer'
    }
    return tokenizer_dirs


def bootstrap_task_names():
    return ["fantasy_reasoning", "analogical_similarity", "codenames", "conceptual_combinations", "elementary_math_qa",
            "discourse_marker_prediction", "boolean_expressions", "causal_judgment", "date_understanding"]


def bootstrap_number_of_shots():
    return 2
