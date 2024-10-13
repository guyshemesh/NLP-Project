import bigbench.api.model as model
import torch


class PythiaModel(model.Model):
    """Initialize the specified Pythia model."""
    def __init__(self, model_name, model, tokenizer, model_description):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.model_description = model_description
    
    
    """Generates text for given inputs."""
    def generate_text(self, inputs, max_length=100, stop_string=None, output_regex=None):
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    
    """Computes conditional log probabilities of targets given inputs."""
    def cond_log_prob(self, inputs, targets, absolute_normalization=False):
        log_probs = []
        for input_str, target_list in zip(inputs, targets):
            input_ids = self.tokenizer(input_str, return_tensors="pt").input_ids
            input_len = input_ids.shape[1]
            target_log_probs = []

            for target_str in target_list:
                full_input = input_str + target_str
                full_input_ids = self.tokenizer(full_input, return_tensors="pt").input_ids

                with torch.no_grad():
                    output = self.model(full_input_ids, labels=full_input_ids)
                    log_prob = -output.loss.item()

                # We are interested in the log prob of the target completion only, not the input
                target_log_prob = log_prob - input_len  # Subtract log prob of the input sequence
                target_log_probs.append(target_log_prob)

            log_probs.append(target_log_probs)
        
        return log_probs
    

    """Returns ModelData for this Pythia model."""
    def model_data(self):   
        return model.ModelData(
            model_family="Pythia",
            model_name=self.model_name,
            total_params=self.model.num_parameters(),
            non_embedding_params=self.model.num_parameters(),  # self.model.config.n_params - self.model.config.vocab_size * self.model.config.n_embd
            flop_matched_non_embedding_params=self.model.num_parameters(),  # self.model.config.n_params
            training_batch_size=1024,
            training_steps=50000,
            description=self.model_description,
            decoding_params={}
        )
