import bigbench.api.model as model
import re
import torch


class LlamaModel(model.Model):
    """Initialize the specified Pythia model."""
    def __init__(self, model_name, model, tokenizer):
        super().__init__()
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = '[PAD]'


    """Generates text for given inputs."""
    def generate_text(self, inputs, max_length=100, stop_string=None, output_regex=None):
        if isinstance(inputs, str):
            inputs = [inputs]

        outputs = []

        for input_text in inputs:
            # Tokenize the input text
            encoding = self.tokenizer.encode_plus(input_text, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            # Generate output
            with torch.no_grad():
                output_ids = self.model.generate(input_ids, attention_mask=attention_mask, max_length=max_length,
                                                 num_return_sequences=1, pad_token_id=self.model.config.pad_token_id)

            # Decode and clean the generated output
            response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

            # Apply stop_string and regex filtering if needed
            if stop_string and stop_string in response:
                response = response.split(stop_string)[0] + stop_string
            if output_regex:
                match = re.search(output_regex, response)
                response = match.group(0) if match else ""

            outputs.append(response)

        return outputs if len(outputs) > 1 else outputs[0]

    
    """Computes conditional log probabilities of targets given inputs."""
    def cond_log_prob(self, inputs, targets, absolute_normalization=False):
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(targets, list) and isinstance(targets[0], str):
            targets = [targets]

        log_probs = []
        for input_text, target_list in zip(inputs, targets):
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
            probs = []
            for target in target_list:
                target_ids = self.tokenizer(target, return_tensors="pt").input_ids
                combined_ids = torch.cat((input_ids, target_ids[:, 1:]), dim=1)
                
                token_log_probs = []
                for i in range(target_ids.size(1)):
                    with torch.no_grad():
                        outputs = self.model(combined_ids[:, :i+1])[0]
                    log_prob = torch.nn.functional.log_softmax(outputs[:, -1, :], dim=-1)
                    if i < target_ids.size(1):
                        token_log_prob = log_prob[0, target_ids[0, i]].item()
                        token_log_probs.append(token_log_prob)
                
                log_prob_sum = sum(token_log_probs)
                probs.append(log_prob_sum)
            log_probs.append(probs)
        
        return log_probs if len(log_probs) > 1 else log_probs[0]
    

    """Returns ModelData for this Pythia model."""
    def model_data(self):
        return model.ModelData(
            model_family="llama3",
            model_name=self.model_name,
            total_params=8000000000,
            non_embedding_params=8000000000,
            flop_matched_non_embedding_params=8000000000,
            training_batch_size=250000,
            training_steps=2000,
            description=self.model_name,
            decoding_params={}
        )
