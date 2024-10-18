import bigbench.api.model as model
import re
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from torch.nn.functional import log_softmax


class PythiaModel(model.Model):
    """Initialize the specified Pythia model."""
    def __init__(self, model_name, model, tokenizer, model_description):
        super().__init__()
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.model_description = model_description
    
    
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
        # print("start cond_log_prob()")
        # log_probs = []
        # for input_str, target_list in zip(inputs, targets):
        #     input_ids = self.tokenizer(input_str, return_tensors="pt").input_ids
        #     input_len = input_ids.shape[1]
        #     target_log_probs = []

        #     for target_str in target_list:
        #         full_input = input_str + target_str
        #         full_input_ids = self.tokenizer(full_input, return_tensors="pt").input_ids

        #         with torch.no_grad():
        #             output = self.model(full_input_ids, labels=full_input_ids)
        #             log_prob = -output.loss.item()

        #         # We are interested in the log prob of the target completion only, not the input
        #         target_log_prob = log_prob - input_len  # Subtract log prob of the input sequence
        #         target_log_probs.append(target_log_prob)

        #     log_probs.append(target_log_probs)
        
        # print("end cond_log_prob()")
        # return log_probs
        
        # if isinstance(inputs, str):
        #     inputs = [inputs]
        # if isinstance(targets[0], str):
        #     targets = [targets]

        # log_probs_list = []

        # for input_text, target_list in zip(inputs, targets):
        #     # Tokenize the input and get input length
        #     input_tokens = self.tokenizer(input_text, return_tensors="pt").input_ids
        #     input_len = input_tokens.size(1)

        #     target_log_probs = []

        #     for target in target_list:
        #         # Tokenize the target and concatenate with input tokens
        #         target_tokens = self.tokenizer(target, return_tensors="pt").input_ids
        #         combined_tokens = torch.cat([input_tokens, target_tokens], dim=1)
        #         combined_tokens = combined_tokens.to(self.model.device)  # ?

        #         # Get logits from the model
        #         with torch.no_grad():
        #             output = self.model(combined_tokens)
        #             if isinstance(output, tuple):
        #                 logits = output[0]
        #             else:
        #                 logits = output.logits

            #     # Extract relevant logits for the target tokens
            #     target_logits = logits[:, input_len - 1:-1, :]  # (1, target_len, vocab_size)

            #     # Calculate log probabilities along the vocab dimension
            #     log_probs = log_softmax(target_logits, dim=-1)  # (1, target_len, vocab_size)

            #     # Gather the log probabilities for the correct target tokens
            #     selected_log_probs = log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)  # (1, target_len)

            #     # Sum the log probabilities to get total log prob for this target
            #     total_log_prob = selected_log_probs.sum().item()

            #     if absolute_normalization:
            #         total_log_prob /= target_tokens.size(1)

                # target_log_probs.append(total_log_prob)

        #     log_probs_list.append(target_log_probs)

        # return log_probs_list

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
        # return model.ModelData(
        #     model_family="Pythia",
        #     #model_name=self.model_name,
        #     model_name="test",
        #     total_params=self.model.num_parameters(),
        #     non_embedding_params=self.model.num_parameters(),  # self.model.config.n_params - self.model.config.vocab_size * self.model.config.n_embd
        #     flop_matched_non_embedding_params=self.model.num_parameters(),  # self.model.config.n_params
        #     training_batch_size=1024,
        #     training_steps=50000,
        #     # description=self.model_description,
        #     description="test",
        #     decoding_params={}
        # )
        return model.ModelData()
