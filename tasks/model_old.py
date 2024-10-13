import bigbench.api.model as model
import bigbench.models.model_utils as model_utils
import torch


def escape_markdown_chars(s):
    # replace markdown special characters
    replace_chars = r"\*_{}[]<>()#+-.!|`"
    for char in replace_chars:
        s = s.replace(char, "\\" + char)
    # replace leading and trailing spaces with nonbreaking spaces, for proper rendering of spaces in markdown
    s_s = s.lstrip()
    s = "&numsp;" * (len(s) - len(s_s)) + s_s
    s_s = s.rstrip()
    s = s_s + "&numsp;" * (len(s) - len(s_s))
    return s


def format_quote(q):
    new_lines = [escape_markdown_chars(line) for line in q.splitlines()]
    return "\\\n".join(new_lines)


class PythiaModel(model.Model):
    """Initialize the specified Pythia model."""
     def __init__(self, model_name, model, tokenizer, model_description, device="cuda"):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.model_description = model_description
        self.device = device
        self.transcript = """## Human readable transcript of prompts and responses with PythiaModel
        Transcript truncated if longer than approximately 50,000 characters.
        """

    
    def add_to_transcript(self, inputs, outputs, targets=None):
        if len(self.transcript) > 5e4:
            return

        if isinstance(inputs, list):
            if targets is None:
                targets = (None,) * len(outputs)
            for input, output, target in zip(inputs, outputs, targets):
                self.add_to_transcript(input, output, target)
            return
        elif not isinstance(inputs, str):
            raise ValueError("inputs has unexpected type %s" % type(inputs))

        inputs = format_quote(inputs)
        self.transcript += f"\n\n### Prompt:\n{inputs}"
        if isinstance(outputs, str):
            out_str = format_quote(outputs)
        else:
            out_str = f"""
| log probability | output string |
| --------------- | ------------- |
"""
            for output, target in zip(outputs, targets):
                out_str += f"| {output} | {target} |\n"

        self.transcript += f"\n\n### Pythia model response:\n{out_str}"
    
    
    """Generates text for given inputs."""
    def generate_text(self, inputs, max_length=100, stop_string=None, output_regex=None):
        self.model.eval()  # ?
        if isinstance(inputs, str):
            inputs = [inputs]  # If input is a single string, process it as a single prompt

        encoded_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.generate(**encoded_inputs, max_length=max_length)

        generated_text = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        if isinstance(inputs, str):
            generated_text = generated_text[0]

        # Apply post-processing
        processed_text = model_utils.postprocess_output(generated_text, max_length, stop_string, output_regex)

        self.add_to_transcript(inputs, processed_text)

        return processed_text

    
    """Computes conditional log probabilities of targets given inputs."""
    def cond_log_prob(self, inputs, targets, absolute_normalization=False):
        if isinstance(inputs, str):
            inputs = [inputs]  # If input is a single string, process it as a single prompt
        if isinstance(targets[0], str):
            targets = [targets]  # Ensure targets match input structure

        self.model.eval()
        log_probs_list = []  # Store log probabilities for each input-target pair

        # Tokenize inputs and targets
        for input_text, target_list in zip(inputs, targets):
            input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.device).input_ids
            target_probs = []
            
            # Compute log probabilities for each target
            for target in target_list:
                target_ids = self.tokenizer(target, return_tensors="pt").to(self.device).input_ids
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, labels=target_ids)
                    log_probs = outputs.logits.softmax(dim=-1)
                    target_log_prob = log_probs.gather(-1, target_ids).log().sum().item()
                    target_probs.append(target_log_prob)
            
            # Normalize probabilities if required
            if not absolute_normalization:
                norm_factor = scipy.special.logsumexp(target_probs)
                target_probs = [p - norm_factor for p in target_probs]

            log_probs_list.append(target_probs)

        self.add_to_transcript(inputs, log_probs_list, targets)
        return log_probs_list
    

    """Returns ModelData for this Pythia model."""
    def model_data(self):        
        return model.ModelData(
            model_family="Pythia",
            model_name=self.model_name,
            total_params=self.model.num_parameters()  #self.model.config.n_params,
            non_embedding_params=self.model.config.n_params - self.model.config.vocab_size * self.model.config.n_embd,
            flop_matched_non_embedding_params=self.model.config.n_params, # Assuming no sparsity for Pythia
            training_batch_size=1024,  # Example value
            training_steps=500000,    # Example value
            description=self.model_description,
            decoding_params={"top_k": 50, "top_p": 0.95, "max_length": 100}  # ?
        )
