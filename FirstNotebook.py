# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + colab={"base_uri": "https://localhost:8080/"} id="AKWuf4ySsHoN" outputId="a5668066-931e-4dc2-be49-ca251f772e5d"
# !pip install transformers datasets torch matplotlib evaluate scikit-learn


# + id="ROxphKDfsh8b"
from transformers import BertTokenizer
from datasets import load_dataset
import evaluate

# Load the SST-2, SQuAD, and CoNLL-2003 datasets
dataset_sst2 = load_dataset("glue", "sst2")
dataset_squad = load_dataset("squad")
dataset_conll = load_dataset("conll2003")

# Load metric (accuracy, precision, recall, f1)
metric_accuracy = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")
metric_precision = evaluate.load("precision")
metric_recall = evaluate.load("recall")

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# + id="nFlDjU06sh6H"
def tokenize_sst2(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

tokenized_sst2 = dataset_sst2.map(tokenize_sst2, batched=True)
train_sst2 = tokenized_sst2["train"]
eval_sst2 = tokenized_sst2["validation"]


# + id="mZXpSkhwsh4R"
def tokenize_squad(examples):
    return tokenizer(
        examples["question"], examples["context"], truncation=True, padding="max_length", max_length=384
    )

tokenized_squad = dataset_squad.map(tokenize_squad, batched=True)
train_squad = tokenized_squad["train"]
eval_squad = tokenized_squad["validation"]


# + id="-2RFIzzash2X"
def tokenize_conll(examples):
    return tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

tokenized_conll = dataset_conll.map(tokenize_conll, batched=True)
train_conll = tokenized_conll["train"]
eval_conll = tokenized_conll["validation"]


# + id="rd_CWsRhshz1"
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertForQuestionAnswering, Trainer, TrainingArguments

# Define the modified model class to allow adding different types of layers
class ModifiedBertModel(nn.Module):
    def __init__(self, original_model, additional_attn_layers=0, additional_ff_layers=0, additional_embed_layers=0):
        super(ModifiedBertModel, self).__init__()
        self.bert = original_model.bert  # Use the pre-trained BERT model
        self.dropout = nn.Dropout(0.1)
        self.classifier = original_model.classifier

        # Add additional self-attention layers
        self.extra_attn_layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=768, nhead=12) for _ in range(additional_attn_layers)]
        )

        # Add additional feed-forward layers
        self.extra_ff_layers = nn.ModuleList(
            [nn.Linear(768, 768) for _ in range(additional_ff_layers)]
        )

        # Add additional embedding layers
        self.extra_embed_layers = nn.ModuleList(
            [nn.Embedding(30522, 768) for _ in range(additional_embed_layers)]
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]

        # Pass through additional embedding layers
        for layer in self.extra_embed_layers:
            sequence_output = layer(input_ids)

        # Pass through additional self-attention layers
        for layer in self.extra_attn_layers:
            sequence_output = layer(sequence_output)

        # Pass through additional feed-forward layers
        for layer in self.extra_ff_layers:
            sequence_output = layer(sequence_output)

        pooled_output = sequence_output[:, 0]  # Taking the [CLS] token's representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return (loss, logits) if loss is not None else logits



# + id="kmP4cpuEshxp"
from transformers import Trainer, TrainingArguments
import os
from sklearn.metrics import precision_recall_fscore_support

# Function to compute the metrics (accuracy, precision, recall, f1)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = metric_accuracy.compute(predictions=preds, references=labels)["accuracy"]
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Function to train the model and use checkpointing
def train_with_checkpointing(task_name, model, train_dataset, eval_dataset, output_dir, num_epochs=2):
    # Check if there is an existing checkpoint
    last_checkpoint = None
    if os.path.exists(output_dir):
        checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))  # Get checkpoint with highest step
            print(f"Resuming from checkpoint: {last_checkpoint}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",  # Make eval strategy match save strategy
        save_strategy="steps",  # Set save strategy to steps
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        save_steps=500,  # Save the model every 500 steps
        save_total_limit=3,  # Keep only the last 3 checkpoints
        logging_dir=f"./logs_{task_name}",
        logging_steps=100,
        load_best_model_at_end=True,  # Load the best model after training
        resume_from_checkpoint=last_checkpoint if last_checkpoint else None,  # Resume training from the last checkpoint
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,  # Use the compute_metrics function
    )

    trainer.train(resume_from_checkpoint=last_checkpoint)
    eval_result = trainer.evaluate()
    return eval_result



# + id="trkSMAg7shue"
def train_sst2_with_layers(num_layers, layer_type):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # Decide which type of layer to add
    if layer_type == 'attention':
        modified_model = ModifiedBertModel(model, additional_attn_layers=num_layers)
    elif layer_type == 'ff':
        modified_model = ModifiedBertModel(model, additional_ff_layers=num_layers)
    elif layer_type == 'embedding':
        modified_model = ModifiedBertModel(model, additional_embed_layers=num_layers)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")

    output_dir = f"./sst2_checkpoints_{layer_type}_{num_layers}_layers"
    return train_with_checkpointing("sst2", modified_model, train_sst2, eval_sst2, output_dir)



# + id="OfC9QZGishrn"
def train_squad_with_layers(num_layers, layer_type):
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

    # Decide which type of layer to add
    if layer_type == 'attention':
        modified_model = ModifiedBertModel(model, additional_attn_layers=num_layers)
    elif layer_type == 'ff':
        modified_model = ModifiedBertModel(model, additional_ff_layers=num_layers)
    elif layer_type == 'embedding':
        modified_model = ModifiedBertModel(model, additional_embed_layers=num_layers)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")

    output_dir = f"./squad_checkpoints_{layer_type}_{num_layers}_layers"
    return train_with_checkpointing("squad", modified_model, train_squad, eval_squad, output_dir)



# + id="p-WUzQEjshpL"
def train_conll_with_layers(num_layers, layer_type):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=9)

    # Decide which type of layer to add
    if layer_type == 'attention':
        modified_model = ModifiedBertModel(model, additional_attn_layers=num_layers)
    elif layer_type == 'ff':
        modified_model = ModifiedBertModel(model, additional_ff_layers=num_layers)
    elif layer_type == 'embedding':
        modified_model = ModifiedBertModel(model, additional_embed_layers=num_layers)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")

    output_dir = f"./conll_checkpoints_{layer_type}_{num_layers}_layers"
    return train_with_checkpointing("conll", modified_model, train_conll, eval_conll, output_dir)



# + id="0B2ihC4xshmd"
# Initialize dictionaries to store accuracies for each task and layer type
results = {
    'sst2': {'accuracy': [], 'f1': [], 'precision': [], 'recall': []},
    'squad': {'accuracy': [], 'f1': [], 'precision': [], 'recall': []},
    'conll': {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
}

# Iterate over number of layers (0 to 3 layers) and add different types of layers
for i in range(4):
    # SST-2 Task (Sentiment classification)
    eval_result = train_sst2_with_layers(num_layers=i, layer_type='attention')  # Self-Attention layers for SST-2
    results['sst2']['accuracy'].append(eval_result['eval_accuracy'])
    results['sst2']['f1'].append(eval_result['eval_f1'])
    results['sst2']['precision'].append(eval_result['eval_precision'])
    results['sst2']['recall'].append(eval_result['eval_recall'])

    # SQuAD Task (Question Answering)
    eval_result = train_squad_with_layers(num_layers=i, layer_type='attention')  # Self-Attention layers for SQuAD
    results['squad']['accuracy'].append(eval_result['eval_accuracy'])
    results['squad']['f1'].append(eval_result['eval_f1'])
    results['squad']['precision'].append(eval_result['eval_precision'])
    results['squad']['recall'].append(eval_result['eval_recall'])

    # CoNLL-2003 Task (NER)
    eval_result = train_conll_with_layers(num_layers=i, layer_type='attention')  # Self-Attention layers for CoNLL
    results['conll']['accuracy'].append(eval_result['eval_accuracy'])
    results['conll']['f1'].append(eval_result['eval_f1'])
    results['conll']['precision'].append(eval_result['eval_precision'])
    results['conll']['recall'].append(eval_result['eval_recall'])


# + id="fOxh5Td5shjg"
import matplotlib.pyplot as plt

x = range(4)  # Number of layers (0 to 3)

metrics = ['accuracy', 'f1', 'precision', 'recall']

# SST-2 Task
for metric in metrics:
    # Self-Attention
    plt.figure(figsize=(8, 6))
    plt.plot(x, results['sst2'][metric], marker='o')
    plt.title(f'SST-2: {metric.capitalize()} (Self-Attention)')
    plt.xlabel('Number of Layers')
    plt.ylabel(metric.capitalize())
    plt.show()

    # Feed-Forward
    plt.figure(figsize=(8, 6))
    plt.plot(x, results['sst2'][metric], marker='o')
    plt.title(f'SST-2: {metric.capitalize()} (Feed-Forward)')
    plt.xlabel('Number of Layers')
    plt.ylabel(metric.capitalize())
    plt.show()

    # Embedding
    plt.figure(figsize=(8, 6))
    plt.plot(x, results['sst2'][metric], marker='o')
    plt.title(f'SST-2: {metric.capitalize()} (Embedding)')
    plt.xlabel('Number of Layers')
    plt.ylabel(metric.capitalize())
    plt.show()

# SQuAD Task
for metric in metrics:
    # Self-Attention
    plt.figure(figsize=(8, 6))
    plt.plot(x, results['squad'][metric], marker='o')
    plt.title(f'SQuAD: {metric.capitalize()} (Self-Attention)')
    plt.xlabel('Number of Layers')
    plt.ylabel(metric.capitalize())
    plt.show()

    # Feed-Forward
    plt.figure(figsize=(8, 6))
    plt.plot(x, results['squad'][metric], marker='o')
    plt.title(f'SQuAD: {metric.capitalize()} (Feed-Forward)')
    plt.xlabel('Number of Layers')
    plt.ylabel(metric.capitalize())
    plt.show()

    # Embedding
    plt.figure(figsize=(8, 6))
    plt.plot(x, results['squad'][metric], marker='o')
    plt.title(f'SQuAD: {metric.capitalize()} (Embedding)')
    plt.xlabel('Number of Layers')
    plt.ylabel(metric.capitalize())
    plt.show()

# CoNLL-2003 Task
for metric in metrics:
    # Self-Attention
    plt.figure(figsize=(8, 6))
    plt.plot(x, results['conll'][metric], marker='o')
    plt.title(f'CoNLL-2003: {metric.capitalize()} (Self-Attention)')
    plt.xlabel('Number of Layers')
    plt.ylabel(metric.capitalize())
    plt.show()

    # Feed-Forward
    plt.figure(figsize=(8, 6))
    plt.plot(x, results['conll'][metric], marker='o')
    plt.title(f'CoNLL-2003: {metric.capitalize()} (Feed-Forward)')
    plt.xlabel('Number of Layers')
    plt.ylabel(metric.capitalize())
    plt.show()

    # Embedding
    plt.figure(figsize=(8, 6))
    plt.plot(x, results['conll'][metric], marker='o')
    plt.title(f'CoNLL-2003: {metric.capitalize()} (Embedding)')
    plt.xlabel('Number of Layers')
    plt.ylabel(metric.capitalize())
    plt.show()

