from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, Dataset, DatasetDict
import json
import torch
import re

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\\n', '\n', text)
    return text

def load_dataset_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    data = {'input': [clean_text(item['input']) for item in data], 'output': [clean_text(item['output']) for item in data]}
    dataset = Dataset.from_dict(data)
    dataset = dataset.train_test_split(test_size=0.1)
    return dataset

def tokenize_data(examples):
    model_inputs = tokenizer(examples['input'], padding='max_length', truncation=True, max_length=512)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['output'], padding='max_length', truncation=True, max_length=512)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)

file_path = 'model_tuning/formatted_data_for_t5.json'
dataset = load_dataset_from_json(file_path)
tokenized_dataset = dataset.map(tokenize_data, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=30,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    eval_steps=100,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')

print("Model training complete and saved to './trained_model'")
