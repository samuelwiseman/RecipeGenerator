from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset
import json
import torch
import re


# Function to determine and return the computational device (GPU or CPU)
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


device = get_device()  # Set the device for model computation


# Function to clean text by removing non-ASCII characters and fixing newline issues
def clean_text(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII characters
    text = re.sub(r"\\n", "\n", text)  # Replace literal '\n' with actual newlines
    return text


# Function to load and clean a dataset from a JSON file, then split it into training and test sets
def load_dataset_from_json(file_path):
    with open(file_path, "r") as file:  # Open the file in read mode
        data = json.load(file)  # Load data from the JSON file
    data = {
        "input": [clean_text(item["input"]) for item in data],  # Clean 'input' data
        "output": [clean_text(item["output"]) for item in data],  # Clean 'output' data
    }
    dataset = Dataset.from_dict(data)  # Create a dataset from the cleaned data
    dataset = dataset.train_test_split(
        test_size=0.1
    )  # Split dataset into training and test sets
    return dataset


# Function to tokenize the dataset using the T5 tokenizer
def tokenize_data(examples):
    model_inputs = tokenizer(
        examples["input"], padding="max_length", truncation=True, max_length=512
    )
    with tokenizer.as_target_tokenizer():  # Switch tokenizer context to target text
        labels = tokenizer(
            examples["output"], padding="max_length", truncation=True, max_length=512
        )
    model_inputs["labels"] = labels["input_ids"]  # Attach tokenized outputs as labels
    return model_inputs


# Initialize the T5 tokenizer and model, and move the model to the appropriate device
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

# Load and tokenize the dataset
file_path = "model_tuning/formatted_data_for_t5.json"
dataset = load_dataset_from_json(file_path)
tokenized_dataset = dataset.map(tokenize_data, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save the results
    num_train_epochs=30,  # Number of training epochs
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=8,  # Batch size for evaluation
    warmup_steps=500,  # Number of warmup steps
    weight_decay=0.01,  # Weight decay for regularization
    logging_dir="./logs",  # Directory for logs
    logging_steps=10,  # Frequency of logging
    save_strategy="epoch",  # Saving model at the end of each epoch
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    eval_steps=100,  # Steps between each evaluation
    load_best_model_at_end=True,  # Load the best model at the end of training based on evaluation
)

# Initialize the Trainer with the model, training arguments, datasets, and early stopping callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()  # Start training

# Save the trained model and tokenizer
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

print("Model training complete and saved to './trained_model'")  # Confirmation message
