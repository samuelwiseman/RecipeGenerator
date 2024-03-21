import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
import torch

# Load the Parquet file
parquet_file_path = "output.parquet"
df = pd.read_parquet(parquet_file_path)

# Load pre-trained T5 model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("t5-large")
model = T5ForConditionalGeneration.from_pretrained("t5-large").to(device)

# Prepare data from Parquet file
input_texts = df["input"].tolist()
target_texts = df["output"].tolist()

# Fine-tune the model
optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 3  # cycle forwards and backwards through the dataset 3 times
for epoch in range(num_epochs):
    for input_text, target_text in zip(input_texts, target_texts):
        input_ids = tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True
        )["input_ids"].to(device)
        target_ids = tokenizer(
            target_text, return_tensors="pt", padding=True, truncation=True
        )["input_ids"].to(device)

        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Save the fine-tuned model to the new directory
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
