import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()

model_path = './trained_model'
print("Loading model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
print("Model loaded successfully.")

def clean_text(text):
    text = re.sub(r'00e9edn|nndos|u00e[\da-f]{2}', '', text)  # Clean specific encoding issues
    return text

def generate_recipe(ingredients):
    print(f"Generating recipe for: {ingredients}")
    input_text = f"Ingredients: {ingredients}. Generate a recipe."
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=200,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            temperature=0.9,
            do_sample=True
        )[0]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    cleaned_output = clean_text(output)
    return cleaned_output

parser = argparse.ArgumentParser(description='Generate a recipe based on input ingredients.')
parser.add_argument('ingredients', type=str, help='Ingredients separated by commas')
args = parser.parse_args()

recipe = generate_recipe(args.ingredients)
print("Generated Recipe:")
print(recipe)
