import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

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
            do_sample=True  # Enable sampling
        )[0]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output

parser = argparse.ArgumentParser(description='Generate a recipe based on input ingredients.')
parser.add_argument('ingredients', type=str, help='Ingredients separated by commas')
args = parser.parse_args()

recipe = generate_recipe(args.ingredients)
print("Generated Recipe:")
print(recipe)
