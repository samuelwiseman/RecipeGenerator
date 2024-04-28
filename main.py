import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
from collections import defaultdict
import random
import json
import openai
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

class MarkovChain:
    def __init__(self):
        self.transitions = defaultdict(list)

    def train(self, recipes):
        for recipe in recipes:
            for i in range(len(recipe) - 1):
                self.transitions[recipe[i]].append(recipe[i + 1])

    def suggest_next_ingredient(self, ingredient):
        possible_ingredients = self.transitions.get(ingredient, [])
        if not possible_ingredients:
            return None
        return random.choice(possible_ingredients)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def clean_text(text):
    text = re.sub(r'00e9edn|nndos|u00e[\da-f]{2}', '', text)
    return text

def generate_recipe(ingredients, model, tokenizer, device):
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
    return clean_text(tokenizer.decode(output_ids, skip_special_tokens=True))

def enhance_ingredients(user_ingredients, markov_chain, num_suggestions=3):
    original_set = set(user_ingredients)
    enhanced_ingredients = set(user_ingredients)
    added_ingredients = []
    for ingredient in user_ingredients:
        for _ in range(num_suggestions):
            suggested_ingredient = markov_chain.suggest_next_ingredient(ingredient)
            if suggested_ingredient and suggested_ingredient not in enhanced_ingredients:
                enhanced_ingredients.add(suggested_ingredient)
                added_ingredients.append(suggested_ingredient)
    return list(enhanced_ingredients), added_ingredients

def load_recipes_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return [item['ingredients'] for item in data]

def refine_recipe_with_gpt3(recipe_text, updated_ingredients, openai_api_key):
    openai.api_key = openai_api_key
    ingredients_list = ', '.join(updated_ingredients)
    prompt_text = (
        f"Here is a cooking recipe:\n\n{recipe_text}\n\n"
        f"Here are the ingredients: {ingredients_list}"
        ". Please ensure only the necessary amendments are made. Provide your updated recipe with no additional language."
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant trained to improve cooking recipes by ensuring clarity, coherence, and the use of all listed ingredients."},
            {"role": "user", "content": prompt_text}
        ],
        temperature=0.0
    )
    
    return response['choices'][0]['message']['content']

# Main execution block
device = get_device()
tokenizer = T5Tokenizer.from_pretrained('./trained_model')
model = T5ForConditionalGeneration.from_pretrained('./trained_model').to(device)

recipes = load_recipes_from_file('ingredients/cleaned_ingredients_dataset.json')
markov_chain = MarkovChain()
markov_chain.train(recipes)

parser = argparse.ArgumentParser(description='Generate a recipe based on input ingredients.')
parser.add_argument('ingredients', type=str, help='Ingredients separated by commas')
args = parser.parse_args()

user_ingredients = args.ingredients.split(',')
enhanced_ingredients, added_ingredients = enhance_ingredients(user_ingredients, markov_chain)

recipe = generate_recipe(', '.join(enhanced_ingredients), model, tokenizer, device)

refined_recipe = refine_recipe_with_gpt3(recipe, enhanced_ingredients, api_key) 

if added_ingredients:
    print("")
    print("Suggested Additional Ingredients:")
    print(', '.join(added_ingredients))
    print("")

print("")
print("Generated Recipe:")
print(refined_recipe)
