import argparse
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
)
import torch
import re
import json
from dotenv import load_dotenv
import os
from markov_chain.markov_chain import (
    MarkovChain,
)
from recipe_evaluator.recipe_evaluator import (
    refine_recipe_with_gpt3,
)

load_dotenv()  # Loading environment variables from .env file
api_key = os.getenv("OPENAI_API_KEY")


def get_device():
    # Function to get the device (GPU or CPU) available for torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def clean_text(text):
    # Function to clean text using regular expressions
    text = re.sub(r"00e9edn|nndos|u00e[\da-f]{2}", "", text)
    return text


def generate_recipe(ingredients, model, tokenizer, device):
    # Function to generate a recipe using the T5 trained model
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
            do_sample=True,
        )[0]
    return clean_text(tokenizer.decode(output_ids, skip_special_tokens=True))


def enhance_ingredients(user_ingredients, markov_chain, num_suggestions=3):
    # Function to enhance ingredients using Markov Chain

    enhanced_ingredients = set(
        user_ingredients
    )  # Initialise enhanced ingredients with user-provided ingredients
    added_ingredients = []  # Initialise list to store added ingredients

    for ingredient in user_ingredients:  # Iterate over each user-provided ingredient
        for _ in range(
            num_suggestions
        ):  # Iterate for the specified number of suggestions
            suggested_ingredient = markov_chain.suggest_next_ingredient(
                ingredient
            )  # Get suggested next ingredient from Markov Chain
            if (
                suggested_ingredient
                and suggested_ingredient not in enhanced_ingredients
            ):
                # If suggested ingredient exists and not already in enhanced ingredients
                enhanced_ingredients.add(
                    suggested_ingredient
                )  # Add suggested ingredient to enhanced ingredients
                added_ingredients.append(
                    suggested_ingredient
                )  # Append suggested ingredient to added ingredients list

    return (
        list(enhanced_ingredients),
        added_ingredients,
    )  # Return enhanced ingredients and added ingredients list


def load_recipes_from_file(file_path):
    # Function to load recipes from a JSON file
    with open(file_path, "r") as file:
        data = json.load(file)
    return [item["ingredients"] for item in data]


if __name__ == "__main__":
    # Main execution block
    device = get_device()  # Get the device (GPU or CPU) available for torch
    tokenizer = T5Tokenizer.from_pretrained(
        "./trained_model"
    )  # Initialise T5 tokenizer
    model = T5ForConditionalGeneration.from_pretrained("./trained_model").to(
        device
    )  # Initialise T5 model

    recipes = load_recipes_from_file(
        "ingredients/cleaned_ingredients_dataset.json"
    )  # Load recipes from a JSON file
    markov_chain = MarkovChain()  # Initialise Markov Chain
    markov_chain.train(recipes)  # Train Markov Chain with the loaded recipes

    parser = argparse.ArgumentParser(
        description="Generate a recipe based on input ingredients."
    )  # Create an argument parser
    parser.add_argument(
        "ingredients", type=str, help="Ingredients separated by commas"
    )  # Add a command-line argument for ingredients
    args = parser.parse_args()  # Parse command-line arguments

    user_ingredients = args.ingredients.split(",")  # Get user-provided ingredients
    enhanced_ingredients, added_ingredients = enhance_ingredients(
        user_ingredients, markov_chain
    )  # Enhance user-provided ingredients using Markov Chain

    recipe = generate_recipe(
        ", ".join(enhanced_ingredients), model, tokenizer, device
    )  # Generate a recipe using T5 model with enhanced ingredients

    refined_recipe = refine_recipe_with_gpt3(
        recipe, enhanced_ingredients, api_key
    )  # Refine the generated recipe using OpenAI's GPT-3 model

    if added_ingredients:
        print("")
        print("Suggested Additional Ingredients:")
        print(", ".join(added_ingredients))
        print("")

    print("")
    print("Generated Recipe:")
    print(refined_recipe)
