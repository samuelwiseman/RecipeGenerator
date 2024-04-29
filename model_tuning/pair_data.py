import json


def format_data(ingredients_file, recipes_file, output_file):
    # Load the ingredients data
    with open(ingredients_file, "r") as file:
        ingredients_data = json.load(file)

    # Load the recipe data
    with open(recipes_file, "r") as file:
        recipes_data = file.readlines()
        recipes_data = [line.strip() for line in recipes_data]

    # Ensure both files have the same number of entries
    if len(ingredients_data) != len(recipes_data):
        raise ValueError("The number of ingredients and recipes must be the same")

    # Combine the ingredients and recipes into input-output pairs
    formatted_data = []
    for ingredients, recipe in zip(ingredients_data, recipes_data):
        input_text = (
            "Ingredients: "
            + ", ".join(ingredients["ingredients"])
            + ". Generate a recipe."
        )
        output_text = recipe
        formatted_data.append({"input": input_text, "output": output_text})

    # Save the formatted data to a JSON file
    with open(output_file, "w") as file:
        json.dump(formatted_data, file, indent=4)


format_data(
    "ingredients\shrunk_ingredients_dataset.json",
    "recipes/recipe_dataset.json",
    "model_tuning/formatted_data_for_t5.json",
)
