import json

# Define the file path
file_path = "generated_recipes.txt"
output_json_file = "generated_recipes.json"

# Initialize an empty list to store recipes
recipes = []

# Open the file and read its contents
with open(file_path, "r") as file:
    # Read the entire content of the file
    content = file.read()

    # Split the content based on the "Recipe Name:" delimiter
    recipe_parts = content.split("Recipe Name:")

    # Iterate over each part (excluding the first one since it's empty)
    for part in recipe_parts[1:]:
        # Append the recipe to the list, removing leading and trailing whitespaces
        recipes.append(part.strip())

# Write recipes to a JSON file
with open(output_json_file, "w") as json_file:
    json.dump(recipes, json_file, indent=4)

print("Recipes saved to generated_recipes.json.")
