import json


def remove_unwanted_elements(file_path, output_file_path):
    # Read the original JSON file
    with open(file_path, "r") as file:
        data = json.load(file)

    # Count the number of JSON objects
    num_objects = len(data)
    print(f"Number of JSON objects in the file: {num_objects}")

    # Remove 'id' and 'cuisine' elements from each object
    for item in data:
        item.pop("id", None)
        item.pop("cuisine", None)

    # Write the modified data to a new JSON file
    with open(output_file_path, "w") as file:
        json.dump(data, file, indent=4)


remove_unwanted_elements("ingredients_dataset.json", "cleaned_ingredients_dataset.json")
