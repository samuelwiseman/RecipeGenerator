import json

def trim_json_array(file_path, output_file_path, desired_length):
    # Read the original JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Trim the array to the desired length
    trimmed_data = data[:desired_length]

    # Write the trimmed data to a new JSON file
    with open(output_file_path, 'w') as file:
        json.dump(trimmed_data, file, indent=4)

trim_json_array('cleaned_ingredients_dataset.json', 'shrunk_ingredients_dataset.json', 1033)
