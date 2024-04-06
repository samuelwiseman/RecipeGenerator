def add_comma_to_json(file_path, output_file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(output_file_path, 'w') as output_file:
        for line in lines:
            # Add a comma at the end of each line and write it to the output file
            output_file.write(line.rstrip('\n') + ',\n')


add_comma_to_json('recipe_dataset.json', 'model_tuning\cleaned_recipe_dataset.json')
