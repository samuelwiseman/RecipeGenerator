from openai import OpenAI
import json

def process_data(description, input_data):
    api_key = ""
    client = OpenAI(api_key=api_key)

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": description},
            {"role": "user", "content": input_data},
        ],
        model="gpt-4",
        temperature=0.0,
    )
    return chat_completion.choices[0].message.content

def gpt_convert_toy_data():
    # Read the JSON file 'cleaned_ingredients_dataset.json'
    with open("cleaned_ingredients_dataset.json", "r") as file:
        data = json.load(file)

    # Open the output file in append mode
    with open("recipe_dataset.json", "a") as output_file:
        # Start processing from the 590th item
        start_index = 596

        # Process each item in the JSON array starting from the 590th item
        for index, item in enumerate(data[start_index:], start=start_index):
            ingredients = item.get("ingredients", [])

            description = (
                "Here are some ingredients: " + ", ".join(ingredients) + ". "
                "Please provide a recipe name, required amounts of these ingredients, "
                "and step-by-step cooking instructions. "
                "Only include these details, nothing extra."
            )

            # Ensure the input_data is a JSON string
            input_data_json = json.dumps({"ingredients": ingredients})
            response = process_data(description, input_data_json)

            # Write the response immediately to the file
            json.dump(response, output_file)
            output_file.write("\n")  # New line for each response
            output_file.flush()  # Flush the buffer to ensure data is written to disk

            print(f"Processed item {index + 1}")

    print("Conversion completed. Data written to 'recipe_dataset.json'.")

gpt_convert_toy_data()
