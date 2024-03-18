import json
from openai import OpenAI


def process_data(description, input_data):
    api_key = "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"
    client = OpenAI(api_key=api_key)

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": description},
            {"role": "user", "content": input_data},
        ],
        model="gpt-3.5-turbo-0125",  # gpt3 turbo
        temperature=0.0,
    )
    return chat_completion.choices[0].message.content


def gpt_convert_toy_data():
    # Read JSON objects from generated_data.json
    with open("generated_data.json", "r") as file:
        data = json.load(file)

    # Process each JSON object and collect responses
    generated_outputs = []
    for i, item in enumerate(data, start=1):
        description = (
            "This is a json data structure containing a list of ingredients: "
            f"{json.dumps(item)}. "
            "You must convert this data into a human readable recipe containing these ingredients, "
            "combining them with other suitable ones. Can you provide the required additional ingredients "
            "and the instructions required to cook the recipe? "
            "Do not include any extra descriptions, just the result of your conversion in JSON format"
        )
        input_data = json.dumps(item)  # Convert object to JSON string
        response = process_data(description, input_data)
        generated_outputs.append(response)
        print(response)
        print("------------------------------------------")

    # Write responses to generated_outputs.json
    with open("generated_data_recipes.json", "w") as output_file:
        json.dump(generated_outputs, output_file, indent=4)

    print("Generated outputs have been written to generated_outputs.json")


gpt_convert_toy_data()
