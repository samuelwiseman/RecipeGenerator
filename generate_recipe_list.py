from openai import OpenAI


def process_data(description, input_data):
    api_key = ""
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
    # Read lines from generated_ingredients.json
    with open("generated_ingredients.json", "r") as file:
        data = file.readlines()

    # Process each line and collect responses
    generated_outputs = []
    for i, item in enumerate(data, start=1):
        description = (
            "This is a string data structure representing a list of ingredients: "
            f"{item.strip()}. "
            "You must convert this data into a human readable recipe containing these ingredients, "
            "combining them with other suitable ones. Can you provide the recipe name, ingredients "
            "(with their amount required) and step by step instructions required to cook the recipe? "
            "Do not include any extra descriptions, just the result of your conversion in string format."
        )

        response = process_data(description, item.strip())
        generated_outputs.append(response)
        print(response)
        print("------------------------------------------")

    # Write responses to generated_recipes.json
    with open("generated_recipes.txt", "w") as output_file:
        for output in generated_outputs:
            output_file.write(output + "\n")

    print("Processed outputs have been written to generated_recipes.txt")


gpt_convert_toy_data()
