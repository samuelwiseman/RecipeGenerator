from openai import OpenAI


def gpt_generate_toy_data():
    api_key = ""
    client = OpenAI(
        api_key=api_key,
    )
    prompt = {
        "role": "user",
        "content": "I need your help to generate data points for my data set. "
        "For the data set, I need toy examples in a string data structure,"
        "with different fundamental ingredients. This data will "
        "be used to train a data to text model for generating recipes from ingredients given by the user "
        "Here is an example: 'chicken, rice'."
        "The data points should be diverse to support multi shot capabilities."
        "Please respond with only ingredients in a string, and remove and natural language in your response.",
    }
    chat_completion = client.chat.completions.create(
        messages=[prompt],
        model="gpt-4-0613",  # gpt4
        temperature=0.0,
    )
    return chat_completion.choices[0].message.content


def write_to_file(content, filename="generated_recipes.txt"):
    with open(filename, "a") as file:
        file.write(content)


for i in range(250):
    # Generate data
    generated_data = gpt_generate_toy_data()
    print(f"Iteration: {i}")
    # Append to the same file
    write_to_file(generated_data, "generated_recipes.txt")
