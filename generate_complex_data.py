from openai import OpenAI


def gpt_generate_toy_data():
    api_key = "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"
    client = OpenAI(
        api_key=api_key,
    )
    prompt = {
        "role": "user",
        "content": "I need your help to generate data points for my data set. "
        "For the data set, I need toy examples of json data structures,"
        "with different fundamental ingredients. This data will "
        "be used to train a data to text model for generating recipes from ingredients leftover "
        "Here is an example: {'ingredients': ['chicken']}. The array can be of length 1 or more."
        "The data points should be diverse to support multi shot capabilities."
        "Please respond with only json objects in an array, and remove and natural language in your response.",
    }
    chat_completion = client.chat.completions.create(
        messages=[prompt],
        model="gpt-3.5-turbo",  # gpt3 turbo
        temperature=0.0,
    )
    return chat_completion.choices[0].message.content


def write_to_file(content, filename="generated_data.json"):
    with open(filename, "a") as file:
        file.write(content)


for i in range(96):
    # Generate data
    generated_data = gpt_generate_toy_data()
    print(f"Iteration: {i}")
    # Append to the same file
    write_to_file(generated_data, "generated_data.json")
