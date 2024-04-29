import openai


def refine_recipe_with_gpt3(recipe_text, updated_ingredients, openai_api_key):
    openai.api_key = openai_api_key
    ingredients_list = ", ".join(updated_ingredients)
    prompt_text = (
        f"Here is a cooking recipe:\n\n{recipe_text}\n\n"
        f"Here are the ingredients: {ingredients_list}. "
        "Please evaluate and improve the recipe to ensure the recipe instructions to ensure they are valid for the ingredients provided. "
        "Provide your updated recipe with no additional language and do not amend the ingredients in the recipe."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant trained to improve cooking recipes.",
                },
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.0,
        )
        refined_recipe = response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"An error occurred with the GPT API call: {e}")
        print("Returning the recipe generated from T5 without revisions.")
        refined_recipe = recipe_text

    return refined_recipe
