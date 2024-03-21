import json
from transformers import T5ForConditionalGeneration, T5Tokenizer

def main():
    # Load the fine-tuned model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained("fine_tuned_model")
    tokenizer = T5Tokenizer.from_pretrained("fine_tuned_model")

    # Prompt user for ingredients input
    ingredients_input = input("Enter the ingredients separated by commas: ")

    # Tokenize the input text directly
    input_ids = tokenizer.encode(ingredients_input, return_tensors="pt", max_length=512, truncation=True)

    # Generate text based on the input
    output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, early_stopping=False)

    # Decode the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Print the generated text
    print("Generated text:", generated_text)

if __name__ == "__main__":
    main()
