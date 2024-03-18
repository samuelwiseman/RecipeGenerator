import pandas as pd
import json

# Load generated_data.json
with open("generated_data.json", "r") as f:
    data = json.load(f)

# Load generated_outputs.json
with open("generated_data_recipes.json", "r") as f:
    outputs = json.load(f)

# Create a DataFrame to store the data
df = pd.DataFrame(columns=["input", "output"])

# Iterate over each pair of data and output
for input_data, output_data in zip(data, outputs):
    # Convert input json into flat string
    input_str = json.dumps(input_data)
    # Append a row to the DataFrame
    df = pd.concat(
        [df, pd.DataFrame({"input": [input_str], "output": [output_data]})],
        ignore_index=True,
    )
    # Print the added pair
    print("Added pair:")
    print("Input:", input_str)
    print("Output:", output_data)
    print()

# Save the DataFrame to a Parquet file
df.to_parquet("output.parquet", index=False)
