def remove_duplicate_lines(input_file, output_file):
    # Open the input file in read mode
    with open(input_file, "r") as f:
        lines_seen = set()  # Set to store unique lines
        lines_to_write = []  # List to store unique lines to be written

        # Iterate over each line in the input file
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespaces
            # Check if the line is not in the set of seen lines
            if line not in lines_seen:
                lines_seen.add(line)  # Add line to set of seen lines
                lines_to_write.append(line)  # Add line to list of unique lines

    # Write unique lines to the output file
    with open(output_file, "w") as f:
        f.write("\n".join(lines_to_write))


input_file = "generated_recipes.txt"
output_file = "generated_ingredients.json"
remove_duplicate_lines(input_file, output_file)
print("Duplicate lines removed. Output written to", output_file)
