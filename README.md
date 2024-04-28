# Recipe Generator

This tool leverages markov chaining, a pre-trained model (T5), and AI to generate cooking recipes based on a set of input ingredients.



## Setup


### Install Python

Ensure Python 3.8 or higher is installed on your machine. You can download it from [python.org](https://www.python.org/downloads/).

### Install Dependencies

Navigate to the directory where you extracted the zip file and install the required dependencies:

```bash
pip install -r requirements.txt
```


## Running the Program

Execute the script from the command line by navigating to the project directory and running:

```bash
python main.py "ingredient1, ingredient2, ingredient3"
```
Replace "ingredient1, ingredient2, ingredient3" with the actual ingredients you want to use. For example:

```bash
python main.py "chicken, tomato, basil"
```

The program will generate a recipe based on the provided ingredients. The recipe will be displayed in the terminal.

