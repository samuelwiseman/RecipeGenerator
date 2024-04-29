# Recipe Generator

This tool leverages markov chaining, a pre-trained model (T5), and AI to generate cooking recipes based on a set of input ingredients.



## Setup


### Install Python

Ensure Python 3.8 or higher is installed on your machine. You can download it from [python.org](https://www.python.org/downloads/).

### Activate Virtual Environment and Install Dependencies

Navigate to the directory where you extracted and follow the steps below:

1. Create a virtual environment using Python's venv module. Run the following command to create a new virtual environment named env:
```bash
python -m venv env
```

2. Activate the virtual environment by running the activation script appropriate for your operating system:
On Windows:

```bash
.\env\Scripts\activate
```
On macOS and Linux:
```bash
source env/bin/activate
```

3. Once the virtual environment is activated, install the required dependencies using the requirements.txt file:
```bash
pip install -r requirements.txt
```

Once you have successfully completed the above steps, you can proceed to Running the Program.

## Running the Program

Execute the script from the command line by navigating to the project directory and running:

```bash
python3 main.py "ingredient1, ingredient2, ingredient3"
```
Replace "ingredient1, ingredient2, ingredient3" with the actual ingredients you want to use. For example:

```bash
python3 main.py "chicken, tomato, basil"
```

The program will generate a recipe based on the provided ingredients. The recipe will be displayed in the terminal.

