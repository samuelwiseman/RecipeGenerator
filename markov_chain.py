from collections import defaultdict
import random


class MarkovChain:
    def __init__(self):
        # Initialising transitions dictionary with defaultdict
        self.transitions = defaultdict(list)

    def train(self, recipes):
        # Train the Markov Chain model using a list of recipes.
        for recipe in recipes:
            # Iterate over each recipe in the list of recipes
            for i in range(len(recipe) - 1):
                # Iterate over each ingredient in the recipe
                self.transitions[recipe[i]].append(recipe[i + 1])
                # Update transitions dictionary with ingredient transitions

    def suggest_next_ingredient(self, ingredient):
        # Suggest the next ingredient based on the given ingredient.
        possible_ingredients = self.transitions.get(ingredient, [])
        # Get possible next ingredients for the given ingredient
        if not possible_ingredients:
            # If no possible next ingredients found
            return None
            # Return None
        return random.choice(possible_ingredients)
        # Return a random choice from the list of possible next ingredients
