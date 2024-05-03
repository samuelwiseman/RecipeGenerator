from collections import defaultdict, Counter
import random

class MarkovChain:
    def __init__(self):
        # Initialising transitions dictionary with defaultdict of Counter
        self.transitions = defaultdict(Counter)

    def train(self, recipes):
        # Train the Markov Chain model using a list of recipes
        for recipe in recipes:
            for i in range(len(recipe) - 1):
                # Update transitions dictionary with ingredient transitions and frequencies
                self.transitions[recipe[i]][recipe[i + 1]] += 1

    def suggest_next_ingredient(self, ingredient):
        # Suggest the next ingredient based on the given ingredient using weighted probability
        possible_ingredients = self.transitions.get(ingredient, None)
        if not possible_ingredients:
            return None
        
        # Total occurrences of subsequent ingredients to form probability distribution
        total = sum(possible_ingredients.values())
        # Randomly select an ingredient based on the distribution of occurrences
        r = random.uniform(0, total)
        cumulative = 0
        for item, count in possible_ingredients.items():
            cumulative += count
            if r < cumulative:
                return item
