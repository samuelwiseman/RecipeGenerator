from collections import defaultdict
import random


class MarkovChain:
    def __init__(self):
        self.transitions = defaultdict(list)

    def train(self, recipes):
        for recipe in recipes:
            for i in range(len(recipe) - 1):
                self.transitions[recipe[i]].append(recipe[i + 1])

    def suggest_next_ingredient(self, ingredient):
        possible_ingredients = self.transitions.get(ingredient, [])
        if not possible_ingredients:
            return None
        return random.choice(possible_ingredients)
