from sklearn.neural_network import MLPClassifier
from deap import base, creator, tools, algorithms
import random
import numpy as np

def evolutionary_training(X_train, y_train):
    """
    Optimize neural network architecture using evolutionary algorithms.
    """
    def evaluate(individual):
        hidden_layer_sizes = tuple(individual)
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=300)
        model.fit(X_train, y_train)
        accuracy = model.score(X_train, y_train)
        return accuracy,

    # DEAP setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 5, 50)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=5, up=50, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run evolutionary algorithm
    population = toolbox.population(n=10)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)
