import random
from random import randint, sample, uniform
import numpy as np
import statistics


def geometric_mutation(individual,ms=0.1):
    for layer in range(len(individual)):
        for elem in range(0, len(individual[layer])):
            r = random.uniform(-ms,ms)
            individual[layer][elem] += r
    return individual

def gaussian_mutation(individual, **kwargs):
    for layer in range(len(individual)):
        if len(individual[layer]) > 0:
            ms = statistics.stdev(individual[layer])
            for elem in range(len(individual[layer])):
                mutation = np.random.normal(loc=0.0, scale=ms)
                individual[layer][elem] += mutation
    return individual


def custom_swap_mutation(individual, **kwargs):
    for layer in range(len(individual)):
        if len(individual[layer]) > 0:
            swaps_count = int(len(individual[layer]) * 0.05)
            for _ in range(swaps_count):
                mut_indexes = sample( range(len(individual[layer])),2)
                individual[layer][mut_indexes[0]], individual[layer][mut_indexes[1]] = individual[layer][mut_indexes[1]], individual[layer][mut_indexes[0]]
    return individual

def inversion_mutation(individual, **kwargs):
    for layer in range(len(individual)):
        if len(individual[layer]) > 0:
            mut_indexes = sample(range(0, len(individual[layer])), 2)
            mut_indexes.sort()
            ind = individual[layer][mut_indexes[0]:mut_indexes[1]][::-1]
            individual[layer][mut_indexes[0]:mut_indexes[1]] = ind
    return individual


if __name__ == '__main__':
    test = [[1,2,3,4],[6,5,4,3]]
    test = inversion_mutation(test)
    print(test)

