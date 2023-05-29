from random import uniform, choice, sample
from operator import attrgetter
from charles.charles import Population
import numpy as np


def fps(population):
    """Fitness proportionate selection implementation.

    Args:
        population (Population): The population we want to select from.

    Returns:
        Individual: selected individual.
    """

    if population.optim == "max":

        # Sum total fitness
        total_fitness = sum([i.fitness for i in population])
        # Get a 'position' on the wheel
        spin = uniform(0, total_fitness)
        position = 0
        # Find individual in the position of the spin
        for individual in population:
            position += individual.fitness
            if position > spin:
                return individual

    elif population.optim == "min":
        raise NotImplementedError

    else:
        raise Exception("No optimization specified (min or max).")


def tournament_sel(population, size=2):
    tournament = [choice(population.individuals) for _ in range(size)]
    if population.optim == 'max':
       return max(tournament,key=attrgetter('fitness'))
    if population.optim == 'min':
       return min(tournament,key=attrgetter('fitness'))

def nested_tournament_sel(population,  num_tournaments=2, tournament_size=5):
    selected = []

    for _ in range(num_tournaments):
        pop2 = Population(size=0, optim="max")
        tournament = [choice(population.individuals) for _ in range(tournament_size)]
        for indiv in tournament:
            pop2.individuals.append(indiv)
        winner = tournament_sel(pop2, tournament_size)
        selected.append(winner)
    if population.optim == 'max':
        return max(selected, key=attrgetter('fitness'))
    if population.optim == 'min':
        return min(selected, key=attrgetter('fitness'))

def rank_based_selection(population):
    sorted_population = sorted(population.individuals, key=attrgetter("fitness"), reverse=True)
    total_rank = sum(range(1, population.size + 1))
    spin = uniform(0, total_rank)
    position = 0
    for rank, individual in enumerate(sorted_population, 1):
        position += rank
        if position > spin:
            return individual



