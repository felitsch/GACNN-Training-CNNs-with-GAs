import uuid
from random import shuffle, choice, sample, random
from operator import attrgetter
from copy import deepcopy
import numpy as np
import time




# change representation to list of lists
# create more crossover methods

verbose = True

def vprint(*kwargs):
    if verbose:
        print(*kwargs)


class Individual:
    def __init__(
            self,
            representation=None,
            size=None,
            valid_set=None,  # do we need a valid set??
    ):
        if representation is None:
            self.representation = [np.random.uniform(size=size[i]) for i in range(len(size))]
        else:
            self.representation = [np.array(representation[rep_i]) for rep_i in range(4)]
        self.fitness = self.get_fitness()

    def get_fitness(self):
        raise Exception("You need to monkey patch the fitness path.")

    def set_weights(self):
        raise Exception("You need to monkey patch the set_weights path.")

    def index(self, value):
        return self.representation.index(value)

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, position):
        return self.representation[position]

    def __setitem__(self, position, value):
        self.representation[position] = value

    def __repr__(self):
        return f"Individual(size={len(self.representation)}); Fitness: {self.fitness}"


class Population:
    def __init__(self, size, optim, individuals=[], **kwargs):
        self.individuals = []
        self.size = size
        self.optim = optim
        self.elite= []

        for _ in range(size):
            self.individuals.append(
                Individual(
                    size=kwargs["sol_size"]
                )
            )




    def evolve(self, gen_start, gen_end, xo_prob, mut_prob, select, mutate, crossover, elitism=True, ms=0.1, runkey=uuid.uuid4(), patience=2, xo_multiplier=0.25, mut_multiplier=2, max_switch_possibilities=1):
        lstReturn = []
        best_fitness = None
        last_improved_gen = gen_start # STORES THE LAST GENERATION WITH IMPROVEMENT


        for igen in range(gen_start, gen_end+1):
            new_pop = []
            stepstart = time.time()

            if elitism > 0:
                if self.optim == "max":
                    elite = sorted(self.individuals, key=attrgetter("fitness"), reverse=True)[:elitism]
                elif self.optim == "min":
                    elite = sorted(self.individuals, key=attrgetter("fitness"))[:elitism]

                new_pop.extend(elite)

            while len(new_pop) < self.size:
                parent1, parent2 = select(self), select(self)

                if random() < xo_prob:
                    offspring1, offspring2 = crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2

                # Mutation
                if random() < mut_prob:
                    offspring1 = mutate(individual=offspring1, ms=ms)
                if random() < mut_prob:
                    offspring2 = mutate(individual=offspring2, ms=ms)


                x1 = Individual(representation=offspring1)
                x2 = Individual(representation=offspring2)

                new_pop.append(x1)
                if len(new_pop) < self.size:
                    new_pop.append(x2)

            self.individuals = new_pop

            if self.optim == "max":
                best_individual = max(self, key=attrgetter("fitness"))
                current_fitness = best_individual.fitness
            elif self.optim == "min":
                best_individual = min(self, key=attrgetter("fitness"))
                current_fitness = best_individual.fitness


            # CHECK IF THERE IS AN IMPROVEMENT IN FITNESS
            vprint("best_fitness_update (current gen/fitness):",igen,current_fitness, "  last improve gen/fitness:", last_improved_gen, best_fitness)
            if best_fitness is None or (self.optim == "max" and current_fitness > best_fitness) or (self.optim == "min" and current_fitness < best_fitness):
                vprint("updating best generation")
                best_fitness = current_fitness
                last_improved_gen = igen

            current_xo_prob = xo_prob
            current_mut_prob = mut_prob
            current_switch = max_switch_possibilities


            # MODEL IS NOT IMPROVING
            if igen - last_improved_gen >= patience:
                if max_switch_possibilities == 0:
                    # No longer possible to switch parameters; early stop
                    print(f"No improvement for {patience} generations. Early stopping.")
                    break
                else:
                    # ADJUST PARAMETERS
                    max_switch_possibilities -= 1
                    xo_prob = max(0, xo_prob * xo_multiplier)
                    mut_prob = max(0, mut_prob * mut_multiplier)
                    print(f"Changing parameters: xo_prob={xo_prob}, mut_prob={mut_prob}")
                    last_improved_gen = igen + 1

            stepstop = time.time()
            lstReturn.append([runkey, igen, best_individual.fitness, stepstop - stepstart, last_improved_gen, current_xo_prob, current_mut_prob, current_switch])

        return lstReturn



    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]
