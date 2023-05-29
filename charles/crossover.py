from random import randint, uniform
import numpy as np
import random



def geometric_xo(p1,p2):
    offspring1 = [[None for elem in range(len(p1[layer]))] for layer in range(len(p1))]
    offspring2 = [[None for elem in range(len(p2[layer]))] for layer in range(len(p2))]
    for layer in range(len(p1)):
        r = np.random.uniform(size=len(p1[layer]))
        for elem in range(0,len(p1[layer])):
            offspring1[layer][elem] = r[elem]*p1[layer][elem] + (1-r[elem])*p2[layer][elem]
            offspring2[layer][elem] = r[elem]*p1[layer][elem] + (1-r[elem])*p2[layer][elem]
    return offspring1, offspring2


def arithmetic_xo(p1,p2):
    alpha = uniform(0,1)
    offspring1 = [[None for elem in range(len(p1[layer]))] for layer in range(len(p1))]
    offspring2 = [[None for elem in range(len(p1[layer]))] for layer in range(len(p1))]
    for layer in range(len(p1)):
        for elem in range(0, len(p1[layer])):
            offspring1[layer][elem] = p1[layer][elem]*alpha + (1-alpha)*p2[layer][elem]
            offspring2[layer][elem] = p1[layer][elem]*alpha + (1-alpha)*p2[layer][elem]
    return offspring1, offspring2


def uniform_xo(p1, p2):
    offspring1 = [[None for elem in range(len(p1[layer]))] for layer in range(len(p1))]
    offspring2 = [[None for elem in range(len(p1[layer]))] for layer in range(len(p1))]

    for layer in range(len(p1)):
        for elem in range(0, len(p1[layer])):
            if random.random() < 0.5:
                offspring1[layer][elem] = p1[layer][elem]
                offspring2[layer][elem] = p2[layer][elem]
            else:
                offspring1[layer][elem] = p2[layer][elem]
                offspring2[layer][elem] = p1[layer][elem]

    return offspring1, offspring2



if __name__ == '__main__':
    p1, p2 = [[2,1], [2,2,3]], [[1,4],[2,1,1]]
    o1 = uniform_xo(p1, p2)
    print(o1)
