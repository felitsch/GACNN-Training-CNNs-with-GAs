#!pip install tensorflow

import numpy as np
import csv
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split

from charles.selection import fps, tournament_sel, nested_tournament_sel, rank_based_selection
from charles.mutation import geometric_mutation, gaussian_mutation, custom_swap_mutation, inversion_mutation
from charles.crossover import geometric_xo, arithmetic_xo, uniform_xo
from charles.charles import Population, Individual

from operator import attrgetter
import time
import uuid



#params
modelparams = {}
modelparams["computer"]                 = "Samuel"
modelparams["popsize"]                  = 10
modelparams["generations"]              = 10
modelparams["xo_prob"]                  = 0.9
modelparams["mut_prob"]                 = 0.2
modelparams["selection_function"]       = tournament_sel
modelparams["xo_function"]              = geometric_xo
modelparams["mutation_function"]        = geometric_mutation
modelparams["ms"]                       = 0.2
modelparams["elitism"]                  = 1
modelparams["patience"]                 = 100
modelparams["xo_multiplier"]            = 0.25
modelparams["mut_multiplier"]           = 2
modelparams["max_switch_possibilities"] = 0




params = {}
params["verbose"]   = True
params["logfile"]   = modelparams["computer"] + "_log.csv"    # If empty or undefined, do not log to CSV
params["logdetail"] = modelparams["computer"] + "_detail.csv"
params["runcount"]  = 5

# Load dataset
(val_images, val_labels), (train_images, train_labels) = cifar10.load_data()

# Normalize pixel values to the range [0, 1]
train_images = train_images / 255.0
val_images = val_images / 255.0

# Reduce the train data to 10% while maintaining the class distribution 
train_images, _, train_labels, _= train_test_split(train_images, train_labels, train_size=0.1, stratify=train_labels,
                                                    random_state=42)

# Reduce the val data to 10% while maintaining the class distribution
val_images, _, val_labels, _ = train_test_split(val_images, val_labels, train_size=0.02, stratify=val_labels,
                                                  random_state=42)


# vprint does a conditional print; if verbose is True, then execution provides more output
def vprint(*kwargs):
    if params["verbose"]:
        print(*kwargs)


# create_model creates and compiles the model. The defaults correspond to CIFAR10, but other datasets can be used.
def create_model(output_classes = 10, input_shape = (32, 32, 3), filters=32, kernel_size=(3,3), strides=(2,2)):

    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(filters, kernel_size=kernel_size, activation='relu', strides=strides),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(output_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['sparse_categorical_accuracy'])
    return model


# CIFAR10 has 10 labels, so output_classes = 10
# images are 32x32 pixels, 3 channels, so input_shape = (32, 32, 3)
model = create_model(output_classes = 10, input_shape = (32, 32, 3), filters=32, kernel_size=(3,3), strides=(2,2))

trainable_weights = []
for layer in model.layers:
    layer_param_count = layer.count_params()
    trainable_weights.append(layer_param_count)

trainable_layers = [layer for layer in model.layers if layer.trainable]



# Define the get_fitness for out model; monkey patch it to the Individual class.
def get_fitness(self):
    trainable_layers = [layer for layer in model.layers] # if layer.trainable]

    input_shape = list(model.layers[0].trainable_weights[0].shape)
    input_threshold = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]

    out_shape = list(model.layers[3].trainable_weights[0].shape)
    model.layers[0].set_weights([self.representation[0][:input_threshold].reshape((input_shape[0], input_shape[1], input_shape[2], input_shape[3])), self.representation[0][input_threshold:]])
    model.layers[3].set_weights([self.representation[3][:out_shape[0]*out_shape[1]].reshape((out_shape[0], out_shape[1])), self.representation[3][out_shape[0]*out_shape[1]:]])

    i, accuracy = model.evaluate(train_images, train_labels, verbose=0)  
    return accuracy  # Update the individual's fitness based on accuracy


# Monkey patching
Individual.get_fitness = get_fitness


# Used for logging
def dict2csv(dict, csvfile):
    if csvfile:
        f = open(csvfile, 'a', newline='')
        fcsv = csv.writer(f)
        needs_header = os.stat(csvfile).st_size == 0
        if needs_header:
            fcsv.writerow(list(dict))
            needs_header = False
        # Then, write values
        fcsv.writerow(dict.values())

        f.close()


# Used for logging
def list2csv(mylist, csvfile, header):
    if csvfile:
        f = open(csvfile, 'a', newline='')
        fcsv = csv.writer(f)
        needs_header = os.stat(csvfile).st_size == 0
        if needs_header:
            fcsv.writerow(header)
            needs_header = False
        # Then, write values
        for listitem in mylist:
            fcsv.writerow(listitem)
        f.close()


# function runevolution orchestrates the evolution, centralizing all parameters and the calling the
# population's evolve method with the arguments relevant at each moment
def  runevolution(
    computer                 = modelparams["computer"],
    popsize                  = modelparams["popsize"],
    generations              = modelparams["generations"],
    xo_prob                  = modelparams["xo_prob"],
    mut_prob                 = modelparams["mut_prob"],
    selection_function       = modelparams["selection_function"],
    mutation_function        = modelparams["mutation_function"],
    xo_function              = modelparams["xo_function"],
    elitism                  = modelparams["elitism"],
    ms                       = modelparams["ms"],
    patience                 = modelparams["patience"],
    max_switch_possibilities = modelparams["max_switch_possibilities"],
    xo_multiplier            = modelparams["xo_multiplier"],
    mut_multiplier           = modelparams["mut_multiplier"],
    run                      = 5):

    rundict = {}
    rundict["runkey"]                   = uuid.uuid4()
    rundict["computer"]                 = computer
    rundict["popsize"]                  = popsize
    rundict["generations"]              = generations
    rundict["xo_prob"]                  = xo_prob
    rundict["mut_prob"]                 = mut_prob
    rundict["selection_function"]       = selection_function
    rundict["mutation_function"]        = mutation_function
    rundict["xo_function"]              = xo_function
    rundict["elitism"]                  = elitism
    rundict["ms"]                       = ms
    rundict["patience"]                 = patience
    rundict["max_switch_possibilities"] = max_switch_possibilities
    rundict["xo_multiplier"]            = xo_multiplier
    rundict["mut_multiplier"]           = mut_multiplier
    rundict["runcount"]                 = run

    pop = Population(size=popsize, optim="max", sol_size=trainable_weights)

    start = time.time()
    detaillog = []

    # evolve method performs the evolution directly on the population.
    # returned results are just relevant for logging purposes
    evolveresults = pop.evolve(gen_start=1, gen_end=generations+1, xo_prob=xo_prob, mut_prob=mut_prob, select=selection_function,
                               mutate=mutation_function, crossover=xo_function, elitism=elitism, ms=ms, runkey = rundict["runkey"],
                               xo_multiplier=xo_multiplier, mut_multiplier=mut_multiplier, patience=patience,
                               max_switch_possibilities=max_switch_possibilities)

    # code used for logging results do CSV
    for xitem in evolveresults:
        detaillog.append(xitem)

    # calculate bestfitness, checking whether the problem is maximization or minimization
    if pop.optim == "max":
        valbestfit = max(pop, key=attrgetter("fitness")).fitness
        vprint(f'Best individual @gen {generations}: {max(pop, key=attrgetter("fitness"))}')
    if pop.optim == "min":
        valbestfit = min(pop, key=attrgetter("fitness")).fitness
        vprint(f'Best individual @gen {generations}: {min(pop, key=attrgetter("fitness"))}')

    end = time.time()

    # store execution parameters in the dictionary, so that they're saved to the logfile.
    rundict["bestfit"] = valbestfit
    rundict["duration_secs"] = end - start
    rundict["selection_function"] = rundict["selection_function"].__name__
    rundict["xo_function"] = rundict["xo_function"].__name__
    rundict["mutation_function"] = rundict["mutation_function"].__name__

    # output results to the console
    print( "Fitness (Accuracy): {}. Ran at {}, {} generations, pop size {}, xo prob {}, mut prob {}, Elit: {}, Sel={}, XO={}, Mut={}. Took {:.2f} secs.".format( valbestfit, rundict["computer"], rundict["generations"], rundict["popsize"], rundict["xo_prob"], rundict["mut_prob"], rundict["elitism"], rundict["selection_function"], rundict["xo_function"], rundict["mutation_function"], end - start))

    # Record logs of the whole run
    if "logfile" in params.keys():
        dict2csv(rundict, params["logfile"])

    # Record detailed logs, with results after each generation
    if "logdetail" in params.keys():
        list2csv(detaillog, params["logdetail"], header=["UUID", "Generation", "fitness", "timelapse", "last_improved_gen", "current_xo_prob", "current_mut_prob", "current_switch"])



# In order to test distinct values for parameters, we nested for loops, where each for will iterate through the
# corresponding list.
# Empty lists in practice mean that we're just setting that value.
# Parameters below correspond to the last run, in which we tested our best model with patience 20 (switching parameters
# after 20 generations without improvement) or unlimited patience (patience=1000 is higher than the number of generations)

xo_probs = [0.99]
mut_probs = [0.2]
pop_sizes = [100]
generations = [200]
msvals = [0.1]
elite_sizes = [1]

crossover_functions = [uniform_xo]
selection_functions = [nested_tournament_sel]
mutation_functions = [custom_swap_mutation]
patiences = [20, 1000]


for sel_func in selection_functions:
    for cross_func in crossover_functions:
        for mut_func in mutation_functions:
            for xo_probab in xo_probs:
                for mut_probab in mut_probs:
                    for pop_size in pop_sizes:
                        for generat in generations:
                            for msval in msvals:
                                for elitsiz in elite_sizes:
                                    for pat in patiences:
                                        for run in range(params["runcount"]): # perform 5 runs
                                            # Perform the evolution process with current combination of functions
                                            runevolution( selection_function=sel_func, xo_function=cross_func, mutation_function=mut_func,
                                                        xo_prob=xo_probab, mut_prob=mut_probab, popsize=pop_size, generations=generat,
                                                        elitism=elitsiz, ms=msval, run=run,
                                                        patience=pat, max_switch_possibilities=50,
                                                        xo_multiplier=0.75, mut_multiplier=1.4)

