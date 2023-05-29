#!pip install tensorflow

import numpy as np
import csv
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split

from operator import attrgetter
import time
import uuid

# Import individual; we need it go calculte the fitness
from charles.charles import Individual

# SWARM PARTICPLE OPTIMIZATION
from random import uniform, random
from copy import deepcopy


psoparams = {}
psoparams["computer"] = "PSO_Samuel"
psoparams["swarmsize"] = 500
psoparams["ndimensions"] = 16586 # [896, 0, 0, 15690]
psoparams["niterations"] = 10
psoparams["boxmin"] = -1
psoparams["boxmax"] = 1
psoparams["hyperparam_W"] = .2
psoparams["hyperparam_Cognitive"] = 2
psoparams["hyperparam_Social"] = 2
psoparams["psologfile"] = psoparams["computer"] + "_log.csv"  # If empty or undefined, do not log to CSV
psoparams["verbose"] = True


# Load dataset
(val_images, val_labels), (train_images, train_labels) = cifar10.load_data()

# Normalize pixel values to the range [0, 1]
train_images = train_images / 255.0
val_images = val_images / 255.0

# Reduce the train data to 10% while maintaining the class distribution
train_images, _, train_labels, _ = train_test_split(train_images, train_labels, train_size=0.1, stratify=train_labels,
                                                    random_state=42)

# Reduce the val data to 10% while maintaining the class distribution
val_images, _, val_labels, _ = train_test_split(val_images, val_labels, train_size=0.02, stratify=val_labels,
                                                random_state=42)

# vprint does a conditional print; if verbose is True, then execution provides more output
def vprint(*kwargs):
    if psoparams["verbose"]:
        print(*kwargs)


# create_model creates and compiles the model. The defaults correspond to CIFAR10, but other datasets can be used.
def create_model(output_classes=10, input_shape=(32, 32, 3), filters=32, kernel_size=(3, 3), strides=(2, 2)):
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
model = create_model(output_classes=10, input_shape=(32, 32, 3), filters=32, kernel_size=(3, 3), strides=(2, 2))

trainable_weights = []
for layer in model.layers:
    layer_param_count = layer.count_params()
    trainable_weights.append(layer_param_count)

trainable_layers = [layer for layer in model.layers if layer.trainable]


# Define the get_fitness for out model; monkey patch it to the Individual class.
def get_fitness(self):
    trainable_layers = [layer for layer in model.layers]  # if layer.trainable]

    input_shape = list(model.layers[0].trainable_weights[0].shape)
    input_threshold = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]

    out_shape = list(model.layers[3].trainable_weights[0].shape)
    model.layers[0].set_weights([self.representation[0][:input_threshold].reshape(
        (input_shape[0], input_shape[1], input_shape[2], input_shape[3])), self.representation[0][input_threshold:]])
    model.layers[3].set_weights(
        [self.representation[3][:out_shape[0] * out_shape[1]].reshape((out_shape[0], out_shape[1])),
         self.representation[3][out_shape[0] * out_shape[1]:]])

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




##########################################
##  PSO - PARTICLE SWARM OPTIMIZATION  ##

def particleconstant(constant = 9e99):
    lst = [constant for _ in range(psoparams["ndimensions"])]
    return Individual([lst[:896], [], [], lst[896:]])


def particlezero():
    #lst = [0 for _ in range(params["ndimensions"])]
    #return Individual([lst[:896], [], [], lst[896:]])
    return particleconstant(0)
    #return Individual([0 for _ in range(params["ndimensions"])])

def particlerandom():
    lst = [uniform(psoparams["boxmin"], psoparams["boxmax"]) for _ in range(psoparams["ndimensions"])]
    return Individual([lst[:896], [], [], lst[896:]])
    #return Individual([uniform(params["boxmin"], params["boxmax"]) for _ in range(params["ndimensions"])])


def particledistance(indiv1, indiv2=None):
    #If no second particle is passed, calculate the distance to the (theoretical) global optimum
    ret = 0
    if indiv2==None:
        ret = 1 - indiv1.get_fitness()
    else:
        ret = abs( indiv1.fitness - indiv2.fitness)
    return ret


def particledistancetoglobaloptimum(indiv1):
    return particledistance(indiv1)


def individual2list( indiv):
    lst = np.append(list(indiv.representation[0]), list(indiv.representation[3]))
    return lst


def list2individual(lst):
    return Individual([lst[:896], [], [], lst[896:]])



# Initialize a swarm of "swarmsize" particles
# Initialize an equivalent swarm to represent initial (random) weights
# Consider the particle's initial positions as the best positions - which they certainly are, so far
swarm = [particlerandom() for _ in range(psoparams["swarmsize"])]
velocity = [ particlerandom() for _ in range(psoparams["swarmsize"])]
bestpos = deepcopy(swarm)


# Initialize g as first particle
g = swarm[0]
dist = 9e99

# Update g to be the best particle's position that is found
for p in bestpos:
    tempdist = particledistancetoglobaloptimum(p)
    if tempdist < dist:
        dist = tempdist
        g = p

# Initialize parameters for inertia, cognitive strength (individual's best position) and
# social strength (leader's best position, so actually best position ever reached)
W          = psoparams["hyperparam_W"]
cCognitive = psoparams["hyperparam_Cognitive"]
cSocial    = psoparams["hyperparam_Social"]

# Set a unique key in the dictionary, so that upon exporting logs to disk we can separate different runs
psoparams["runkey"] = uuid.uuid4()

for iterations in range(psoparams["niterations"]):
    print( "Iteration {}, initial fitness is {}.".format(iterations, g.fitness))

    start = time.time()
    # for each particle in the swarm
    for i in range(len(swarm)):

        # Convert position, velocity and best position to the necessary format for calculations
        lstparticleposition = individual2list(swarm[i])
        lstparticlevelocity = individual2list(velocity[i])
        lstparticlebestpos = individual2list(bestpos[i])

        # move the particle according to current position and velocity
        lstparticleposition = np.add(lstparticleposition, lstparticlevelocity)
        swarm[i] = list2individual(lstparticleposition)

        # calculate the representations for velocity, cognitive and social influence,
        # introducing hyperparameters W, cognitive and social
        lstparticlevelocity = np.multiply(W, lstparticlevelocity)
        cognitiveinfluence = np.multiply(cCognitive, [random() for _ in range(psoparams["ndimensions"])])
        socialinfluence = np.multiply(cSocial, [random() for _ in range(psoparams["ndimensions"])])

        particlemovementtob = (np.subtract(lstparticlebestpos, lstparticleposition))
        particlemovementtog = (np.subtract(individual2list(g), lstparticleposition))
        cognitivemovement = np.multiply(cognitiveinfluence, particlemovementtob)
        socialmovement = np.multiply(socialinfluence, particlemovementtog)
        lstparticlevelocity = np.add(lstparticlevelocity, cognitivemovement)
        lstparticlevelocity = np.add(lstparticlevelocity, socialmovement)

        velocity[i] = list2individual(lstparticlevelocity)

        if particledistancetoglobaloptimum(bestpos[i]) > particledistancetoglobaloptimum(swarm[i]):
            bestpos[i] = swarm[i]
            if particledistancetoglobaloptimum(g) > particledistancetoglobaloptimum(swarm[i]):
                g = swarm[i]
                print("Iter {}. Improving known g; fitness is: {} distance is: {}".format(iterations+1, g.fitness, particledistancetoglobaloptimum(g)))

    end = time.time()

    # Record logs of the whole run
    if "psologfile" in psoparams.keys():
        psoparams["iteration"] = iterations + 1
        psoparams["best_global_fitness"] = g.fitness
        psoparams["duration_secs"] = end-start

        dict2csv(psoparams, psoparams["psologfile"])

