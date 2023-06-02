# Genetic Algorithm Based Neural Network Optimization
This repository contains the implementation of a genetic algorithm based neural network optimization model that we applied on the CIFAR-10 dataset. The ultimate objective of this project is to optimize the weights and biases of the neural network model using genetic algorithms.

## Repository Structure
```
.
├── README.md
├── nn.py
├── pso.py
├── report.pdf
├── requirements.txt
└── charles
    ├── charles.py
    ├── crossover.py
    ├── mutation.py
    ├── selection.py
    └── utils.py
```

## Project Overview
The search space for this optimization problem consists of all combinations of the total number of weights and biases in the neural network, where each weight is a real number. The accuracy of the neural network is the fitness function, with the objective being to maximize the fitness.

## Neural Network Structure
The project defines a simple 5-layer neural network model including an input layer, a convolutional layer, a pooling layer, a flatten layer, and a dense layer. However, only the weights of the convolutional and dense layers are being optimized. To apply the code to other neural networks, minor parameter adjustments are required.

## Representation
The individual in our genetic algorithm is a list of lists, where each list contains the weights and biases for the corresponding layer in the neural network. The initial weights are randomly assigned continuous values in the range of [0,1], which are then allowed to change to any real value.

## Getting Started
Clone the repository.
Install all necessary dependencies using pip install -r requirements.txt.
Run the nn.py file to start the optimization process.

## Performance & Further Improvement
Currently, the model optimizes tens of thousands of variables and demands significant computational resources. The best accuracy achieved is below 0.25, which is a result of the simple architecture of the network.
We are confident that the results can be improved by running the model for more generations, given the model keeps improving incrementally. There's also the potential for implementing more crossover, mutation, or selection functions.

We have also implemented two new strategies to enhance the performance: "grafting" and Particle Swarm Optimization (PSO). Grafting evolves the parameters when the model shows no improvement after many generations, and PSO is another evolutionary method that seeks solutions in the existing search space.

## Future Work
Future work could include the exploration of hybrid models that combine the strengths of different evolutionary methods. This could potentially yield better optimization results with less computational requirements. Also, the neural network structure can be enhanced with more complex configurations or transfer learning techniques to further improve the performance.


## References
Esfahaniani, 2019, "Neural Network Weight Optimization using Genetic Algorithms"
