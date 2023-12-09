# NAIL Net
NAIL Net is a python package for making neural networks.

You can install it with pip like this:
`pip install NAILNet`

and import it into your project like this:
`import NAILNet as nail`

## Docs
Make a model using this:
```
model = nail.Model(layerCounts, activations, mutationsPerLayer)
```
LayerCounts is a list of integers where each integer represents the number of nodes in a layer(input, hidden, or output) in the neural network.
Activations is a list of the activation functions of each layer. 
In both of these lists, the first value corresponds with the input layer, the last value corresponds with the output layer, and the values inbetween, the hidden layers. 
Mutations per layer is an argument that tells the model how many mutations should happen to a layer it picks to mutate. 


To train the model, you can use the NEATSim:
```
neat = nail.NEATSim(model_lcs, activations, scoreFunc, populationSize, reproSize, mutSize, mutationsPerLayer)
```
The model lcs, activations, and mutations per layer are the same as the values with these names when creating a Model object. 
The scoreFunc is a reference to a function that is passed a model and outputs a fitness score for that model.
The population size is the size of the population(integer) and the reproSize is the number of models that get to reproduce.
Finally, the mutSize is the number of mutations that happens to a model that should mutate.

Here's an example:
```
scoreModel = lambda x: sum(x.evaluate([1, 2, 3]))
neat = nail.NEATSim([3, 3, 3], ["linear", "sigmoid", "linear"], scoreModel, 20, 5, 3, 3)

```
This setup means the simulation will have a population of 20, each round 5 AIs will reproduce back to 20, and each mutation to happen will have 3 layers mutated and 3 mutations in that layer.

The possible activation functions are:
`linear`, `sigmoid`, `tanh`, `relu`, and `leaky_relu`

To run the simulation, you can decide between running it for a set number of rounds or until it reaches a specific fitness score.
These functions will return the best model produced after these rounds.
```
model = neat.simUntilFitnessReached(26)
```
or
```
model = neat.simForRounds(5)
```
