# Function:model
import numpy as np
import matplotlib.pyplot as plt
from opt_utils import *
from initialize_velocity import *
from initialize_adam import *
from random_mini_batches import *
from update_parameters_with_gd import *
from update_parameters_with_momentum import *
from update_parameters_with_adam import *

def model(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_epochs = 10000, print_cost = True):
    """
    Neural network model which use different optimizer modes

    Arguments:
    X -- input data of shape (input size, number of examples)
    Y -- true "label" vector of shape (1, number of examples), 1 for blue dot / 0 for red dot
    layers_dims -- list containing the size of each layer
    learning rate -- the learning rate, scalar
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- exponential decay hyperparameter for the past gradients estimates
    beta2 -- exponential decay hyperparameter for the past squared gradients estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs

    Returns:
    parameters -- python dictionary containing updated parameters
    """

    L = len(layers_dims)        # Number of layers in the neural networks
    costs = []
    t = 0
    seed = 10

    # Initialize parameters
    parameters = initialize_parameters(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # Optimization loop
    for i in range(num_epochs):
        seed = seed + 1     # Random for every epoch
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch                          # Select a minibatch
            a3, caches = forward_propagation(minibatch_X, parameters)       # Forward propagation
            cost = compute_cost(a3, minibatch_Y)                            # Compute cost
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)  # Backward propagation

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1       # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)

        # Print cost every 1000 epoch
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" %(i, cost))

    # Plot the cost
    plt.figure(2)
    plt.plot(costs)
    plt.xlabel('epochs (per 100)')
    plt.ylabel('cost')
    plt.title('Learning rate = ' + str(learning_rate))

    return parameters