# Function: model
import numpy as np
from init_utils import *
from initialize_parameters_zeros import *
from initialize_parameters_random import *
from initialize_parameters_he import *


def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
    """
    Implements a three-layer neural network: Linear -> ReLU -> Linear -> ReLU -> Linear -> Sigmoid

    Arguments:
    X -- input data of shape (2, number of examples)
    Y -- true "label" vector of shape (1, number of examples), containing 0 for red dots & 1 for blue dots
    learning rate -- learning rate for gradient descent
    num_iterations -- number of iterations to run gradient descent
    print_cost -- if True, print the cost every 1000 iterations
    initialization -- flag to choose which initialization to use ("zeros", "random", "he")

    Returns:
    parameters -- parameters learnt by the model
    """

    grads = {}
    costs = []
    m = X.shape[1]
    layer_dims = [X.shape[0], 10, 5, 1]

    # Initialize parameters
    if initialization == "zeros": parameters = initialize_parameters_zeros(layer_dims)
    elif initialization == "random": parameters = initialize_parameters_random(layer_dims)
    elif initialization == 'he': parameters = initialize_parameters_he(layer_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        a3, cache = forward_propagation(X, parameters)      # Forward propagationg:
        cost = compute_loss(a3, Y)      # Loss
        grads = backward_propagation(X, Y, cache)           # Backward propagation
        parameters = update_parameters(parameters, grads, learning_rate)        # Update parameters

        # Print the loss every 1000 iterations
        if i % 1000 == 0: costs.append(cost)
        if print_cost and i % 1000 == 0: print("Cost after iteration {}: {}".format(i, cost))
    # Plot the loss
    plt.figure(1)
    plt.plot(costs)
    plt.xlabel('iterations (per thousand)')
    plt.ylabel('cost')
    plt.title('Learning_rate = ' + str(learning_rate))

    return parameters