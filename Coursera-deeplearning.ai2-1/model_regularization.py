# Function: model_regularization
import numpy as np
from reg_utils import *
from forward_propagation_with_dropout import *
from compute_cost_with_regularization import *
from backward_propagation_with_regularization import *
from backward_propagation_with_dropout import *

def model_regularization(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
    """
    Implements a three-layer neural network: Linear -> ReLU -> Linear -> ReLU -> Linear -> Sigmoid

    Arguments:
    X -- input data of shape (input size, number of examples)
    Y -- true "label" vector of shape (output size, number of examples), containing 0 for red dots & 1 for blue dots
    learning rate -- learning rate of optimization
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, print the cost every 10000 iterations
    lambd -- regularization hyperparameter, scalar
    keep_prob -- probability of keeping a neuron active during drop-out, scalar

    Returns:
    parameters -- parameters learned by the model which can be used to predict
    """

    grads = {}
    costs = []
    m = X.shape[1]
    layer_dims = [X.shape[0], 20, 3, 1]

    # Initialize parameters
    parameters = initialize_parameters(layer_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)

        # Cost function
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

        # Backward propagation
        assert(lambd == 0 or keep_prob == 1)        # This assignment don't use L2 regularization and dropout simultaneous

        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the loss every 10000 iteration
        if i % 1000 == 0:
            costs.append(cost)
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))

    # Plot the cost
    plt.figure(1)
    plt.plot(costs)
    plt.xlabel('iterations (x 1000)')
    plt.ylabel('cost')
    plt.title("Learning rate = " + str(learning_rate))

    return parameters