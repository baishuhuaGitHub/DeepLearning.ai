# Function: model_training -- training L-layer neural network parameters: [LINEAR->RELU]*(L-1) -> [LINEAR->SIGMOID]
import numpy as np
import matplotlib.pyplot as plt
from initialize import *
from forward import *
from compute_cost import *
from backward import *
from update_parameters import *


def dnn_model(train_x, train_y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
    """
    Arguments:
    train_x -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    train_y -- true "label" vector (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1)
    learning rate -- learning rate of the gradient descent update
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model, which can be used to predict
    """
    
    np.random.seed(1)
    costs = []
    parameters = initialize(layers_dims)

    for i in range (0, num_iterations):
        AL, caches = forward(train_x, parameters)
        grads = backward(AL, train_y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        cost = compute_cost(AL, train_y)

        # Print cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)

    if print_cost:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate = " + str(learning_rate))
    return parameters