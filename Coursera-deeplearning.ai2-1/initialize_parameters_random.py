# Function: initialize_parameters_random
import numpy as np


def initialize_parameters_random(layer_dims):
    """
    Arguments:
    layer_dims -- python array containing the size of each layer

    Returns:
    parameters -- python dictionary containing parameters "W1", "b1", ..., "WL", "bL"
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)         # Number of layers in the network

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 10
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


# Test
"""
parameters = initialize_parameters_random([3, 2, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
"""

# Expected output
"""
W1 = [[ 17.88628473   4.36509851   0.96497468]
 [-18.63492703  -2.77388203  -3.54758979]]
b1 = [[ 0.]
 [ 0.]]
W2 = [[-0.82741481 -6.27000677]]
b2 = [[ 0.]]
"""