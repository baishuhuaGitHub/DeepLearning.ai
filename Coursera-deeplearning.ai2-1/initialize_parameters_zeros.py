# Function: initialize_parameters_zeros
import numpy as np


def initialize_parameters_zeros(layer_dims):
    """
    Arguments:
    layer_dims -- python array containing the size of each layer

    Returns:
    parameters -- python dictionary containing parameters "W1", "b1", ..., "WL", "bL"
    """

    parameters = {}
    L = len(layer_dims)         # Number of layers in the network

    for l in range(1, L):
        parameters["W" + str(l)] = np.zeros((layer_dims[l], layer_dims[l - 1]))
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


# Test
"""
parameters = initialize_parameters_zeros([3, 2, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
"""

# Expected output
"""
W1 = [[ 0.  0.  0.]
 [ 0.  0.  0.]]
b1 = [[ 0.]
 [ 0.]]
W2 = [[ 0.  0.]]
b2 = [[ 0.]]
"""