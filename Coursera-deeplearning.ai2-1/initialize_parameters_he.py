# Function: initialize_parameters_he
import numpy as np


def initialize_parameters_he(layer_dims):
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
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


# Test
"""
parameters = initialize_parameters_he([2, 4, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
"""

# Expected output
"""
W1 = [[ 1.78862847  0.43650985]
 [ 0.09649747 -1.8634927 ]
 [-0.2773882  -0.35475898]
 [-0.08274148 -0.62700068]]
b1 = [[ 0.]
 [ 0.]
 [ 0.]
 [ 0.]]
W2 = [[-0.03098412 -0.33744411 -0.92904268  0.62552248]]
b2 = [[ 0.]]
"""