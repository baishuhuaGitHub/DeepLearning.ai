# Function: linear_activation_forward -- implement the forward propagation for the LINEAR->ACTIVATION layer
import numpy as np
from linear_forward import *
from relu import *
from sigmoid import *


def linear_activation_forward(A_prev, W, b, activation):
    """
    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- "sigmoid" or "relu" function

    Returns:
    A -- the output the the activation function, also alled the post-activation value
    cache -- a python dictionary containing linear_cache ("A", "W", "b") and activation_cache ("Z"), stored for computing the backward pass efficiencyly
    """
    
    # Sigmoid activation function
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    # ReLU activation function
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


# Test
"""
np.random.seed(2)
A_prev = np.random.randn(3,2)
W = np.random.randn(1,3)
b = np.random.randn(1,1)

A, cache = linear_activation_forward(A_prev, W, b, "sigmoid")
print("With sigmoid: A = " + str(A))

A, cache = linear_activation_forward(A_prev, W, b, "relu")
print("With ReLU: A = " + str(A))
"""

# Expected output
"""
With sigmoid: A = [[ 0.96890023  0.11013289]]
With ReLU: A = [[ 3.43896131  0.        ]]
"""