# Function: linear_forward -- implement the linear part of a layer's forward propagation
import numpy as np


def linear_forward(A, W, b):
    """
    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    lieanr_cache -- a python dictionary containing "A", "W", "b", stored for computing the backward pass efficiencyly
    """
    
    Z = W.dot(A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    linear_cache = (A, W, b)
    return Z, linear_cache

# Test
"""
np.random.seed(1)
A = np.random.randn(3,2)
W = np.random.randn(1,3)
b = np.random.randn(1,1)
Z, linear_cache = linear_forward(A, W, b)
print("Z = " + str(Z))
"""

# Expected output
"""
Z = [[ 3.26295337 -1.23429987]]
"""