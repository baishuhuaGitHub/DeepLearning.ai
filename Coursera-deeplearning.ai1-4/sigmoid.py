# Function: sigmoid -- implements sigmoid activation
import numpy as np


def sigmoid(Z):
    """
    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(Z), same shape as Z
    activation_cache -- returns Z as well, useful during backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    assert(A.shape == Z.shape)
    activation_cache = Z

    return A, activation_cache
