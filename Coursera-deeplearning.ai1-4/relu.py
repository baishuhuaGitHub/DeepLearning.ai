# Function: relu -- implements RELU activation
import numpy as np


def relu(Z):
    """
    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of RELU(Z), same shape as Z
    activation_cache -- returns Z as well, useful during backpropagation
    """

    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    activation_cache = Z

    return A, activation_cache
