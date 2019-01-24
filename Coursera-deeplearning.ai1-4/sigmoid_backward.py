# Function: sigmoid_backward -- implements the backward propagation for a single sigmoid unit
import numpy as np


def sigmoid_backward(dA, activation_cache):
    """
    Arguments:
    dA -- post-activation gradient of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- gradient of the cost with respect to Z
    """

    Z = activation_cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert(dZ.shape == Z.shape)

    return dZ
