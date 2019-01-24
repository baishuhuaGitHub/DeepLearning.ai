# Function: relu_backward -- implement the backward propagation for a single relu unit
import numpy as np


def relu_backward(dA, activation_cache):
    """
    Arguments:
    dA -- post-activation gradient of any shape
    activation_cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- gradient of the cost with respect to Z
    """

    Z = activation_cache
    dZ = np.array(dA, copy = True)          # just converting dZ to a correct object
    dZ[Z <=0] = 0                           # When Z <=0, set dZ to 0

    assert(dZ.shape == Z.shape)

    return dZ
