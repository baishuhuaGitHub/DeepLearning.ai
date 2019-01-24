# Function: sigmoid
import numpy as np


def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Returns:
    a -- sigmoid(z)
    """

    a = 1 / (1 + np.exp(-z))
    
    return a