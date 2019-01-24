# Function: compute_cost -- calculate the cost function
import numpy as np


def compute_cost(AL, Y):
    """
    Arguments:
    AL -- probability vector corresponding to predictions, shape (1, number of examples)
    Y -- true "label" vector, shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]
    cost = np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL))) / (- m)
    cost = np.squeeze(cost)
    
    return cost


# Test
"""
Y = np.asarray([[1, 1, 1]])
AL = np.array([[.8, .9, 0.4]])
print("cost = " + str(compute_cost(AL,Y)))
"""

# Expected output
"""
cost = 0.414931599615
"""