# Function: compute_cost
import numpy as np


def compute_cost(A2, Y):
    """
    Compute the cross-entropy cost

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of size (1, number of samples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]
    cost = np.sum(np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))) / (- m)
    cost = np.squeeze(cost)

    assert (isinstance(cost, float))

    return cost


# Test
"""
np.random.seed(1)
Y = (np.random.randn(1, 3) > 0)
A2 = (np.array([[ 0.5002307 ,  0.49985831,  0.50023963]]))
cost = compute_cost(A2, Y)
print("cost = " + str(cost))
"""

# Expected Output
"""
cost = 0.693058761039
"""