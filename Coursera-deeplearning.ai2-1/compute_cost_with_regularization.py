# Functiion: compute_cost_with_regularization
import numpy as np
from reg_utils import *

def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    Implement the cost function with L2 rgularization

    Arguments:
    A3 -- post-activation, output of forward propagation of shape (output size,  number of examples)
    Y -- "true" labels vector of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of th model

    Returns:
    cost -- value of the regularized loss function
    """

    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost = compute_cost(A3, Y)
    L2_regularization_cost = (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) * lambd / (2 * m)

    cost = cross_entropy_cost + L2_regularization_cost

    return cost


# Test
"""
np.random.seed(1)
Y_assess = np.array([[1, 1, 0, 1, 0]])
W1 = np.random.randn(2, 3)
b1 = np.random.randn(2, 1)
W2 = np.random.randn(3, 2)
b2 = np.random.randn(3, 1)
W3 = np.random.randn(1, 3)
b3 = np.random.randn(1, 1)
parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
A3 = np.array([[ 0.40682402,  0.01629284,  0.16722898,  0.10118111,  0.40682402]])
print("cost = " + str(compute_cost_with_regularization(A3, Y_assess, parameters, lambd = 0.1)))
"""

# Expected output
"""
cost = 1.78648594516
"""