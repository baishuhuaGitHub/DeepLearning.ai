# Function: forward
import numpy as np
from sigmoid import *


def forward(X, parameters):
    """
    Arguments:
    X -- input data of size (nx, m)
    parameters -- dictionary containing parameters "W1", "b1", "W2", "b2"(output of initialize function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """

    Z1 = np.dot(parameters["W1"], X) + parameters["b1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["W2"], A1) + parameters["b2"]
    A2 = sigmoid(Z2)

    assert(A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    return A2, cache


# Test
"""
np.random.seed(1)
X = np.random.randn(2, 3)
b1 = np.random.randn(4,1)
b2 = np.array([[ -1.3]])
parameters = {'W1': np.array([[-0.00416758, -0.00056267], [-0.02136196,  0.01640271], [-0.01793436, -0.00841747], [ 0.00502881, -0.01245288]]),
              'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
              'b1': b1,
              'b2': b2}
A2, cache = forward(X, parameters)
print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))
"""

# Expected Output
"""
0.262818640198 0.091999045227 -1.30766601287 0.212877681719
"""