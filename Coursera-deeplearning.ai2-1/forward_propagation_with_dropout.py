# Function: forward_propagation_with_dropout
import numpy as np
from reg_utils import *


def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    """
    Implements the forward propagation: Linear -> ReLU + Dropout -> Linear -> ReLU + Dropout -> Linear -> Sigmoid

    Arguments:
    X -- input dataset of shape (2, number of examples)
    parameters -- python dictionary containing parameters "W1", "b1", "W2", "b2", "W3", "b3"
    keep_prob - probability of keep a neuron active during drop-out, scalar

    Returns:
    A3 -- last activation value, output of the forward propagation of shape (1, 1)
    cache -- tuple, information stored for computing the backward propagation
    """

    np.random.seed(1)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # Linear -> ReLU -> Linear -> ReLU -> Linear -> Sigmoid
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1]) < keep_prob
    A1 = np.multiply(A1, D1) / keep_prob

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    D2 = np.random.rand(A2.shape[0], A2.shape[1]) < keep_prob
    A2 = np.multiply(A2, D2) / keep_prob

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache


# Test
"""
np.random.seed(1)
X_assess = np.random.randn(3, 5)
W1 = np.random.randn(2, 3)
b1 = np.random.randn(2, 1)
W2 = np.random.randn(3, 2)
b2 = np.random.randn(3, 1)
W3 = np.random.randn(1, 3)
b3 = np.random.randn(1, 1)
parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
A3, cache = forward_propagation_with_dropout(X_assess, parameters, keep_prob = 0.7)
print ("A3 = " + str(A3))
"""

# Expected output
"""
A3 = [[ 0.36974721  0.00305176  0.04565099  0.49683389  0.36974721]]
"""