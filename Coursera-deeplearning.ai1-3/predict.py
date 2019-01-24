# Function: predict
import numpy as np
from forward import *


def predict(X, parameters):
    """
    Arguments:
    X -- input data of shape (2, number of examples)
    parameters -- parameters learnt by model "W1", "b1", "W2", "b2"

    Returns:
    predictions -- vector of predictions of the nn_model (red: 0 / blue: 1)
    """

    A2, cache = forward(X, parameters)
    predictions = (A2 > 0.5)

    return predictions


# Test
"""
np.random.seed(1)
X = np.random.randn(2, 3)
parameters = {'W1': np.array([[-0.00615039,  0.0169021 ], [-0.02311792,  0.03137121], [-0.0169217 , -0.01752545], [ 0.00935436, -0.05018221]]),
              'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
              'b1': np.array([[ -8.97523455e-07], [  8.15562092e-06], [  6.04810633e-07], [ -2.54560700e-06]]),
              'b2': np.array([[  9.14954378e-05]])}
predictions = predict(X, parameters)
print("predictions mean = " + str(np.mean(predictions)))
"""

# Expected Output
"""
predictions mean = 0.666666666667
"""