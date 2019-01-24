# Function: optimize
import numpy as np
from sigmoid import *


def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w,b)

    Arguments:
    w -- weights, a numpy array of size (nx, 1)
    b -- bias, a scalar
    X -- data of size (nx, number of samples m)

    Returns:
    Y_prediction -- a numpy array containing all predictions (0/1) for the examples in X
    """

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)     # Calculate probabilities
    Y_prediction[A > 0.5] = 1           # Convert probabilities to actual prediction

    assert(Y_prediction.shape == (1, m))

    return Y_prediction


# Test
"""
w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(w, b, X)))
"""

# Expected Output
"""
predictions = [[ 1.  1.  0.]]
"""


