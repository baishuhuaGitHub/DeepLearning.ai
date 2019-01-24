# Function:initialize
import numpy as np


def initialize(nx, nh, ny):
    """
    Arguments:
    nx -- size of input layer
    nh -- size of hidden layer
    ny -- size of output layer

    Returns:
    parameters -- dictionary containing parameters:
                    W1: weight matrix of shape (nh, nx)
                    b1: bias vector of shape (nh, 1)
                    W2: weight matrix of shape (ny, nh)
                    b2: bias vector of shape (ny, 1)
    """

    np.random.seed(2)
    W1 = np.random.randn(nh, nx) * 0.01
    b1 = np.zeros((nh, 1))
    W2 = np.random.randn(ny, nh) * 0.01
    b2 = np.zeros((ny, 1))

    assert (W1.shape == (nh, nx))
    assert (b1.shape == (nh, 1))
    assert (W2.shape == (ny, nh))
    assert (b2.shape == (ny, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2":b2}

    return parameters


# Test
"""
parameters = initialize(2, 4, 1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
"""

# Expected Output
"""
W1 = [[-0.00416758 -0.00056267]
 [-0.02136196  0.01640271]
 [-0.01793436 -0.00841747]
 [ 0.00502881 -0.01245288]]
b1 = [[ 0.]
 [ 0.]
 [ 0.]
 [ 0.]]
W2 = [[-0.01057952 -0.00909008  0.00551454  0.02292208]]
b2 = [[ 0.]]
"""