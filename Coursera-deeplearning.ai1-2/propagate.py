# Function: propagate
import numpy as np
from sigmoid import *


def propagate(w, b, X, Y):
    """
    This function implement the cost function and its gradient for the propatation

    Arguments:
    w -- weights, a numpy array of size (nx, 1)
    b -- bias, a scalar
    X -- data of size (nx, number of samples m)
    Y -- true "label" vector of size (1, number of samples m)

    Returns:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, same shape as w
    db -- gradient of the loss with respect to b, same shape as b
    """
    
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)         # compute activation
    cost = np.sum(np.multiply(Y, np.log(A)) + np.multiply(1 - Y, np.log(1 - A))) / (- m)      # compute cost
    
    dZ = A - Y
    dw = np.dot(X, dZ.T) / m
    db = np.sum(dZ) / m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw, "db": db}

    return grads, cost


# Test
"""
w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
print("cost = " + str(cost))
"""

# Expected Output
"""
dw = [[ 0.99845601]
 [ 2.39507239]]
db = 0.00145557813678
cost = 5.80154531939
"""