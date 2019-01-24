# Function: forward -- implement forward propagation for the [LINEAR->RELU]*(L-1)->[LINEAR->SIGMOID] computation
from linear_activation_forward import *

def forward(X, parameters):
    """
    Arguments:
    X -- data of shape (input size, number of examples)
    parameters -- output of initialize(layer_dims)

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid forward() (there is one, indexed L-1)

    """

    caches = []
    A = X
    L = len(parameters) // 2            # number of layers in the neural network

    # Implement [LINEAR->RELU]*(L-1)
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))

    return AL, caches
    

# Test
"""
np.random.seed(6)
X = np.random.randn(5,4)
W1 = np.random.randn(4,5)
b1 = np.random.randn(4,1)
W2 = np.random.randn(3,4)
b2 = np.random.randn(3,1)
W3 = np.random.randn(1,3)
b3 = np.random.randn(1,1)
parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
AL, caches = forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))
"""

# Expected output
"""
AL = [[ 0.03921668  0.70498921  0.19734387  0.04728177]]
Length of caches list = 3
"""