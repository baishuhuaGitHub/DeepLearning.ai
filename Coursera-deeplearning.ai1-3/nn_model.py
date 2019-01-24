# Function: nn_model
import numpy as np
from initialize import *
from forward import *
from compute_cost import *
from backward import *
from update_parameters import *


def nn_model(X, Y, nh, num_iterations = 10000, print_cost = False):
    """
    Arguments:
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of size (1, number of samples)
    nh -- size of hidden layer
    num_iterations -- number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameter -- parameters learnt by the model, which can be used to predict
    """
    np.random.seed(3)
    nx = X.shape[0]
    ny = Y.shape[0]
    m = X.shape[1]

    parameters = initialize(nx, nh, ny)

    for i in range(0, num_iterations):
        A2, cache = forward(X, parameters)
        grads = backward(X, Y, parameters, cache)
        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1000 == 0:
            cost = compute_cost(A2, Y)
            print("Cost after iteration %i: %f " %(i, cost))

    return parameters


# Test
"""
np.random.seed(1)
X = np.random.randn(2, 3)
Y = (np.random.randn(1, 3) > 0)
parameters = nn_model(X, Y, 4, num_iterations = 10000, print_cost = True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
"""

# Expected Output
"""
Cost after iteration 0: 0.692739
Cost after iteration 1000: 0.000218
Cost after iteration 2000: 0.000107
Cost after iteration 3000: 0.000071
Cost after iteration 4000: 0.000053
Cost after iteration 5000: 0.000042
Cost after iteration 6000: 0.000035
Cost after iteration 7000: 0.000030
Cost after iteration 8000: 0.000026
Cost after iteration 9000: 0.000023
W1 = [[-0.65848169  1.21866811]
 [-0.76204273  1.39377573]
 [ 0.5792005  -1.10397703]
 [ 0.76773391 -1.41477129]]
b1 = [[ 0.287592  ]
 [ 0.3511264 ]
 [-0.2431246 ]
 [-0.35772805]]
W2 = [[-2.45566237 -3.27042274  2.00784958  3.36773273]]
b2 = [[ 0.20459656]]
"""