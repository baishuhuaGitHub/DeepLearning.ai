# Function: update_parameters_with_momentum
import numpy as np


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum
    
    Arguments:
    parameters -- python dictionary containing parameters to be updated, "Wl" and "bl"
    grads -- python dictionary containing gradients to update each parameters, "dWl" and "dbl"
    v -- python dictionary containing current velocity, "vdWl" and "vdbl"
    beta -- momentum hyperparameter, scalar
    learning_rate -- the learning_rate, scalar

    Returns:
    parameters -- python dictionary containing updated parameters
    v -- python dictionary containing updated velocity
    """

    L = len(parameters) // 2        # Number of layers in neural networks

    for l in range(L):
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]
        parameters["W" + str(l + 1)] += - learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] += - learning_rate * v["db" + str(l + 1)]

    return parameters, v


# Test
"""
np.random.seed(1)
W1 = np.random.randn(2,3)
b1 = np.random.randn(2,1)
W2 = np.random.randn(3,3)
b2 = np.random.randn(3,1)
dW1 = np.random.randn(2,3)
db1 = np.random.randn(2,1)
dW2 = np.random.randn(3,3)
db2 = np.random.randn(3,1)
parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
v = {'dW1': np.array([[ 0.,  0.,  0.], [ 0.,  0.,  0.]]),
     'dW2': np.array([[ 0.,  0.,  0.], [ 0.,  0.,  0.], [ 0.,  0.,  0.]]),
     'db1': np.array([[ 0.], [ 0.]]),
     'db2': np.array([[ 0.], [ 0.], [ 0.]])}
parameters, v = update_parameters_with_momentum(parameters, grads, v, beta = 0.9, learning_rate = 0.01)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print("v[\"dW1\"] = " + str(v["dW1"]))
print("v[\"db1\"] = " + str(v["db1"]))
print("v[\"dW2\"] = " + str(v["dW2"]))
print("v[\"db2\"] = " + str(v["db2"]))
"""

# Expected output
"""
W1 = [[ 1.62544598 -0.61290114 -0.52907334]
 [-1.07347112  0.86450677 -2.30085497]]
b1 = [[ 1.74493465]
 [-0.76027113]]
W2 = [[ 0.31930698 -0.24990073  1.4627996 ]
 [-2.05974396 -0.32173003 -0.38320915]
 [ 1.13444069 -1.0998786  -0.1713109 ]]
b2 = [[-0.87809283]
 [ 0.04055394]
 [ 0.58207317]]
v["dW1"] = [[-0.11006192  0.11447237  0.09015907]
 [ 0.05024943  0.09008559 -0.06837279]]
v["db1"] = [[-0.01228902]
 [-0.09357694]]
v["dW2"] = [[-0.02678881  0.05303555 -0.06916608]
 [-0.03967535 -0.06871727 -0.08452056]
 [-0.06712461 -0.00126646 -0.11173103]]
v["db2"] = [[ 0.02344157]
 [ 0.16598022]
 [ 0.07420442]]
"""