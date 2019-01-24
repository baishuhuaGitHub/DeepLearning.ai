# Function: update_parameters_with_gd
import numpy as np


def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using one step of gradient descent

    Arguments:
    parameters -- python dictionary containing parameters to be updated, "Wl" and "bl"
    grads -- python dictionary containing gradients to update each parameters, "dWl" and "dbl"
    learning_rate -- the learning_rate, scalar

    Returns:
    parameters -- python dictionary containing updated parameters
    """

    L =len(parameters) // 2     # Number of layers in neural networks

    for l in range(L):
        parameters["W" + str(l + 1)] += - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] += - learning_rate * grads["db" + str(l + 1)]
    
    return parameters


# Test
"""
np.random.seed(1)
learning_rate = 0.01
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
parameters = update_parameters_with_gd(parameters, grads, learning_rate)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
"""

# Expected output
"""
W1 = [[ 1.63535156 -0.62320365 -0.53718766]
 [-1.07799357  0.85639907 -2.29470142]]
b1 = [[ 1.74604067]
 [-0.75184921]]
W2 = [[ 0.32171798 -0.25467393  1.46902454]
 [-2.05617317 -0.31554548 -0.3756023 ]
 [ 1.1404819  -1.09976462 -0.1612551 ]]
b2 = [[-0.88020257]
 [ 0.02561572]
 [ 0.57539477]]
"""