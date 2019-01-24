# Function: update_parameters
import numpy as np


def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent

    Arguments:
    parameters -- dictionary containing parameters "W1", "b1", "W2", "b2" (output of initialize function)
    grads -- dictionary containing gradients "dW1", "db1", "dW2", "db2"
    learning_rate -- learning rate

    Returns:
    parameters -- updated parameters "W1", "b1", "W2", "b2"
    """

    parameters["W1"] = parameters["W1"] - learning_rate * grads["dW1"]
    parameters["b1"] = parameters["b1"] - learning_rate * grads["db1"]
    parameters["W2"] = parameters["W2"] - learning_rate * grads["dW2"]
    parameters["b2"] = parameters["b2"] - learning_rate * grads["db2"]

    return parameters


# Test
"""
parameters = {'W1': np.array([[-0.00615039,  0.0169021 ], [-0.02311792,  0.03137121], [-0.0169217 , -0.01752545], [ 0.00935436, -0.05018221]]),
              'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
              'b1': np.array([[ -8.97523455e-07], [  8.15562092e-06], [  6.04810633e-07], [ -2.54560700e-06]]),
              'b2': np.array([[  9.14954378e-05]])}
grads = {'dW1': np.array([[ 0.00023322, -0.00205423], [ 0.00082222, -0.00700776], [-0.00031831,  0.0028636 ], [-0.00092857,  0.00809933]]),
         'dW2': np.array([[ -1.75740039e-05,   3.70231337e-03,  -1.25683095e-03, -2.55715317e-03]]),
         'db1': np.array([[  1.05570087e-07], [ -3.81814487e-06], [ -1.90155145e-07], [  5.46467802e-07]]),
         'db2': np.array([[ -1.08923140e-05]])}
parameters = update_parameters(parameters, grads)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
"""

# Expected Output
"""
W1 = [[-0.00643025  0.01936718]
 [-0.02410458  0.03978052]
 [-0.01653973 -0.02096177]
 [ 0.01046864 -0.05990141]]
b1 = [[ -1.02420756e-06]
 [  1.27373948e-05]
 [  8.32996807e-07]
 [ -3.20136836e-06]]
W2 = [[-0.01041081 -0.04463285  0.01758031  0.04747113]]
b2 = [[ 0.00010457]]
"""