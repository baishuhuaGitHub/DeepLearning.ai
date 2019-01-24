# Function: backward
import numpy as np


def backward(X, Y, parameters, cache):
    """
    Implement the backward propagation

    Arguments:
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- dictionary containing parameters "W1", "b1", "W2", "b2"
    cache -- dictionary containing "Z1", "A1", "Z2" and "A2"

    Returns:
    grads -- dictionary containing gradients "dW1", "db1", "dW2", "db2"
    """

    m = Y.shape[1]

    dZ2 = cache["A2"] - Y
    dW2 = np.dot(dZ2, cache["A1"].T) / m
    db2 = np.sum(dZ2, axis = 1, keepdims = True) / m
    dZ1 = np.multiply(np.dot(parameters["W2"].T, dZ2), (1 - np.power(cache["A1"], 2)))       # For tanh, g'(z) = 1 - g(z)^2
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis = 1, keepdims = True) / m

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads


# Test
"""
np.random.seed(1)
X = np.random.randn(2, 3)
Y = (np.random.randn(1, 3) > 0)
parameters = {'W1': np.array([[-0.00416758, -0.00056267], [-0.02136196,  0.01640271], [-0.01793436, -0.00841747], [ 0.00502881, -0.01245288]]),
              'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
              'b1': np.array([[ 0.], [ 0.], [ 0.], [ 0.]]),
              'b2': np.array([[ 0.]])}
cache = {'A1': np.array([[-0.00616578,  0.0020626 ,  0.00349619], [-0.05225116,  0.02725659, -0.02646251], [-0.02009721,  0.0036869 ,  0.02883756], [ 0.02152675, -0.01385234,  0.02599885]]),
         'A2': np.array([[ 0.5002307 ,  0.49985831,  0.50023963]]),
         'Z1': np.array([[-0.00616586,  0.0020626 ,  0.0034962 ], [-0.05229879,  0.02726335, -0.02646869], [-0.02009991,  0.00368692,  0.02884556], [ 0.02153007, -0.01385322,  0.02600471]]),
         'Z2': np.array([[ 0.00092281, -0.00056678,  0.00095853]])}
grads = backward(X, Y, parameters, cache)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))
"""

# Expected Output
"""
dW1 = [[ 0.00301023 -0.00747267]
 [ 0.00257968 -0.00641288]
 [-0.00156892  0.003893  ]
 [-0.00652037  0.01618243]]
db1 = [[ 0.00176201]
 [ 0.00150995]
 [-0.00091736]
 [-0.00381422]]
dW2 = [[ 0.00078841  0.01765429 -0.00084166 -0.01022527]]
db2 = [[-0.16655712]]
"""