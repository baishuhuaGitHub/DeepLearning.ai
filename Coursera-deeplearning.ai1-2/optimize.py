# Function: optimize
import numpy as np
from propagate import *


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimize w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (nx, 1)
    b -- bias, a scalar
    X -- data of size (nx, number of samples m)
    Y -- true "label" vector of size (1, number of samples m)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the cost function with respect to the weights dw and bias db
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve
    """
    
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0: costs.append(cost)
        if print_cost and i % 100 == 0: print("Cost after iteration %i: %f" %(i,cost))

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    
    return params, grads, costs
 
    
# Test
"""
w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
"""

# Expected Output
"""
w = [[ 0.19033591]
 [ 0.12259159]]
b = 1.92535983008
dw = [[ 0.67752042]
 [ 1.41625495]]
db = 0.219194504541
"""