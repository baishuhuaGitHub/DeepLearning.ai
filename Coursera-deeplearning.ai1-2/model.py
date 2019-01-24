# Function: model
import numpy as np
from initialize_zeros import *
from optimize import *
from predict import *


def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    This function builds the logistic regression model

    Arguments:
    X_train -- training set of size (nx, m_train)
    Y_train -- training "label" set of size (1, m_train)
    X_test -- test set of size (nx, m_test)
    Y_test -- test "label" set of size (1, m_test)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    d -- dictionary containing information about the model
    """

    w, b = initialize_zeros(X_train.shape[0])       # Initialize
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)      # Obtain optimal parameters w and b

    w = params["w"]
    b = params["b"]

    Y_prediction_train = predict(w, b, X_train)     # Predict using train set
    Y_prediction_test = predict(w, b, X_test)       # Predict using test set

    # Print train/test accuracy
    print("Train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("Test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs, "Y_prediction_train": Y_prediction_train, "Y_prediction_test": Y_prediction_test, "w": w, "b": b, "learning_rate": learning_rate, "num_iterations": num_iterations}

    return d