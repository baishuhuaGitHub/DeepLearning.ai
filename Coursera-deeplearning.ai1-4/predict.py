# Function: predict
import numpy as np
from forward import *


def predict(X, Y, parameters):
    """
    This function is used to predict the results of L-layer neural network

    Argument:
    X -- data set of examples
    Y -- "true" label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    probas, caches = forward(X, parameters)
    prediction = (probas > 0.5)
    accuracy = np.sum(prediction == Y) / m

    return prediction, accuracy