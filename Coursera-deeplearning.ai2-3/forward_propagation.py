# Function: forward_propagation
import tensorflow as tf
from create_placeholders import *
from initialize_parameters import *


def forward_propagation(X, parameters):
    """
    Implement forward propagation for the model: Linear -> ReLU -> Linear -> ReLU -> Linear -> Softmax

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing parameters "W1", "b1", "W2", "b2", "W3", "b3"

    Returns:
    Z3 -- output of the last Linear unit
    """

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3


# Test
"""
tf.reset_default_graph()
with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))
"""

# Expected output
"""
Z3 = Tensor("Add_2:0", shape=(6, ?), dtype=float32)
"""