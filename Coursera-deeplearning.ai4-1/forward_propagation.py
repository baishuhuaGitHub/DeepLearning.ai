# Function: forward_propagation
import tensorflow as tf


def forward_propagation(X, parameters):
    """
    Implement the forward propagation for model: Conv2d -> ReLU -> Maxpool -> Conv2d -> ReLU -> Maxpool -> Flatten -> Fullyconnected

    Arguments:
    X -- input dataset placeholder of shape (input size, number of examples)
    parameters -- python dictionary containing parameters "W1" "W2"

    Returns:
    Z3 -- output of the last Linear unit
    """

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    Z1 = tf.nn.conv2d(X, W1, strides = [1, 1, 1, 1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = [1, 8, 8, 1], strides = [1, 8, 8, 1], padding = 'SAME')
    Z2 = tf.nn.conv2d(P1, W2, strides = [1, 1, 1, 1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding = 'SAME')
    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn = None)
    return Z3


# Test
"""
tf.reset_default_graph()
import numpy as np
from create_placeholders import *
from initialize_parameters import *
with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(64, 64, 3, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(Z3, {X: np.random.randn(2, 64, 64, 3), Y: np.random.randn(2, 6)})
    print("Z3 = " + str(a))
"""

# Expected output
"""
Maybe different
Z3 = [[ 1.44169843 -0.24909666  5.45049906 -0.26189619 -0.20669907  1.36546707]
 [ 1.40708458 -0.02573211  5.08928013 -0.48669922 -0.40940708  1.26248586]]
"""