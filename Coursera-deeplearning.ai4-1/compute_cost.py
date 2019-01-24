# Function: compute_cost
import tensorflow as tf


def compute_cost(Z3, Y):
    """
    Compute the cost

    Arguments:
    Z3 -- output of forward propagation of shape (number of classes, number of examples)
    Y -- "true" label vector placeholder, same shape as Z3

    Returns:
    cost -- tensor of the cost function
    """

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))

    return cost


# Test
"""
tf.reset_default_graph()
from create_placeholders import *
from initialize_parameters import *
from forward_propagation import *
import numpy as np
with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(64, 64, 3, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(cost, {X: np.random.randn(4, 64, 64, 3), Y: np.random.randn(4, 6)})
    print("cost = " + str(a))
"""

# Expected output
"""
Maybe different
cost = 4.66487
"""