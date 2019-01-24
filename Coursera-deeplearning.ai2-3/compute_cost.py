# Function: compute_cost
import tensorflow as tf
from create_placeholders import *
from initialize_parameters import *
from forward_propagation import *

def compute_cost(Z3, Y):
    """
    Compute the cost
    
    Arguments:
    Z3 -- output of forward propagation of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost -- Tensor of the cost function
    """

    # Fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits()
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    return cost


# Test
"""
tf.reset_default_graph()
with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    print("cost = " + str(cost))
"""

# Expected output
"""
cost = Tensor("Mean:0", shape=(), dtype=float32)
"""    