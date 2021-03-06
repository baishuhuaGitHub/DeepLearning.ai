# Function: create_placeholders
import tensorflow as tf

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Create the placeholders for the tensorflow session

    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels of shape [None, n_y] and dtype "float"
    """

    X = tf.placeholder(tf.float32, shape = (None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32, shape = (None, n_y))

    return X, Y


# Test
"""
X, Y = create_placeholders(64, 64, 3, 6)
print("X = " + str(X))
print("Y = " + str(Y))
"""

# Expected output
"""
X = Tensor("Placeholder:0", shape=(?, 64, 64, 3), dtype=float32)
Y = Tensor("Placeholder_1:0", shape=(?, 6), dtype=float32)
"""