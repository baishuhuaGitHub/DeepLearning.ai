# Function: create_placeholders
import tensorflow as tf


def create_placeholders(n_x, n_y):
    """
    Create the placeholders for the tensorflow session

    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so ->6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    """

    X = tf.placeholder(tf.float32, shape = (n_x, None))
    Y = tf.placeholder(tf.float32, shape = (n_y, None))
    
    return X, Y


# Test
"""
X, Y = create_placeholders(12288, 6)
print("X = " + str(X))
print("Y = " + str(Y))
"""

# Expected output
"""
X = Tensor("Placeholder:0", shape=(12288, ?), dtype=float32)
Y = Tensor("Placeholder_1:0", shape=(6, ?), dtype=float32)
"""