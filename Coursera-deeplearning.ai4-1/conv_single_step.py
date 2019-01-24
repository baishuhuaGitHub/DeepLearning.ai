# Function: conv_single_step
import numpy as np


def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice of the output activation of the previous layer.

    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- weight parameters contained in a window of shape (f, f, n_C_prev)
    b -- bias parameters contained in a window of shape (1, 1, 1)

    Returns:
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """

    s = np.sum(np.multiply(a_slice_prev, W)) + float(b)

    return s


# Test
"""
np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)
Z = conv_single_step(a_slice_prev, W, b)
print("Z =", Z)
"""

# Expected output
"""
Z = -6.99908945068
"""