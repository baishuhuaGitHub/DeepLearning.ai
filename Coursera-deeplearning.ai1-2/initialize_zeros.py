# Function: initialize_zeros
import numpy as np


def initialize_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w, and initizlizes b to 0

    Arguments:
    dim -- size of the w vector

    Returns:
    w -- initialized vector of shape (dim,1)
    b -- initialized scalar
    """
    
    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim,1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b


# Test
"""
dim = 2
w, b = initialize_zeros(dim)
print("w = " + str(w))
print("b = " + str(b))
"""

# Expected Output
"""
w = [[ 0.]
 [ 0.]]
b = 0
"""
