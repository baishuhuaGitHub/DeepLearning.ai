# Function: distribute_value
import numpy as np


def distribute_value(dz, shape):
    """
    Distribute the input value in the matrix of dimension shape

    Arguments:
    dz -- input scalar
    shape -- shape (n_H, n_W) of output matrix that distributing the value of dz

    Returns:
    a -- array that distributing the value of dz, shape (n_H, n_W)
    """
    
    a = np.ones(shape) * dz / (shape[0] * shape[1])

    return a


# Test
"""
a = distribute_value(2, (2,2))
print('distributed value =', a)
"""

# Expected output
"""
distributed value = [[ 0.5  0.5]
 [ 0.5  0.5]]
"""