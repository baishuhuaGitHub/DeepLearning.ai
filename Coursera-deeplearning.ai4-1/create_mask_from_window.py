# Function: create_mask_from_window
import numpy as np


def create_mask_from_window(x):
    """
    Create a mask from an input matrix x, to identify the max entry of x

    Arguments:
    x -- array of shape (f,f)

    Returns:
    mask -- array of the same as window, containing a True at the position corresponding to the max entry of x
    """
    
    mask = (x == np.max(x))

    return mask


# Test
"""
np.random.seed(1)
x = np.random.randn(2,3)
mask = create_mask_from_window(x)
print('x = ', x)
print("mask = ", mask)
"""

# Expected output
"""
x =  [[ 1.62434536 -0.61175641 -0.52817175]
 [-1.07296862  0.86540763 -2.3015387 ]]
mask =  [[ True False False]
 [False False False]]
"""