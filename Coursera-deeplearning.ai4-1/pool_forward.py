# Function: pool_forward
import numpy as np


def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer

    Arguments:
    A_prev -- output activations of the previous layer of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- pooling mode "max" or "average"

    Returns:
    A -- output of the pooling layer of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward of pooling layer, containing the input and hparameters
    """
    
    # Size and dimension
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_C_prev - f) / stride) + 1
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))            # Initialize
      
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # Find the current "slice"
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    # Compute pooling operation on the slice
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    if mode == "average":
                        A[i, h, w, c] = np.average(a_prev_slice)

    assert(A.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, hparameters)

    return A, cache


# Test
"""
np.random.seed(1)
A_prev = np.random.randn(2, 4, 4, 3)
hparameters = {"stride" : 2, "f": 3}
A, cache = pool_forward(A_prev, hparameters)
print("mode = max")
print("A =", A)
print()
A, cache = pool_forward(A_prev, hparameters, mode = "average")
print("mode = average")
print("A =", A)
"""

# Expected output
"""
mode = max
A = [[[[ 1.74481176  0.86540763  1.13376944]]]

 [[[ 1.13162939  1.51981682  2.18557541]]]]

mode = average
A = [[[[ 0.02105773 -0.20328806 -0.40389855]]]

 [[[-0.22154621  0.51716526  0.48155844]]]]
"""