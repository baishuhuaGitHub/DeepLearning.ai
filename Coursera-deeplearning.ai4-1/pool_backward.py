# Function: pool_backward
import numpy as np
from create_mask_from_window import *
from distribute_value import *


def pool_backward(dA, cache, mode = "max"):
    """
    Implement the backward pass of the pooling layer

    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, containing the layer's input and hparameters
    mode -- the pooling mode "max" or "average"

    Returns:
    dA_prev -- gradient of the cost with respect to the input of the pooling layer, same shape as A_prev
    """
    
    # Size and dimension
    (A_prev, hparameters) = cache
    stride = hparameters["stride"]
    f = hparameters["f"]
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (m, n_H, n_W, n_C) = dA.shape

    dA_prev = np.zeros(A_prev.shape)            # Initialize
    
    # Loop
    for i in range(m):
        a_prev = A_prev[i, :, :, :]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # Find the current "slice"
                    vert_start = h * stride
                    vert_end = h * stride + f
                    horiz_start = w * stride
                    horiz_end = w * stride + f
                    
                    # Compute the backward propagation in both modes
                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                    elif mode == "average":
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(dA[i, h, w, c], (f, f))

    assert(dA_prev.shape == A_prev.shape)

    return dA_prev


# Test
"""
from pool_forward import *
np.random.seed(1)
A_prev = np.random.randn(5, 5, 3, 2)
hparameters = {"stride" : 1, "f": 2}
A, cache = pool_forward(A_prev, hparameters)
dA = np.random.randn(5, 4, 2, 2)
dA_prev = pool_backward(dA, cache, mode = "max")
print("mode = max")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1,1] = ', dA_prev[1,1])  
print()
dA_prev = pool_backward(dA, cache, mode = "average")
print("mode = average")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1,1] = ', dA_prev[1,1]) 
"""

# Expected output
"""
mode = max
mean of dA =  0.145713902729
dA_prev[1,1] =  [[ 0.          0.        ]
 [ 5.05844394 -1.68282702]
 [ 0.          0.        ]]

mode = average
mean of dA =  0.145713902729
dA_prev[1,1] =  [[ 0.08485462  0.2787552 ]
 [ 1.26461098 -0.25749373]
 [ 1.17975636 -0.53624893]]
"""