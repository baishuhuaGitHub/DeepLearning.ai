# Function: backward -- implement backward propagation for the [LINEAR->RELU]*(L-1)->[LINEAR->SIGMOID] group
from linear_activation_backward import *


def backward(AL, Y, caches):
    """
    Arguments:
    AL -- probability vector, output of the forward propagation (forward())
    Y -- true "label" vector
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], l = 0, ..., L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    
    grads = {}
    L = len(caches)        # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)    # After this line, Y is the same as AL
    
    # Initializing the backpropagation
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))        # derivative of cost with respect to AL

    # Layer of L-1, "sigmoid"
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    
    # Layer of others, "relu"
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads["dA" + str(l + 1)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")

    return grads
    

# Test
"""
np.random.seed(3)
AL = np.random.randn(1, 2)
Y = np.array([[1, 0]])

A1 = np.random.randn(4, 2)
W1 = np.random.randn(3, 4)
b1 = np.random.randn(3, 1)
Z1 = np.random.randn(3, 2)
linear_cache_activation_1 = ((A1, W1, b1), Z1)

A2 = np.random.randn(3, 2)
W2 = np.random.randn(1, 3)
b2 = np.random.randn(1, 1)
Z2 = np.random.randn(1, 2)
linear_cache_activation_2 = ((A2, W2, b2), Z2)

caches = (linear_cache_activation_1, linear_cache_activation_2)

grads = backward(AL, Y, caches)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dA2 = "+ str(grads["dA2"])) 
"""

# Expected output
"""
dW1 = [[ 0.41010002  0.07807203  0.13798444  0.10502167]
 [ 0.          0.          0.          0.        ]
 [ 0.05283652  0.01005865  0.01777766  0.0135308 ]]
db1 = [[-0.22007063]
 [ 0.        ]
 [-0.02835349]]
dA2 = [[ 0.12913162 -0.44014127]
 [-0.14175655  0.48317296]
 [ 0.01663708 -0.05670698]]
"""