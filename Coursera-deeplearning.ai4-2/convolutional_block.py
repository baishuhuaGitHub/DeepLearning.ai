# Function: convolutional_block
import tensorflow as tf
import keras
import numpy as np


def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implement the convolutional block

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the Conv's window
    filters -- python list of integers, defining the number of filters in the Conv layers
    stage -- integer, used to name the layers, depending their position in the network
    block -- string/character, used to name the layers, depending their position in the network
    s -- integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # Define name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters    # Retrieve filters
    X_shortcut = X          # Save the input value

    # First component of main path
    X = keras.layers.Conv2D(F1, (1, 1), strides = (s, s), name = conv_name_base + '2a',
                            kernel_initializer = keras.initializers.glorot_uniform(seed = 0))(X)
    X = keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = keras.layers.Activation('relu')(X)

    # Second component of main path
    X = keras.layers.Conv2D(F2, (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b',
                            kernel_initializer = keras.initializers.glorot_uniform(seed = 0))(X)
    X = keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = keras.layers.Activation('relu')(X)

    # Third component of main path
    X = keras.layers.Conv2D(F3, (1, 1), strides = (1, 1), name = conv_name_base + '2c',
                            kernel_initializer = keras.initializers.glorot_uniform(seed = 0))(X)
    X = keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Shortcut path
    X_shortcut = keras.layers.Conv2D(F3, (1, 1), strides = (s, s), name = conv_name_base + '1',
                                     kernel_initializer = keras.initializers.glorot_uniform(seed = 0))(X_shortcut)
    X_shortcut = keras.layers.BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step
    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)

    return X


# Test
"""
tf.reset_default_graph()
with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = convolutional_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict= {A_prev: X, keras.backend.learning_phase(): 0})
    print("out = " + str(out[0][1][1][0]))
"""

# Expected output
"""
out = [ 0.09018463  1.23489773  0.46822017  0.0367176   0.          0.65516603]
"""