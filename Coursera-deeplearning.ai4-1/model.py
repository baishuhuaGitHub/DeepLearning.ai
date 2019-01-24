# Function: model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.framework import ops
from create_placeholders import *
from initialize_parameters import *
from forward_propagation import *
from compute_cost import *
from cnn_utils import *


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009, num_epochs = 100, minibatch_size = 64, print_cost = True):
    """
    Implements a three-layer ConvNet in TensorFlow: Conv2d -> ReLU -> Maxpool -> Conv2d -> ReLU -> Maxpool -> Flatten -> Fullyconnected

    Arguments:
    X_train -- training set of shape (None, 64, 64, 3)
    Y_train -- training "label" of shape (None, n_y = 6)
    X_test -- test set of shape (None, 64, 64, 3)
    Y_test -- test "label" of shape (None, n_y = 6)
    learning_rate -- learning rate of optimization
    num_epochs -- number of epochs of optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- accuracy on the training set
    test_accuracy -- accuracy on the test set
    parameters -- parameters learnt by the model which can bee used to predict
    """

    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_Y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_Y)           # Create placeholders
    parameters = initialize_parameters()                        # Initialize parameters
    Z3 = forward_propagation(X, parameters)                     # Forward propagation
    cost = compute_cost(Z3, Y)                                  # Compute cost
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)        # Adam optimizer

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # Loop
        for epoch in range(num_epochs):
            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, temp_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches

            costs.append(minibatch_cost)
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))

        # Plot the cost
        plt.plot(np.squeeze(costs))
        plt.xlabel('Iterations (per tens)')
        plt.ylabel('cost')
        plt.title("Learning rate = " + str(learning_rate))

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters