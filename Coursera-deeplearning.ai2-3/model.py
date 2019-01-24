# Function: model
import tensorflow as tf
from create_placeholders import *
from initialize_parameters import *
from forward_propagation import *
from compute_cost import *
from tf_utils import *
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implement a three-layer tensorflow neural netword: Linear -> ReLU -> Linear -> ReLU -> Linear ->Softmax

    Arguments:
    X_train -- training set of shape (input size, number of training examples)
    Y_train -- training label of shape (output size, number of training examples)
    X_test -- test set of shape (input size, number of test samples)
    Y_test -- test label of shape (output size, number of test samples)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model which can be used to predict
    """

    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X, Y = create_placeholders(n_x, n_y)	# Create placeholders
    parameters = initialize_parameters()	# Initialize parameters
    Z3 = forward_propagation(X, parameters)	# Forward propagation
    cost = compute_cost(Z3, Y)			    # Compute cost

    # Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # Initialize all variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.					                # Define cost related to an epoch
            num_minibatches = int(m / minibatch_size)		# Number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y:minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost
            if epoch % 5 == 0:
                costs.append(epoch_cost)
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
        
        # Plot the cost
        plt.plot(np.squeeze(costs))
        plt.xlabel("iterations (per tens)")
        plt.ylabel("cost")
        plt.title("Learning_rate =" + str(learning_rate))

        # Save parameters
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Prediction
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy:", accuracy.eval({X: X_train, Y:Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y:Y_test}))

        return parameters