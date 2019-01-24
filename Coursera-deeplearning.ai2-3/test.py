# Test
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tf_utils import *
from model import *

# Load data set
train_dataset = h5py.File('train_signs.h5', "r")
X_train_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
Y_train_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
test_dataset = h5py.File('test_signs.h5', "r")
X_test_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
Y_test_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels
classes = np.array(test_dataset["list_classes"][:])  # the list of classes
Y_train_orig = Y_train_orig.reshape((1, Y_train_orig.shape[0]))
Y_test_orig = Y_test_orig.reshape((1, Y_test_orig.shape[0]))

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

# Visualize data set
"""
print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
index = 0
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
"""

# Training
parameters = model(X_train, Y_train, X_test, Y_test)

plt.show()