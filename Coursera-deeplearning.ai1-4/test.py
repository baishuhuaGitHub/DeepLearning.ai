# Test for cat vs non-cat pictures
import numpy as np
import matplotlib.pyplot as plt
import h5py
from dnn_model import *
from predict import *

np.random.seed(1)

# Load original data
train_dataset = h5py.File('train_catvnoncat.h5', "r")
train_x_orig = np.array(train_dataset["train_set_x"][:])        # Train set features
train_y_orig = np.array(train_dataset["train_set_y"][:])        # Train set labels

test_dataset = h5py.File('test_catvnoncat.h5', "r")
test_x_orig = np.array(test_dataset["test_set_x"][:])           # Test set features
test_y_orig = np.array(test_dataset["test_set_y"][:])           # Test set labels
classes = np.array(test_dataset["list_classes"][:])  # the list of classes

m_train = train_x_orig.shape[0]
m_test = test_x_orig.shape[0]
num_px = train_x_orig.shape[1]

# Reshape original data
train_x = train_x_orig.reshape(m_train, -1).T / 255
train_y = train_y_orig.reshape(1, train_y_orig.shape[0])
test_x = test_x_orig.reshape(m_test, -1).T / 255
test_y = test_y_orig.reshape(1, test_y_orig.shape[0])

"""
# Example of a picture
index = 25
plt.figure(1)
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

# Information about train and test data
print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_x shape: " + str(train_x.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_x shape: " + str(test_x.shape))
print ("test_y shape: " + str(test_y.shape))
"""

# DNN Model
layer_dims = [12288, 20, 7, 5, 1]
parameters = dnn_model(train_x, train_y, layer_dims, num_iterations = 2500, print_cost = True)

# Predict Accuracy
train_prediction, train_accuracy = predict(train_x, train_y, parameters)
test_prediction, test_accuracy = predict(test_x, test_y, parameters)
print("Train Accuracy = " + str(train_accuracy))
print("Test Accuracy = " + str(test_accuracy))

# Mislabeled_images
test_actual = test_prediction + test_y
mislabeled_indices = np.asarray(np.where(test_actual == 1))
plt.rcParams['figure.figsize'] = (40.0, 40.0)       # set default size of plots
num_images = len(mislabeled_indices[0])
for i in range(num_images):
    index = mislabeled_indices[1][i]
    plt.subplot(2, num_images, i + 1)
    plt.imshow(test_x[:, index].reshape(64, 64, 3), interpolation='nearest')
    plt.axis('off')
    plt.title("Prediction: " + classes[int(test_prediction[0, index])].decode("utf-8") + " \n Class: " + classes[test_y[0, index]].decode("utf-8"))

plt.show()