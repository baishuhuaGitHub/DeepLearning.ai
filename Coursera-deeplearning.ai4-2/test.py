# Test
import tensorflow as tf
import keras
import numpy as np
from resnets_utils import *
from ResNet50 import *
import datetime

# Load dataset
starttime = datetime.datetime.now()
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

# Dataset shape
"""
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
"""

# Model training
model = ResNet50(input_shape = (64, 64, 3), classes = 6)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs = 20, batch_size = 32, verbose = 2)

# Predict
preds = model.evaluate(X_test, Y_test, verbose = 0)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

print((datetime.datetime.now() - starttime).seconds)