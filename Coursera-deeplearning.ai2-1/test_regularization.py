# Test: regularization
import numpy as np
import matplotlib.pyplot as plt
from reg_utils import *
import sklearn
import sklearn.datasets
import scipy.io
from model_regularization import *
# from testCases import *


# Load data
data = scipy.io.loadmat('data.mat')
train_X = data['X'].T
train_Y = data['y'].T
test_X = data['Xval'].T
test_Y = data['yval'].T
# plt.scatter(train_X[0, :], train_X[1, :], c = train_Y.flatten(), s = 40, cmap = plt.cm.Spectral)

# Model learning
# parameters = model_regularization(train_X, train_Y)
# parameters = model_regularization(train_X, train_Y, lambd = 0.7)
parameters = model_regularization(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)
print("Train", end = " ")
predictions_train = predict(train_X, train_Y, parameters)
print("Test", end = " ")
predictions_test = predict(test_X, test_Y, parameters)

# Visualization
plt.figure(2)
# plt.title("Model without regularization")
# plt.title("Model with L2-regularization")
plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
x_min, x_max = train_X[0, :].min() - 1, train_X[0, :].max() + 1
y_min, y_max = train_X[1, :].min() - 1, train_X[1, :].max() + 1
h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = predict_dec(parameters, np.c_[xx.ravel(), yy.ravel()].T)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap = plt.cm.Spectral)
plt.ylabel('x2')
plt.xlabel('x1')
plt.scatter(train_X[0, :], train_X[1, :], c = train_Y.flatten(), cmap = plt.cm.Spectral)

plt.show()