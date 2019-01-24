# Initialize test
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from model import *
from init_utils import *

# Generate train and test data
np.random.seed(1)
train_X, train_Y = sklearn.datasets.make_circles(n_samples = 300, noise = 0.05)
np.random.seed(2)
test_X, test_Y = sklearn.datasets.make_circles(n_samples = 100, noise = 0.05)
# plt.scatter(train_X[:, 0], train_X[:, 1], c = train_Y.flatten(), s = 40, cmap = plt.cm.Spectral)
train_X = train_X.T
train_Y = train_Y.reshape(1, train_Y.shape[0])
test_X = test_X.T
test_Y = test_Y.reshape(1, test_Y.shape[0])

# Model training
# parameters = model(train_X, train_Y, initialization = "zeros")
# parameters = model(train_X, train_Y, initialization = "random")
parameters = model(train_X, train_Y, initialization = "he")

# Predictions
print("Train set", end = " ")
predictions_train = predict(train_X, train_Y, parameters)
print("Test set", end = " ")
predictions_test = predict(test_X, test_Y, parameters)

# Visualization
x_min, x_max = train_X[0, :].min() - 1, train_X[0, :].max() + 1
y_min, y_max = train_X[1, :].min() - 1, train_X[1, :].max() + 1
h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = predict_dec(parameters, np.c_[xx.ravel(), yy.ravel()].T)
Z = Z.reshape(xx.shape)
plt.figure(2)
plt.contourf(xx, yy, Z, cmap = plt.cm.Spectral)
plt.ylabel('x2')
plt.xlabel('x1')
plt.scatter(train_X[0, :], train_X[1, :], c=train_Y.flatten(), cmap = plt.cm.Spectral)

plt.show()


