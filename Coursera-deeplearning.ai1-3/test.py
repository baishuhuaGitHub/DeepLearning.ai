# Test: binary classification for colored points
import numpy as np
import matplotlib.pyplot as plt
from nn_model import *
from predict import *
import sklearn
import sklearn.datasets
import sklearn.linear_model

# Generate training data
np.random.seed(1)
m = 400                 # number of samples
N = int(m / 2)          # number of points per class
D = 2                   # dimension
X = np.zeros((m, D))    # data matrix where each row is a single example
Y = np.zeros((m ,1), dtype = 'uint8')   # labels vector (0 for red, 1 for blue)
a = 4                   # maximum radius of the flower
for j in range(2):
    ix = range(N * j, N * (j + 1))
    theta = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
    radius = a * np.sin(4 * theta) + np.random.randn(N) * 0.2
    X[ix] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
    Y[ix] = j
X = X.T
Y = Y.T

# Model training
parameters = nn_model(X, Y, nh = 4, num_iterations = 10000, print_cost = True)

# Accuracy
predictions = predict(X, parameters)
print('Training accuracy = %.2f ' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

# Binary classification
x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
h = 0.01        # Meshgrid interval
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))        # Generate a grid of points with interval h
Z = predict(np.c_[xx.ravel(), yy.ravel()].T, parameters)
Z = Z.reshape(xx.shape)
plt.figure(2)
plt.contourf(xx, yy, Z, cmap = plt.cm.Spectral)
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(X[0, :], X[1, :], c = Y.flatten(), s = 40, cmap = plt.cm.Spectral)
plt.title("Decision Boundary for hidden layer size " + str(4))

# Tuning hidden layer size
hidden_layer_sizes = [1, 3, 4, 5, 6, 20]
plt.figure(3)
for i, nh in enumerate(hidden_layer_sizes):
    plt.subplot(3, 2, i + 1)
    plt.title('Hidden layer of size %d' % nh)
    parameters = nn_model(X, Y, nh, num_iterations = 5000)
    Z = predict(np.c_[xx.ravel(), yy.ravel()].T, parameters)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(X[0, :], X[1, :], c=Y.flatten(), s=40, cmap=plt.cm.Spectral)
    predictions = predict(X, parameters)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print('Training accuracy for {} hidden units: {} % '.format(nh, accuracy))

plt.show()