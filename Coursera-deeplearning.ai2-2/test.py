# Test
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn.datasets
from opt_utils import *
from model import *


# Generate data
np.random.seed(3)
train_X, train_Y = sklearn.datasets.make_moons(n_samples = 300, noise =.2)  # 300 #0.2
train_X = train_X.T
train_Y = train_Y.reshape((1, train_Y.shape[0]))
layers_dims = [train_X.shape[0], 5, 2, 1]

# Train 3-layer model
# parameters = model(train_X, train_Y, layers_dims, optimizer = "gd")
# parameters = model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")
parameters = model(train_X, train_Y, layers_dims, optimizer = "adam")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.figure(3)
# plt.title("Model with Gradient Descent optimization")
# plt.title("Model with Momentum optimization")
plt.title("Model with Adam optimization")
plt.xlim(-1.5, 2.5)
plt.ylim(-1, 1.5)
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

plt.show()