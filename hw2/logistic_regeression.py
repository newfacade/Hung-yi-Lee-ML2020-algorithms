# -*- coding: utf-8 -*-
import numpy as np
from utils.activation import sigmoid


class LogisticRegression:
    def __init__(self, iter_num=1000, lamb=0.001, learning_rate=0.005):
        self.iter_num = iter_num
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.w = None
        self.loss = None

    def fit(self, x, y):
        """
        gradient descent
        """
        x = np.array(x)
        y = np.array(y).reshape(-1, 1)
        assert x.shape[0] == y.shape[0]
        n, p = x.shape
        self.w = np.random.rand(p, 1) / np.sqrt(n)
        epsilon = 10e-5
        for index in range(self.iter_num):
            diff = sigmoid(np.dot(x, self.w)) - y
            grad = np.dot(x.T, diff) / n + self.lamb * self.w
            self.w -= self.learning_rate * grad
            y_hat = sigmoid(np.dot(x, self.w))
            cross_entropy = -(np.dot(y.T, np.log(np.maximum(y_hat, epsilon))) +
                              np.dot(1 - y.T, np.log(np.maximum(1 - y_hat, epsilon))))
            self.loss = (cross_entropy / n + self.lamb * np.dot(self.w.T, self.w)) / 2
            if index % 10 == 0:
                print("loss: %f" % self.loss)

    def coefficient(self):
        print(self.w)
        return self.w

    def loss(self):
        print(self.loss)
        return self.loss

    def predict(self, x):
        x = np.array(x)
        assert x.shape[1] == self.w.shape[0]
        return sigmoid(np.dot(x, self.w))

