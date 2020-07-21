# -*- coding: utf-8 -*-
import numpy as np


class LinearRegression:
    def __init__(self, iter_num=100, lamb=1.0, learning_rate=0.1, ):
        self.iter_num = iter_num
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.w = None
        self.loss = None

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        y.reshape(-1, 1)
        assert x.shape[0] == y.shape[0]
        n, p = x.shape
        self.w = np.random.rand(p, 1) / np.sqrt(n)
        for index in range(self.iter_num):
            diff = np.dot(x, self.w) - y
            grad = 2 * np.dot(x.T, diff) / n + 2 * self.lamb * self.w
            self.w -= self.learning_rate * grad
            self.loss = np.dot(diff.T, diff)
            if index % 100 == 0:
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
        return np.dot(x, self.w)


