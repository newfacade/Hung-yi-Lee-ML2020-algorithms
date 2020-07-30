# -*- coding: utf-8 -*-
import numpy as np


class LinearReg:
    """
    线性回归
    """
    def __init__(self, max_iter=100, learning_rate=0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weight = None
        self.loss = None

    def fit(self, x, y):
        """
        gradient descent
        """
        x = np.array(x)
        x = np.concatenate((np.ones(x.shape[0], 1), x), axis=1)
        y = np.array(y).reshape(-1, 1)
        assert x.shape[0] == y.shape[0]
        n, p = x.shape
        self.weight = np.random.rand(p, 1) / np.sqrt(n)
        for index in range(self.max_iter):
            grad = np.dot(x.transpose(), np.dot(x, self.weight) - y) / n
            self.weight -= self.learning_rate * grad
            self.loss = np.sum(np.power(np.dot(x, self.weight) - y, 2)) / n / 2
            if index % 10 == 0:
                print("loss: %f" % self.loss)

    def predict(self, x):
        """
        预测
        """
        x = np.array(x)
        x = np.concatenate((np.ones(x.shape[0], 1), x), axis=1)
        assert x.shape[1] == self.weight.shape[0]
        return np.dot(x, self.weight)


