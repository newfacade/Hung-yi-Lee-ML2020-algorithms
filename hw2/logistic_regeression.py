# -*- coding: utf-8 -*-
import numpy as np


def sigmoid(x):
    """
    can prevent overflow
    """
    return 1 / (1 + np.exp(np.minimum(-x, 100)))


class LogisticReg:
    """
    逻辑回归
    """

    def __init__(self, max_iter=1000, c=1.0, learning_rate=0.005):
        self.max_iter = max_iter
        self.c = c
        self.learning_rate = learning_rate
        self.weight = None
        self.loss = None

    def fit(self, x, y):
        """
        gradient descent
        """
        x = np.array(x)
        x = np.concatenate((np.ones(x.shape[0], 1), x), axis=1)
        y = np.array(np.array(y).reshape(-1, 1))
        assert x.shape[0] == y.shape[0]
        n, p = x.shape
        self.weight = np.random.rand(p, 1) / np.sqrt(n)
        epsilon = 10e-5
        for index in range(self.max_iter):
            grad = self.c * np.dot(x.transpose(), sigmoid(np.dot(x, self.weight)) - y) / n + self.weight
            self.weight -= self.learning_rate * grad
            y_hat = sigmoid(np.dot(x, self.weight))
            cross_entropy = -(np.dot(y.transpose(), np.log(np.maximum(y_hat, epsilon))) +
                              np.dot(1 - y.transpose(), np.log(np.maximum(1 - y_hat, epsilon))))
            self.loss = (self.c * cross_entropy / n + np.sum(np.power(self.weight, 2))) / 2
            if index % 10 == 0:
                print("loss: %f" % self.loss)

    def predict(self, x):
        """
        预测
        """
        x = np.array(x)
        x = np.concatenate((np.ones(x.shape[0], 1), x), axis=1)
        assert x.shape[1] == self.weight.shape[0]
        return sigmoid(np.dot(x, self.weight))

