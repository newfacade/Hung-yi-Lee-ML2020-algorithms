# -*- coding: utf-8 -*-
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(np.minimum(-x, 100)))
