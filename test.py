import numpy as np
import torch

a = np.array([[1, 2], [3, 4]])
print(a.shape)
b = torch.LongTensor(a)
print(b)
