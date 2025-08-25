import torch
import numpy as np
"""
	•	Input = matrix → output = diagonal vector.
	•	Input = vector → output = diagonal matrix.
"""
x = torch.diag(torch.rand(size = (3,3)))
print(torch.diag(torch.rand(3)))
print(x)

tensor = torch.arange(4)
print(tensor, '\n', tensor.bool(), tensor.short(), tensor.half(), tensor.float(), tensor.double()) # types of floats

# conversion from numpy array
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()