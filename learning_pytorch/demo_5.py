import torch

batch_size = 10 # number of examples 
features = 25

x = torch.rand((batch_size,features))

print(x[0].shape) # x[0,:]

print(x[:,0].shape) # all values of a specific feature

print(x[2, 0:10]) # 0:10 creates [0, 1, 2, ..., 9]

x[0,0] = 100 # attribute values
print(x[0,0])

a = torch.arange(1,10,2)
indices = [1,4]
print(a[indices]) # picks certain values out 

print()

matrix = torch.rand(size = (3,5))
print(matrix)
rows = torch.tensor([1,0])
cols = torch.tensor([4,1])
print(matrix[rows, cols]) # picks the 2 elements 


n = torch.arange(10)

print(n[(n < 2) | (n > 8)]) # use or as |

print(n[n.remainder(2) == 0]) # picks out even elements

print(torch.where( n >5, n, n*2))