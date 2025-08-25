import torch 

x1 = 10 + 20 * torch.rand((2,5)) 
x2 = 5 + 5 * torch.rand((5,3))
#print(x1)

x3 = torch.mm(x1,x2) # 2x5 times 5x3 => 2x3
#print(x3)

matrix = torch.rand(5,5)
print(matrix, id(matrix))
matrix = matrix.matrix_power(3) # matrix^3
print(matrix, id(matrix))
matrix **= 3 # raises each element to the power of 3
print(matrix, id(matrix))

#z = torch.dot(x1,x2) 

# batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m)) # we have an additional dimension for the batch
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)
print(out_bmm.shape)

# Useful operations 

x = torch.empty(10).uniform_(1,10) 
sum_x = torch.sum(x, dim =0)
print(sum_x.dtype)
value = sum_x.item()
print(value, type(value))

x= torch.empty(10).uniform_(1,10) 
values, indices = torch.max(x, dim = 0) # the dimensions you want to reduce to
print(values, indices)

x = torch.mean(x.float(), dim = 0) # outputs the mean

z = torch.clamp(x, min = 0, max =10) # it s RELU function basically

boolma = torch.tensor([1,0,1,1,1], dtype = torch.bool)
z = torch.any(boolma) # any one in the tensor
z2 =torch.all(boolma) # all the values are non zero
print(z, z2)
