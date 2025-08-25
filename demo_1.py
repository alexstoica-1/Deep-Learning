import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype = torch.float32,
                        device = device, requires_grad = True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# creates an empty 3x3 matrix, not necessarly 0s
x = torch.empty(size = (3,3))
print(x)
y_1 , y_2 = torch.zeros(size = (3,3)), torch.ones(size = (3,3)) 
print(y_1, y_2)
z = torch.rand((3,3))
print(z)
id = torch.eye(3,3) #directly, the identity 
print(id)

arr = torch.arange(0,5,1)
print(arr)

# 10 values in between the given values, ends included 
x = torch.linspace(0.1, 1, 10) 
print(x)

bell = torch.empty(size = (1,5)).normal_(0,1)
print(bell)