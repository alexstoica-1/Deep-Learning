import torch

x = 1 + 9 * torch.rand(size = (5,5,5), dtype = torch.float16) 
#print(x.ndimension())
#print(x.numel())

a = torch.arange(9)
a_3x3 = a.view(3,3) 
a_3x33 = a.reshape(3,3)
print(a_3x3)

y = a_3x3.t() # .view() would not work on it 
print(y)
print(y.reshape(9))
print(y.contiguous().view(9))

x1 = torch.rand((1,10))
x2 = torch.rand((1,10))

f1 = torch.cat((x1,x2), dim =0)
f2 = torch.cat((x1,x2), dim =1)
print(f1.shape, f2.shape)

n = torch.arange(10)
n_0 = n.unsqueeze(0).shape
