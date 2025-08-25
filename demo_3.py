import torch

x = torch.tensor([1,2,3], dtype = torch.float)
y = torch.tensor([9,8,7], dtype = torch.float)

# all do the same 
z1 = x + y
z2 = torch.add(x,y)
z3 = torch.empty(3)
torch.add(x,y, out = z3)

t = torch.zeros(3)
t.add_(y+x) # it will change the tensor in place
print(t)

y1 = x - y

# Division

div1 = y / x # div1 is a float tensor 
print(div1)

div2 = torch.div(y,x, rounding_mode="floor")
print(div2)

div3 = torch.div(y,x, rounding_mode="trunc")
print(div3)

y.div_(x) # in - place division
print(y)

# broadcasting
br1 = torch.tensor([[10.0, 20.0],[30.0, 40.0]])
br2 = torch.tensor([2.0, 3.0])

print(br1/br2)

# boolean mask
z = x > 0
print(z) 