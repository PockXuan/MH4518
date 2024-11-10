import torch

# Jackel regularisation
a = torch.arange(9).reshape(3,3)
b = (torch.arange(9)+9).reshape(3,3)

print(a)
print(b)
print(torch.stack((a, b), dim=2).flatten().reshape(3,6))