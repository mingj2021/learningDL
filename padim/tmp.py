import torch

a = torch.tensor((1,1,0,1),dtype=torch.bool)
b = ~a
print(b)