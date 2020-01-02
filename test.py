import torch


P = torch.tensor((0.1,0.8,0.1,0,0,0,0))+1e-7
Q = torch.tensor((0,0,0,0.1,0.8,0.1,0))+1e-7
print((P * (P / Q).log()).sum())
