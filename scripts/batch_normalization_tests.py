import torch
import torch.nn as nn

inp = torch.randn(1, 2, 1, 3)
print(inp)
bn = nn.BatchNorm2d(2)

out = bn(inp)
print("Output shape after BatchNorm2d:", out.shape)
print(out )