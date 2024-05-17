import torch
from torch import nn


class Add(nn.Module):
    def __init__(self):
        super(Module, self).__init__()

    def forward(self, x):
        return x + 1;


module = Add()
x = torch.tensor(1.0)
print(x)
x = module.forward(x);
print(x)