import torch
from torch import nn
from user_define_net_save import *


model = Net()
# 注意无法直接通过这条语句导入,需要先引入网络定义
model = torch.load('user_define_net_save.pth')
