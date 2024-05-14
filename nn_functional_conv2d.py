import torch            # 用于创建tensor形式的input与kernel
import torch.nn.functional as F     # 用于完成卷积操作


# 输入
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

# 卷积核
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input,[1, 1, 5, 5])       # 将原本维度为2的转化为维度为4的，分别代表数量，通道(RGB)，长，宽
kernel = torch.reshape(kernel,[1, 1, 3, 3])

output = F.conv2d(input, kernel, stride=1, padding=0)



