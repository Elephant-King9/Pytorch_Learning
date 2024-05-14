import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./dataset', transform=torchvision.transforms.ToTensor(), train=False,
                                       download=False)

dataLoader = DataLoader(dataset, batch_size=64, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


net = Net()

writer = SummaryWriter(log_dir='./logs')

i = 0
for data in dataLoader:
    img, target = data
    output = net.forward(img)
    # print(output.shape)
    writer.add_images('input', img, i)

    # -1是一个占位符，让Pytorch自动计算维度大小
    output = torch.reshape(output,(-1, 3, 30, 30))
    # 无法直接传入6通道，只能3通道
    writer.add_images("output", output, i)
    i = i + 1

writer.close()
