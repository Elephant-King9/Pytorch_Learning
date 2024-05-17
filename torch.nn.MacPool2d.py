import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./dataset', transform=torchvision.transforms.ToTensor(), train=False, download=False)


dataloader = DataLoader(dataset, batch_size=64)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        return x


net = Net()
writer = SummaryWriter(log_dir='./logs')
i = 0
for data in dataloader:
    img, target = data
    output = net(img)
    # torch.Size([64, 6, 10, 10])
    # print(output.shape)
    output = torch.reshape(output, (-1, 3, 10, 10))
    writer.add_images('output', output, i)
    i = i+1

writer.close()
