import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./dataset', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataLoader = torch.utils.data.DataLoader(dataset, batch_size=64)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


net = Net()
criterion = nn.CrossEntropyLoss()  # 损失函数
optim = torch.optim.SGD(net.parameters(), lr=0.1)

# 将整个CIFAR10数据集进行20次循环训练
for epoch in range(20):
    res_loss = 0.0
    for data in dataLoader:
        optim.zero_grad()
        img, target = data
        output = net(img)  # 模型经过CNN后得出的概率值
        loss = criterion(output, target)  # 根据从模型训练出来的概率值与目标值target进行交叉熵损失函数求解
        loss.backward()  # 计算本次参数的梯度
        optim.step()
        res_loss = res_loss + loss
    print(res_loss)
