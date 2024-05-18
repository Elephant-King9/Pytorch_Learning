import time

import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 导入自定义的神经网络
from model import *

# 创建训练集与测试集
train_dataset = torchvision.datasets.CIFAR10('../dataset', train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10('../dataset', train=False, download=True,
                                            transform=torchvision.transforms.ToTensor())

# 记录训练集和测试集的数据量
train_size = len(train_dataset)
test_size = len(test_dataset)

# 创建dataLoader
train_dataLoader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

# 创建tensorboard,用于可视化
writer = SummaryWriter('./CIFAR10_logs')

# 创建自定义的神经网络
CIFAR10_model = CIFAR10Model()
# 用交叉熵损失函数作为损失函数
loss_f = nn.CrossEntropyLoss()
# 使用SGD优化器进行梯度下降,学习率为learn_rate
learn_rate = 0.01
optimizer = torch.optim.SGD(CIFAR10_model.parameters(), lr=learn_rate)

# 总训练次数
train_total = 0
# 总测试次数
test_total = 0

# 训练轮数
epoch = 20


start_time = time.time()
for i in range(epoch):
    # 神经网络的的训练轮数1~20
    # 本轮训练的损失
    train_loss = 0
    # 将模型设置为训练
    CIFAR10_model.train()
    for data in train_dataLoader:
        img, target = data
        optimizer.zero_grad()
        output = CIFAR10_model(img)
        loss = loss_f(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_total += 1
        # 每训练100次输出一下
        if train_total % 100 == 0:
            end_time = time.time()
            print(f"train_time:{end_time - start_time}")
            print(f'i: {i} train_total: {train_total}')
            writer.add_scalar('train_loss', loss.item(), train_total)
    print(f'--------------i: {i} train_loss: {train_loss}--------------')

    # 测试集
    # 测试集上的损失
    test_loss = 0
    # 测试集的正确率
    total_accuracy = 0
    # 将模型设置为测试
    CIFAR10_model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            img, target = data
            output = CIFAR10_model(img)
            loss = loss_f(output, target)

            # 计算损失
            test_loss += loss.item()

            # 计算正确率
            # argmax(1)代表横向判断,argmax(0)代表纵向判断
            accuracy = output.argmax(1).eq(target).sum().item()
            total_accuracy += accuracy

            test_total += 1
    print(f'--------------Test loss: {test_loss}--------------')
    # 添加函数图像
    writer.add_scalar('test_loss', test_loss, test_total)
    writer.add_scalar('test_accuracy', total_accuracy / test_size, test_total)
    # 保存本轮模型
    torch.save(CIFAR10_model, f'./CIFAR10_model_{i}.pth')
    print('--------------save model--------------')

writer.close()
