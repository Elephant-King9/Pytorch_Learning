import time
import torch
import torchvision.transforms
from torch.utils.tensorboard import SummaryWriter

from CIFAR10Model import *

# 1.创建训练数据集
train_dataset = torchvision.datasets.CIFAR10(root='../dataset', train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root='../dataset', train=False, download=True,
                                            transform=torchvision.transforms.ToTensor())
# 配置GPU为mps
device = torch.device("mps")


# 记录数据集大小
train_size = len(train_dataset)
test_size = len(test_dataset)

# 2.创建dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# 3.创建神经网络
model = CIFAR10Model().to(device)

# 4.设置损失函数与梯度下降算法
loss_fn = nn.CrossEntropyLoss().to(device)

learn_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

# 训练轮数
total_train_step = 0
total_test_step = 0

# 训练轮数
epoch = 20

# 创建TensorBoard
writer = SummaryWriter('./CIFAR10_logs')
# 5.开始训练
for i in range(epoch):
    # 将模型设置为训练模式
    print(f"----------------------------开启第{i+1}轮训练----------------------------")
    model.train()
    # 第i轮训练的次数
    pre_train_step = 0
    # 第i轮训练的总损失
    pre_train_loss = 0
    # 第i轮训练的起始时间
    start_time = time.time()
    for data in train_loader:
        # 训练基本流程
        inputs, labels = data
        # 加入gpu训练
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # 第i轮训练次数加一
        pre_train_step += 1
        pre_train_loss += loss.item()
        total_train_step += 1
        # 每100次输出一下
        if pre_train_step % 100 == 0:
            end_train_time = time.time()
            print(f'当前为第{i+1}轮训练,当前训练轮数为:{pre_train_step},已经过时间为:{end_train_time-start_time},当前训练次数的平均损失为:{pre_train_loss / pre_train_step}')
            # 添加可视化
            writer.add_scalar('train_loss', pre_train_loss / pre_train_step, total_train_step)
    print(f"----------------------------第{i + 1}轮训练完成----------------------------")
    # 设置为测试模式
    model.eval()
    # 第i轮训练集的总损失
    pre_test_loss = 0
    # 第i轮训练集的总正确次数
    pre_accuracy = 0
    print(f"----------------------------开启第{i + 1}轮测试----------------------------")
    # 配置没有梯度下降的环境
    with torch.no_grad():
        for data in test_loader:
            # 测试集流程
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # 定义参数
            pre_test_loss += loss.item()
            # 记录训练集的总正确率
            # argmax(1)代表横向判断,argmax(0)代表纵向判断
            pre_accuracy += outputs.argmax(1).eq(labels).sum().item()
    # 记录测试集运行完后的事件
    end_test_time = time.time()
    print(f'当前为第{i + 1}轮测试,已经过时间:{end_test_time - start_time},当前测试集的平均损失为:{pre_test_loss / test_size},当前在测试集的正确率为:{pre_accuracy / test_size}')
    writer.add_scalar('test_accuracy', pre_accuracy / test_size, i)
    print(f"----------------------------第{i + 1}轮测试完成----------------------------")
    # 保存每轮的训练模型
    torch.save(CIFAR10Model, f'./CIFAR10TrainModel{i}.pth')
    print(f"----------------------------第{i + 1}轮模型保存完成----------------------------")

writer.close()





