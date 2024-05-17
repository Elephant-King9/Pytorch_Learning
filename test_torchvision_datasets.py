import torchvision
from torch.utils.tensorboard import SummaryWriter

# 配置tansform的对象
trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

#在transform中添加刚刚的配置
train_set = torchvision.datasets.CIFAR10(root="./dataset", transform=trans, train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", transform=trans, train=False, download=True)


# 引入TensorBoard
writer = SummaryWriter("logs")
for i in range(10):
    img,label = train_set[i]
    writer.add_image("train", img, i)   # 将训练数据集中的img添加到writer中
writer.close()
