import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                            download=True)
# 这里采用测试集是因为测试集较小，运行较快

test_dataLoader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

writer = SummaryWriter(log_dir='./logs')
i = 0
for data in test_dataLoader:  # 从test_dataLoader中取出data
    imgs, labels = data
    print(imgs.shape)
    writer.add_images('test_loader1', imgs, i)       # 注意这是add_images
    i = i + 1

writer.close()