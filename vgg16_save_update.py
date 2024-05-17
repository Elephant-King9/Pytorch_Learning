import torch
import torchvision
from torch import nn

# 训练好的vgg网络
vgg_train = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
# vgg_test = torchvision.models.vgg16(pretrained=True)


# 未训练的vgg网络，只有网络结构
vgg_not_train = torchvision.models.vgg16(weights=None)
# vgg_not_train = torchvision.models.vgg16(pretrained=True)

# 保存神经网络,保存所有信息(结构+参数)
torch.save(vgg_train, 'vgg16_method1.pth')
# 保存神经网络参数
torch.save(vgg_train.state_dict(), 'vgg16_method2.pth')

vgg_not_train = torchvision.models.vgg16(weights=None)

# 已训练好的网络的修改
vgg_not_train.classifier[6] = nn.Linear(4096, 10)
print(vgg_not_train)
