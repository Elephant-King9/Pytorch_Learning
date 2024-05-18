import torch
import torchvision
from PIL import Image
from CIFAR10Model import *

# 图片目录
img_path = "../imgs/dog.png"

img = Image.open(img_path)

# <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=358x312 at 0x1009DAA90>
# 这里我们可以看到默认的格式不是RGB格式的 ,而我们训练出的数据集只能处理三通道，所以我们需要对通道数由RGBA转化为RGB形式
print(img)

# 将图片转化为RGB格式
img = img.convert('RGB')

# <PIL.Image.Image image mode=RGB size=358x312 at 0x103002BE0>
print(img)

# 定义一个转化规则为transform，将图像转化为32x32像素，并且转化为tensor格式
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
img_tensor = transform(img)

# torch.Size([3, 32, 32])
print(img_tensor.shape)

img_tensor = torch.reshape(img_tensor, (1, 3, 32, 32))
# torch.Size([1, 3, 32, 32])
print(img_tensor.shape)

model = CIFAR10Model()
model = torch.load('../models/cifar10_model30.pth', map_location='cpu')

output = model(img_tensor)
# tensor([2])
print(output.argmax(1))

dataset = torchvision.datasets.CIFAR10(root='../dataset', train=False, download=False,
                                       transform=torchvision.transforms.ToTensor())
print(dataset.classes[output.argmax(1)])
