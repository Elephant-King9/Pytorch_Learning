import torch
import torchvision

# 加载模型+参数
vgg16_first = torch.load("vgg16_method1.pth")

# 仅加载参数
vgg16_second = torchvision.models.vgg16(pretrained=False)
# 将参数加载到我们的空白模型中
vgg16_second.load_state_dict(torch.load("vgg16_method2.pth"))

if __name__ == '__main__':
    print(vgg16_first)
