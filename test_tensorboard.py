from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
writer = SummaryWriter("logs")  # 单参数传递，说明将我们的事件和文件存储到logs的文件夹下
img_path = "dataset/hymenoptera_data/train/ants/0013035.jpg"
img_PIL = Image.open(img_path)
img_np = np.array(img_PIL)  # 将其他数据结构转化为numpy类型
print(img_np.shape)
# 主要的方法
writer.add_image("test", img_np, 1, dataformats='HWC')  # 添加图片
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)  # 添加标量


writer.close()  # 关闭
