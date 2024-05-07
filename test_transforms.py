from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import cv2
img_path = "dataset/hymenoptera_data/train/ants/0013035.jpg"
PIL_img = Image.open(img_path)
tensor_tans = transforms.ToTensor()     # 通过transforms中的ToTensor类创建一个对象
img = tensor_tans(PIL_img)     # __call__方法类似于c++中重载了()运算符，我们只需要传入PIL_img格式的图像就可以输出tensor格式的图像

writer = SummaryWriter(log_dir="logs")
writer.add_image("ants", img)
writer.close()