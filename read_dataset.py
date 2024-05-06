from torch.utils.data import Dataset  # 用于导入基类Dataset,主要是大写

from PIL import Image  # 主要用于图像的操作
import os  # 文件操作


class MyData(Dataset):  # 创建一个MyData类，同时继承Dataset类
    def __init__(self, root_dir, label_dir):  # 类似于c++的构造函数
        # root_dir 一般设置为训练集文件夹的地址(train)
        # label_dir 一般设置为分类文件夹的地址(ants)
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)  # 这个函数的作用是将root_dir的地址与label_dir的地址拼接起来
        self.img_path = os.listdir(self.path)  # 将特定文件夹地址(path)中的所有文件列成一个list

    def __getitem__(self, index):  # 重写父类的方法
        img_name = self.img_path[index]  # 获取对应下标的图片名
        img_item_path = os.path.join(self.path, img_name)  # 获取图片路径
        img = Image.open(img_item_path)  # 根据图片路径打开图片
        # img.show()    展示图片
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


# root_dir 一般设置为训练集文件夹的地址(train)
# label_dir 一般设置为分类文件夹的地址(ants)
root_dir = "dataset/hymenoptera_data/train"
ant_label_dir = "ants"
bee_label_dir = "bees"
# 生成对应训练集的图片、标签列表
ants_dataset = MyData(root_dir, ant_label_dir)
bees_dataset = MyData(root_dir, bee_label_dir)

# 列表相加，前提是必须重载__len__方法
train_dataset = ants_dataset + bees_dataset
