import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils import data
import os
from PIL import Image
import random

means = [0.40971587,0.44913619,0.48582455]
stds = [0.22410078,0.22339204,0.22840099]
dogcat_label = {"none": 0, "tongue": 1}
random.seed(1)

class Portrait_dataset(data.Dataset):
    def __init__(self,data_dir, transform=None ):
        self.label_name = {"none": 0, "tongue": 1}
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform


    def __getitem__(self,index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')  # 0~255

        if self.transform is not None:
            img = self.transform(img)  # 在这里做transform，转为tensor等等

        return img, label



    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):

            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root,sub_dir,img_name)
                    label = dogcat_label[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info

    def __len__(self):
        return len(self.data_info)


def imgpre():

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop((224, 224), pad_if_needed=True),
        transforms.ToTensor(),
        transforms.Normalize((0.410, 0.449, 0.486), (0.224, 0.223, 0.228))

    ])

    test_transforms = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.410, 0.449, 0.486), (0.224, 0.223, 0.228))

    ])
    train_data = Portrait_dataset('./data/train', train_transforms)
    valid_data = Portrait_dataset('./data/valid', test_transforms)

    # train_data = datasets.ImageFolder('./data/train', train_transforms)
    # valid_data = datasets.ImageFolder('./data/valid', test_transforms)

    # print(f'Number of training examples: {len(train_data)}')
    # print(f'Number of validation examples: {len(valid_data)}')

    BATCH_SIZE = 8;

    train_iterator = data.DataLoader(dataset=train_data, shuffle=True, batch_size=BATCH_SIZE)
    valid_iterator = data.DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)


    return train_iterator,valid_iterator

if __name__ == '__main__':
    imgpre()