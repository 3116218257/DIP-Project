from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os
from random import sample
from torchvision import transforms

Image_Size = 256

class DataSet(Dataset):
    def __init__(self, image_dir, label_dir = None, test=False):
        # Initialize the dataset with image file paths, labels, and augmentation option
        self.label_dir = label_dir
        self.images = []
        self.labels = []
        self.PILToTensor = transforms.ToTensor()
        self.resize = transforms.Resize((Image_Size, Image_Size))
        self.augmentation = self.get_transform()

        if (label_dir != None): # 训练与验证
            label_file = pd.read_csv(label_dir)                                                                                                                                                
            for _,row in label_file.iterrows(): 
                image_name = row['image name']
                image_path = os.path.join(image_dir, image_name)
                image = Image.open(image_path).convert('RGB')
                image = self.resize(image)
                label = int(row['image quality level'])

                for _ in range(4):
                    self.images.append(image)
                    self.labels.append(label)
        else:   # 测试
            self.image_name_list = os.listdir(image_dir)
            self.image_name_list.sort(key=lambda x:int(x.split('.')[0]))
            for image_name in self.image_name_list:
                image_path = os.path.join(image_dir, image_name)
                image = Image.open(image_path).convert('RGB')
                image = self.resize(image)
                image = self.PILToTensor(image) 
                self.images.append(image)
                self.labels.append(-1)

        print("Building Dataset Complete.")

    def get_transform(self):
        aug_settings={
            'HorizontalFlip': True,
            'VerticalFlip': True,
            # 'contrast': 0.4,
            # 'scale': (0.8, 1.2), 
            # 'ratio': (0.8, 1.2), 
            # 'brightness': 0.4,
            # 'mean': [0.485, 0.456, 0.406],
            # 'std': [0.229, 0.224, 0.225]
        }

        transforms_list = []

        if aug_settings['HorizontalFlip']:
            transforms_list.append(transforms.RandomHorizontalFlip())
        if aug_settings['VerticalFlip']:
            transforms_list.append(transforms.RandomVerticalFlip())
        
        transforms_list.append(transforms.RandomResizedCrop(
            size=(Image_Size, Image_Size),
            # scale=aug_settings['scale'],
            # ratio=aug_settings['ratio']
        ))

        # transforms_list.append(transforms.ColorJitter(
        #     brightness=aug_settings['brightness'],
        #     contrast=aug_settings['contrast']
        # ))

        # transforms_list.extend([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=aug_settings['mean'], std=aug_settings['std'])
        # ])

        return transforms.Compose(transforms_list)


    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if (self.label_dir != None):
            image = self.augmentation(image)
            image = self.resize(image)
            image = self.PILToTensor(image) 
        return image, label

    def __len__(self):
        return len(self.images)

