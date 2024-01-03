from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os
from random import sample
from torchvision import transforms

Image_Size = 512

class DataSet(Dataset):
    def __init__(self, image_dir, label_dir = None, indexes = None):
        # Initialize the dataset with image file paths, labels, and augmentation option
        self.image_names = []
        self.labels = []

        if (label_dir != None):
            label_file = pd.read_csv(label_dir)                                                                                                                                                
            for image_num, row in label_file.iterrows(): 
                image_name = row['image name']
                label = int(row['image quality level'])

                if (indexes == None or image_num in indexes):
                    for _ in range(10):
                        self.image_names.append(image_name)
                        self.labels.append(label)
        else:
            self.image_names = os.listdir(image_dir)
            self.image_names.sort(key=lambda x:int(x.split('.')[0]))
            for image_name in self.image_names:
                self.labels.append(-1)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.labels)


class StageDataSet(Dataset):
    def __init__(self, image_dir, original_dataset, stage):
        super().__init__()

        self.image_dir = image_dir
        self.images = [] 
        self.image_names = []
        self.labels = [] 
        self.PILToTensor = transforms.ToTensor()
        self.resize = transforms.Resize((Image_Size, Image_Size))
        self.augment = self.get_transform()

        for image_num in range(len(original_dataset.labels)):
            label = original_dataset.labels[image_num]
            image_name = original_dataset.image_names[image_num]

            # 先分01和2
            # if (stage == 0):
            #     if (label == 0 or label == 1):
            #         self.images.append(image)
            #         self.labels.append(0)
            #     else:
            #         self.images.append(image)
            #         self.labels.append(1)
                
            # if (stage == 1):
            #     if (label == 0 or label == 1):
            #         self.images.append(image)
            #         self.labels.append(label)

            # 先分0和12
            if (stage == 0):
                if (label == 0):
                    self.image_names.append(image_name)
                    self.labels.append(0)
                else:
                    self.image_names.append(image_name)
                    self.labels.append(1)
                
            if (stage == 1):
                if (label == 1 or label == 2):
                    self.image_names.append(image_name)
                    self.labels.append(label - 1)

    def get_transform(self):
        aug_settings={
            'HorizontalFlip': True,
            'VerticalFlip': True,
            'contrast': 0.4,
            # 'scale': (0.8, 1.2), 
            # 'ratio': (0.8, 1.2), 
            'brightness': 0.4,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
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

        transforms_list.append(transforms.ColorJitter(
            brightness=aug_settings['brightness'],
            contrast=aug_settings['contrast']
        ))

        # transforms_list.extend([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=aug_settings['mean'], std=aug_settings['std'])
        # ])

        return transforms.Compose(transforms_list)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        image = self.augment(image)
        image = self.resize(image)
        image = self.PILToTensor(image) 

        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.image_names)


class ValDataSet(Dataset):
    def __init__(self, image_dir, original_dataset, indexes=None):
        super().__init__()

        self.images = [] 
        self.labels = original_dataset.labels
        self.PILToTensor = transforms.ToTensor()
        self.resize = transforms.Resize((Image_Size, Image_Size))

        for image_num in range(len(self.labels)):
            if (indexes == None or image_num in indexes):
                image_name = original_dataset.image_names[image_num]
                image_path = os.path.join(image_dir, image_name)
                image = Image.open(image_path).convert('RGB')
                image = self.resize(image)
                image = self.PILToTensor(image) 
                self.images.append(image)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.images)
        
