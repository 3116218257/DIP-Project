from PIL import Image, ImageFilter
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms

Image_Size = 256

class DataSet(Dataset):
    def __init__(self, image_paths, labels, augment=False):
        # Initialize the dataset with image file paths, labels, and augmentation option
        self.image_paths = image_paths
        self.labels = labels
        self.augment = augment
        self.transform = self.get_transform()
        self.transform_to_tensor = transforms.ToTensor()
    
    def get_transform(self):
        aug_settings={
            'HorizontalFlip': True,
            'VerticalFlip': True,
            'contrast': 0.4,
            'scale': (0.8, 1.2), 
            'ratio': (0.8, 1.2), 
            'brightness': 0.4,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }

        transforms_list = [
            transforms.Resize((Image_Size, Image_Size)),
        ]

        if self.augment:
            if aug_settings['HorizontalFlip']:
                transforms_list.append(transforms.RandomHorizontalFlip())
            if aug_settings['VerticalFlip']:
                transforms_list.append(transforms.RandomVerticalFlip())
            
            # transforms_list.append(transforms.RandomResizedCrop(
            #     size=(224, 224),
            #     scale=aug_settings['scale'],
            #     ratio=aug_settings['ratio']
            # ))

            # transforms_list.append(transforms.ColorJitter(
            #     brightness=aug_settings['brightness'],
            #     contrast=aug_settings['contrast']
            # ))

        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=aug_settings['mean'], std=aug_settings['std'])
        ])

        return transforms.Compose(transforms_list)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')
        
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)
