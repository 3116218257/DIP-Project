import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

#############construct our dataset###############
class task_B_dataset(data.Dataset):
    def __init__(self, is_train=False, is_val=False, is_all=False, is_test=False, fold_num=0, augment=True):
        if is_train or is_all or is_val:
            image_path = 'data/1. Original Images/a. Training Set/'
            label_path = 'data/2. Groundtruths/a. DRAC2022_ Image Quality Assessment_Training Labels.csv'
        else:
            image_path = 'data/1. Original Images/b. Testing Set/'
        
        self.images = []
        if is_train or is_val:
            label_file = pd.read_csv(label_path)
            label_list = []
            image_list = []
            for _, row in label_file.iterrows():
                name = row['image name']
                label = int(row['image quality level'])
                label_list.append(label)
                image_list.append([image_path + name, label, name])
            skf = StratifiedKFold(n_splits=5)
            for index, (train_index, val_index) in enumerate(skf.split(np.zeros_like(label_list),label_list)):
                if index == fold_num:
                    break
            if is_train:
                for i in train_index:
                    self.images.append(image_list[i])
            else:
                for i in val_index:
                    self.images.append(image_list[i])
        
        elif is_test:
            path_list = os.listdir(image_path)
            path_list.sort(key=lambda x:int(x.split('.')[0]))
            for name in path_list:
                self.images.append([image_path + name, -1, name])
        
        elif is_all:
            label_file = pd.read_csv(label_path)
            for _, row in label_file.iterrows():
                name = row['image name']
                label = int(row['image quality level'])
                self.images.append([image_path + name, label, name])
        
        ####################augmentation#######################

        augmentations = {'contrast': 0.4, 'scale': (0.8, 1.2), 'ratio': (0.8, 1.2), 'brightness': 0.4,}
        if is_train and augment:
            self.transform = transforms.Compose([
                transforms.Resize((420, 420)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomResizedCrop(
                    size=((224, 224)),
                    scale=augmentations['scale'],
                    ratio=augmentations['ratio']
                ),

                transforms.ColorJitter(
                    brightness=augmentations['brightness'],
                    contrast=augmentations['contrast'],
                ),

                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
            ])

        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path, label, name = self.images[index]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image, label, name

    def __len__(self):
        return len(self.images)
        

if __name__ == '__main__':
    data = task_B_dataset(is_train=True, fold_num=5)
    name_list = []
    print(len(data))

    for image, label, name in data:
        name_list.append(name)

    # print(name_list)
