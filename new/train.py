import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import cohen_kappa_score,accuracy_score
from tqdm import tqdm
from model import SBConvNext, SBResNet
from sklearn import metrics
import copy
from config import *
from dataset import DataSet
import torch.nn as nn


def train_and_evaluate_fold(train_loader, val_loader, model, criterion, optimizer, device, epoch):
    model.train()  # 训练模式
    train_loss = 0
    train_pred_list = []
    train_label_list = []

    # 训练过程中的进度条
    pbar_train = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} [train]", leave=True)
    for batch_idx, (inputs, targets) in pbar_train:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_pred_list.extend(predicted.cpu().numpy())
        train_label_list.extend(targets.cpu().numpy())

        # 更新进度条
        pbar_train.set_postfix({"Train Loss": f"{train_loss/(batch_idx+1):.4f}"})

    train_accuracy = accuracy_score(train_label_list, train_pred_list)
    train_kappa = cohen_kappa_score(train_label_list, train_pred_list, weights='quadratic')

    # 验证过程
    model.eval()  # 评估模式
    val_loss = 0
    val_pred_list = []
    val_label_list = []

    pbar_val = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch} [val]", leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in pbar_val:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_pred_list.extend(predicted.cpu().numpy())
            val_label_list.extend(targets.cpu().numpy())

            # 更新进度条
            pbar_val.set_postfix({"Val Loss": f"{val_loss/(batch_idx+1):.4f}"})

    val_accuracy = accuracy_score(val_label_list, val_pred_list)
    val_kappa = cohen_kappa_score(val_label_list, val_pred_list, weights='quadratic')

    print(f'Epoch {epoch}: Train Loss: {train_loss:.3f}, Train Acc: {train_accuracy:.2f}, Train Kappa: {train_kappa:.2f}, Val Loss: {val_loss:.3f}, Val Acc: {val_accuracy:.2f}, Val Kappa: {val_kappa:.2f}')

    return train_accuracy, train_kappa, val_accuracy, val_kappa



def kfold_cross_validation(device, model_origin, image_paths, labels, args, num_folds=5):
    # 将image_paths和labels同时重新排列，以确保对应关系
    image_paths, labels = shuffle(image_paths, labels, random_state=42)
    # 创建KFold对象并进行拆分
    kfold = KFold(n_splits=num_folds, shuffle=False)
    # 创建存储每个折的模型信息的列表
    fold_models_info = []

    for fold, (train_indices, val_indices) in enumerate(kfold.split(image_paths)):
        train_image_paths = [image_paths[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        val_image_paths = [image_paths[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]

        # Create train & val set and DataLoader
        train_dataset = DataSet(train_image_paths, train_labels, augment=True)
        val_dataset = DataSet(val_image_paths, val_labels, augment=False)
        train_loader = DataLoader(train_dataset, 
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    drop_last=True,
                                    pin_memory=True)
        val_loader = DataLoader(val_dataset, 
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    drop_last=True,
                                    pin_memory=True)
        # train_size = train_loader.__len__()*args.batch_size
        # val_size = val_loader.__len__()*args.batch_size

        ########## Train model from scarth ##########
        model = copy.deepcopy(model_origin)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        model = model.to(device=device)

        # Initialization
        best_model_info = {'epoch': 0, 'train_accuracy': 0, 'val_accuracy': 0, 'kappa': -1, 'state_dict': None}

        # Train each model num_epochs epochs
        for epoch in range(0, args.num_epochs):
            train_accuracy, train_kappa, val_accuracy, val_kappa = train_and_evaluate_fold(train_loader, val_loader, model, criterion, optimizer, device, epoch)

            # Check if get better model
            if val_kappa > best_model_info['kappa']:
                best_model_info = {
                    'epoch': epoch,
                    'train_accuracy': train_accuracy,
                    'val_accuracy': val_accuracy,
                    'kappa': val_kappa,
                    'state_dict': copy.deepcopy(model.state_dict())
                }
        
        # Save the model
        best_model_path = os.path.join(args.model_path, f'fold_{fold+1}_best_{args.model}_model.pth')
        torch.save(best_model_info['state_dict'], best_model_path)

        fold_models_info.append(best_model_info)
        print(f"Fold {fold+1} completed. Best Epoch: {best_model_info['epoch']}, Train Acc: {best_model_info['train_accuracy']:.2f}, Val Acc: {best_model_info['val_accuracy']:.2f}, Kappa: {best_model_info['kappa']:.4f}")

    # 打印所有fold的最佳模型信息
    for i, info in enumerate(fold_models_info, 1):
        print(f"Fold {i} - Best Epoch: {info['epoch']}, Train Acc: {info['train_accuracy']:.2f}, Val Acc: {info['val_accuracy']:.2f}, Kappa: {info['kappa']:.4f}")
        ########## End Training one fold ##########


# def train_and_evaluate_fold(train_loader, val_loader, model):
    # 编写一个函数，用于在单个折叠上训练和评估模型
    # 这个函数将接受训练数据加载器、验证数据加载器和模型作为参数，并在该折叠上进行训练和评估


if __name__ == '__main__':
    ########## Data preprocessing ##########
    train_image_dir = '../DRAC2022_TaskB/data/1. Original Images/a. Training Set/'
    train_label_path = '../DRAC2022_TaskB/data/2. Groundtruths/a. DRAC2022_ Image Quality Assessment_Training Labels.csv'
    test_image_dir = '../DRAC2022_TaskB/data/1. Original Images/b. Testing Set/'

    # train set
    train_image_paths = []
    train_labels = []

    label_file = pd.read_csv(train_label_path)                                                                                                                                                
    for _,row in label_file.iterrows(): 
        name = row['image name']
        label = int(row['image quality level'])
        train_image_path = train_image_dir + name
        train_image_paths.append(train_image_path)
        train_labels.append(label)
    
    """
    # test set
    test_image_path_list = os.listdir(test_image_dir)
    test_image_path_list.sort(key=lambda x:int(x.split('.')[0]))
    test_image_paths = []
    for name in test_image_path_list:
        test_images_paths.append(test_image_dir + name)
    """
    
    ########## End Date Preprocessing #########

    args = get_config()

    """
    if args.device is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    """
    args.device = 'cpu' if args.device < 0 else 'cuda:%i' % args.device
    args.device = torch.device(args.device)
    
    model_dict = {
        'resnet': SBResNet,
        'convnext': SBConvNext
    }

    model = model_dict[args.model]().to(args.device)

    ########## 
    if args.k_fold:
        kfold_cross_validation(device=args.device, 
                                model_origin=model, 
                                image_paths=train_image_paths, 
                                labels=train_labels, 
                                args=args, 
                                num_folds=5)                    