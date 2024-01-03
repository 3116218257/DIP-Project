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
from loss import WeightedKappaLoss
from sklearn.model_selection import KFold

def train_and_evaluate(train_loader, val_loader, model, criterion, optimizer, device, epoch_num, fold):
    best_model_info = {'epoch': 0, 'train_accuracy': 0, 'val_accuracy': 0, 'kappa': -1, 'state_dict': None}

    for epoch in range(0, epoch_num):
        train_accuracy, train_kappa = train_one_epoch(train_loader, model, criterion, optimizer, device, epoch)
        val_accuracy, val_kappa = evaluate(val_loader, model, criterion, device)

        print("Epoch: ", epoch, ", train_accuracy: ", train_accuracy, ", train_kappa: ", train_kappa, ", val_accuracy: ", val_accuracy, ", val kappa: ", val_kappa)

        if val_kappa > best_model_info['kappa']:
            best_model_info = {
                'epoch': epoch,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'kappa': val_kappa,
                'state_dict': copy.deepcopy(model.state_dict())
            }
            if (not os.path.exists(os.path.join(args.model_path, str(fold)))):
                os.makedirs(os.path.join(args.model_path, str(fold)))

            best_model_path = os.path.join(args.model_path, str(fold), f'{args.model}_epoch{epoch}.pth')
            torch.save(best_model_info['state_dict'], best_model_path)

    final_model_path = os.path.join(args.model_path, str(fold), f'{args.model}_epoch{epoch}.pth')
    torch.save(copy.deepcopy(model.state_dict()), final_model_path)


def evaluate(val_loader, model, criterion, device):
    # 验证过程
    model.eval()  # 评估模式
    val_loss = 0
    val_pred_list = []
    val_label_list = []

    pbar_val = tqdm(enumerate(val_loader), total=len(val_loader), desc="[val]", leave=True)
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

    return val_accuracy, val_kappa


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
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
    return train_accuracy, train_kappa


if __name__ == '__main__':
    args = get_train_config()

    if (not os.path.exists(args.model_path)):
        os.makedirs(args.model_path)
    
    args.device = 'cpu' if args.device < 0 else 'cuda:%i' % args.device
    device = torch.device(args.device)

    image_dir = "data/train"
    label_dir = "data/labels.csv"
    
    full_dataset = DataSet(image_dir, label_dir)
    # train_size = int(0.8 * len(full_dataset))
    # val_size = len(full_dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # train_loader = DataLoader(train_dataset, 
    #                         batch_size=args.batch_size,
    #                         shuffle=True,
    #                         drop_last=True,
    #                         pin_memory=True)
    
    # val_loader = DataLoader(val_dataset, 
    #                         batch_size=args.batch_size,
    #                         shuffle=True,
    #                         drop_last=True,
    #                         pin_memory=True)

    model_dict = {
        'resnet': SBResNet,
        'convnext': SBConvNext
    }

    model = model_dict[args.model]().to(args.device)

    # weights = torch.tensor([10.36, 5.34, 1]).to(device)
    # criterion = nn.CrossEntropyLoss(weights)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    kfold = KFold(n_splits=args.k_folds, shuffle=True)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(full_dataset)):
    
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(
                        full_dataset, 
                        batch_size=args.batch_size, sampler=train_subsampler)
        val_loader = torch.utils.data.DataLoader(
                        full_dataset,
                        batch_size=args.batch_size, sampler=test_subsampler)

        train_and_evaluate(train_loader, val_loader, model, criterion, optimizer, device, args.num_epochs, fold)                  