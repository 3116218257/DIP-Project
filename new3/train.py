import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score,accuracy_score
from tqdm import tqdm
from model import SBConvNext, SBResNet
import copy
from config import *
from dataset import DataSet, StageDataSet, ValDataSet
import torch.nn as nn
from random import sample

def train_and_evaluate(train_loader, val_loader, model, criterion, optimizer, device, epoch_num, stage):
    best_model_info = {'epoch': 0, 'train_accuracy': 0, 'val_accuracy': 0, 'kappa': -1, 'state_dict': None}

    for epoch in range(epoch_num):
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
            if (not os.path.exists(os.path.join(args.model_path, str(stage)))):
                os.makedirs(os.path.join(args.model_path, str(stage)))
            best_model_path = os.path.join(args.model_path, str(stage), f'{args.model}_epoch{epoch}.pth')
            torch.save(best_model_info['state_dict'], best_model_path)

    model_path = os.path.join(args.model_path, str(stage), f'{args.model}_epoch{epoch}.pth')
    torch.save(copy.deepcopy(model.state_dict()), model_path)

    return best_model_path


def evaluate_both_stages(best_model_stage0_path, best_model_stage1_path, val_dataset, model_stage0, model_stage1, device):
    model_stage0.load_state_dict(torch.load(best_model_stage0_path, map_location=device))
    model_stage0 = model_stage0.to(device)
    model_stage1.load_state_dict(torch.load(best_model_stage1_path, map_location=device))
    model_stage1 = model_stage1.to(device)

    val_dataset_stage0 = ValDataSet("data/train", val_dataset)
    val_loader_stage0 = DataLoader(val_dataset_stage0, 
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    model_stage0.eval()
    model_stage1.eval()
    predictions_stage0 = []
    probabilities_stage0 = []
    predictions_stage1 = []
    probabilities_stage1 = []
    labels_list = []

    pbar_val = tqdm(enumerate(val_loader_stage0), total=len(val_loader_stage0), desc="[val]", leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in pbar_val:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_stage1(inputs)

            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            predictions_stage0.extend(predicted.cpu().numpy())
            probabilities_stage0.extend(probs.cpu().numpy())
            labels_list.extend(targets.cpu().numpy())

    class0_index = []
    class1_index = []
    for image_num in range(len(predictions_stage0)):
        if (predictions_stage0[image_num] == 0):
            class0_index.append(image_num)
        else:
            class1_index.append(image_num)

    # print(predictions_stage0, labels_list)

    # 先分01和2
    # val_dataset_stage1 = ValDataSet("data/train", val_dataset, class0_index)
    # 先分0和12
    val_dataset_stage1 = ValDataSet("data/train", val_dataset, class1_index)
    val_loader_stage1 = DataLoader(val_dataset_stage1, 
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    pbar_val = tqdm(enumerate(val_loader_stage1), total=len(val_loader_stage1), desc="[val]", leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in pbar_val:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_stage1(inputs)

            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            predictions_stage1.extend(predicted.cpu().numpy())
            probabilities_stage1.extend(probs.cpu().numpy())

    predictions = []
    probabilities = []

    stage1_image_num = 0
    for image_num in range(len(predictions_stage0)):
        # 先分01和2
        # if predictions_stage0[image_num] == 0:
        #     if predictions_stage1[stage1_image_num] == 0:
        #         predictions.append(0)
        #         prob0 = probabilities_stage0[image_num][0] * probabilities_stage1[stage1_image_num][0]
        #         prob1 = probabilities_stage0[image_num][0] * probabilities_stage1[stage1_image_num][1]
        #         prob2 = probabilities_stage0[image_num][1]
        #         probabilities.append([prob0, prob1, prob2])
        #     else:
        #         predictions.append(1)
        #         prob0 = probabilities_stage0[image_num][0] * probabilities_stage1[stage1_image_num][0]
        #         prob1 = probabilities_stage0[image_num][0] * probabilities_stage1[stage1_image_num][1]
        #         prob2 = probabilities_stage0[image_num][1]
        #         probabilities.append([prob0, prob1, prob2])
        #     stage1_image_num += 1
        # else:
        #     predictions.append(2)
        #     prob0 = probabilities_stage0[image_num][0] / 2
        #     prob1 = probabilities_stage0[image_num][0] / 2
        #     prob2 = probabilities_stage0[image_num][1]
        #     probabilities.append([prob0, prob1, prob2])

        # 先分0和12
        if predictions_stage0[image_num] == 0:
            predictions.append(0)
            prob0 = probabilities_stage0[image_num][0]
            prob1 = probabilities_stage0[image_num][1] / 2
            prob2 = probabilities_stage0[image_num][1] / 2
            probabilities.append([prob0, prob1, prob2])
        else:
            if predictions_stage1[stage1_image_num] == 0:
                predictions.append(1)
                prob0 = probabilities_stage0[image_num][0] 
                prob1 = probabilities_stage0[image_num][1] * probabilities_stage1[stage1_image_num][0]
                prob2 = probabilities_stage0[image_num][1] * probabilities_stage1[stage1_image_num][1]
                probabilities.append([prob0, prob1, prob2])
            else:
                predictions.append(2)
                prob0 = probabilities_stage0[image_num][0] 
                prob1 = probabilities_stage0[image_num][1] * probabilities_stage1[stage1_image_num][0]
                prob2 = probabilities_stage0[image_num][1] * probabilities_stage1[stage1_image_num][1]
                probabilities.append([prob0, prob1, prob2])
            stage1_image_num += 1
            

    val_accuracy = accuracy_score(labels_list, predictions)
    val_kappa = cohen_kappa_score(labels_list, predictions, weights='quadratic')

    return val_accuracy, val_kappa
        

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
    
    args.device = 'cpu' if args.device < 0 else 'cuda:%i' % args.device
    device = torch.device(args.device)

    image_dir = "data/train"
    label_dir = "data/labels.csv"
    
    train_size = int(0.8 * 665)
    val_size = 665 - train_size
    all_indexes = [i for i in range(665)]
    train_indexes = sample(all_indexes, train_size)
    # train_indexes = [i for i in range(train_size)]
    val_indexes = [i for i in all_indexes if not i in train_indexes]
    train_dataset = DataSet(image_dir, label_dir, train_indexes)
    val_dataset = DataSet(image_dir, label_dir, val_indexes)

    train_dataset_stage0 = StageDataSet("data/train", train_dataset, 0)
    val_dataset_stage0 = StageDataSet("data/train", val_dataset, 0)

    train_loader_stage0 = DataLoader(train_dataset_stage0, 
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=True,
                            pin_memory=True)
    
    val_loader_stage0 = DataLoader(val_dataset_stage0, 
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=True,
                            pin_memory=True)

    model_dict = {
        'resnet': SBResNet,
        'convnext': SBConvNext
    }

    model_stage0 = model_dict[args.model]().to(args.device)

    # weights = torch.tensor([10.36, 5.34, 1]).to(device)
    # criterion = nn.CrossEntropyLoss(weights)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model_stage0.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_model_stage0_path = train_and_evaluate(train_loader_stage0, val_loader_stage0, model_stage0, criterion, optimizer, device, args.num_epochs, 0)       
    

    train_dataset_stage1 = StageDataSet("data/train", train_dataset, 1)
    val_dataset_stage1 = StageDataSet("data/train", val_dataset, 1)

    train_loader_stage1 = DataLoader(train_dataset_stage1, 
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=True,
                            pin_memory=True)
    
    val_loader_stage1 = DataLoader(val_dataset_stage1, 
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=True,
                            pin_memory=True)

    model_stage1 = model_dict[args.model]().to(args.device)
    optimizer = torch.optim.AdamW(model_stage1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_model_stage1_path = train_and_evaluate(train_loader_stage1, val_loader_stage1, model_stage1, criterion, optimizer, device, args.num_epochs, 1)          
    
    model_stage0_eval = model_dict[args.model]()
    model_stage1_eval = model_dict[args.model]()
    val_accuracy, val_kappa = evaluate_both_stages(best_model_stage0_path, best_model_stage1_path, val_dataset, model_stage0_eval, model_stage1_eval, device)
    print("val_accuracy: ", val_accuracy, "val_kappa: ", val_kappa)