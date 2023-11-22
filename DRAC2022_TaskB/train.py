import argparse
import torch.nn as nn
import torch
import torchvision
import os
from dataset import task_B_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score,accuracy_score
from tqdm import tqdm 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epoch', default=40, type=int)
    parser.add_argument('--save_ckpt_dir', default='./ckpt/', type=str)

    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if not os.path.exists(args.save_ckpt_dir):
        os.mkdir(args.save_ckpt_dir)

    sum_kappa = 0
    states = []

    for k in range(5):
        print(f'fold_num: {k}')
        if args.backbone =='resnet50':
            model = torchvision.models.resnet50(weights=True)
        elif args.backbone =='inception_v3':
            model = torchvision.models.inception_v3(weights=True)
        else:
            assert "correct input model" == 'right'

        print(model)
        assert 1==2
        num_features = model.fc.in_features
        num_classes = 3 
        model.fc = nn.Linear(num_features, num_classes)
        model = model.to(device=device)

        train_dataset = task_B_dataset(is_train=True, fold_num=k)
        val_dataset = task_B_dataset(is_val=True, fold_num=k)
        train_loader = DataLoader(train_dataset,  shuffle=True, batch_size=args.batch_size, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4, pin_memory=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True)
        criterion = nn.CrossEntropyLoss()

        best_model = {'state':None, 'kappa': -100, 'epoch': 0,}

        for epoch in range(args.epoch):
            model.train()
            total_loss = 0
            pred_list = []
            label_list = []
            for image, label, name in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}", unit="batch"):
                image, label = image.to(device), label.to(device)
                logits = model(image)
                pred = torch.max(logits, 1)[1]
                loss = criterion(logits, label.to(torch.long))
                total_loss += loss.item()
                pred_list.extend(pred.detach().cpu())
                label_list.extend(label.cpu())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            kappa = cohen_kappa_score(label_list, pred_list, weights='quadratic')
            acc = accuracy_score(label_list, pred_list)

            print(f'Epoch: {epoch + 1}, Loss: {total_loss}, Acc: {acc}, Kappa: {kappa}')

            if (epoch + 1) % 2 == 0:
                with torch.no_grad():
                    model.eval()
                    pred_list = []
                    label_list = []

                    for image, label, name in tqdm(val_loader, desc=f"{1}/{1}", unit="batch"):
                        image = image.to(device=device)
                        logits = model(image)
                        pred_list.extend(logits.max(1)[1].cpu())
                        label_list.extend(label.cpu())
                    
                    kappa = cohen_kappa_score(label_list, pred_list, weights='quadratic')
                    acc = accuracy_score(label_list, pred_list)
                    print(f'Val Epoch: {epoch + 1}, Acc: {acc}, Kappa: {kappa}')

                    if kappa > best_model['kappa']:
                        best_model['epoch'] = epoch
                        best_model['kappa'] = kappa
                        best_model['state'] = model.state_dict()

        states.append(best_model['state'])
        sum_kappa += best_model["kappa"]

        savePath = os.path.join(args.save_ckpt_dir, f'kfold_{k}.pkl')
        print(f'Saving model(epoch={best_model["epoch"]},kappa={best_model["kappa"]}) to {savePath}...')
        torch.save(best_model, savePath)

    print(f'Average kappa is {sum_kappa / 5}')
            


                    
