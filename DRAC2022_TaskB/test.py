import os
import argparse
import torch
import torch.nn as nn
import torchvision
from dataset import task_B_dataset
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--save_csv_dir', default='./csv/', type=str)
    parser.add_argument('--load_ckpt_dir', default='./ckpt/', type=str)
    parser.add_argument('--pkl_name', default='kfold_3.pkl', type=str)

    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if not os.path.exists(args.save_csv_dir):
        os.mkdir(args.save_csv_dir)
    

    if args.backbone =='resnet50':
            model = torchvision.models.resnet50(weights=True)
    else:
        assert "correct input model" == 'right'

    num_features = model.fc.in_features
    num_classes = 3 
    model.fc = nn.Linear(num_features, num_classes)


    ckpt = torch.load(args.load_ckpt_dir + args.pkl_name)
    state_dict = ckpt['state']
    kappa = ckpt['kappa']
    epoch = ckpt['epoch']

    if epoch == -1:
        print(f'load model from {args.load_ckpt_dir + args.pkl_name} with kappa = {kappa} to cross val')
    else:
        print(f'load model from {args.load_ckpt_dir + args.pkl_name} with kappa = {kappa} at epoch {epoch}')
    
    model.load_state_dict(state_dict)
    model = model.to(device=device)
    model.eval()

    test_data = task_B_dataset(is_test=True)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    csv_path = os.path.join(args.save_csv_dir, args.pkl_name[:-3] + 'csv')

    new_flag = True
    with torch.no_grad():
        print('testing...')

        for image, label, name in test_loader:
            image = image.to(device)
            logits = model(image).cpu()

            class_list = logits.max(1)[1]
            probs = nn.functional.softmax(logits, dim=1)
            dataframe = pd.DataFrame({'case':name,'class':class_list,'P0':probs[:,0],'P1':probs[:,1],'P2':probs[:,2]})
            if new_flag:
                dataframe.to_csv(csv_path, header=True, index=None)
                new_flag = False
            else:
                dataframe.to_csv(csv_path, mode='a', header=False, index=None)
        print(f'Saving results to {csv_path}')

