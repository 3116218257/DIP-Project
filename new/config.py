import os
import sys
import logging
import argparse
from datetime import datetime

def get_config():
    parser = argparse.ArgumentParser()

    ## 数据集参数
    parser.add_argument('--train_image_dir', default='data_sup_2/train/')
    parser.add_argument('--test_image_dir', default='data_sup_2/test/')
    parser.add_argument('--train_label_path', default='data_sup_2/labels.csv')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-2)  

    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--model', type=str, default='resnet') 
    parser.add_argument('--model_path', default='./saved_models_resnet101_bs64_base_aug_sup_3')
    parser.add_argument('--k_fold',action='store_true')

    args = parser.parse_args()
    
    return args

def get_test_config():
    parser = argparse.ArgumentParser()

    ## 数据集参数
    parser.add_argument('--test_image_dir', nargs='+', type=str)

    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--model_path', nargs='+', type=str)
    parser.add_argument('--model', nargs='+', type=str) 

    args = parser.parse_args()
    
    return args
