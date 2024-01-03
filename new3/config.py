import os
import sys
import logging
import argparse
from datetime import datetime

def get_train_config():
    parser = argparse.ArgumentParser()

    ## 数据集参数
    parser.add_argument('--train_image_dir', default='data/train/')
    parser.add_argument('--train_label_path', default='data/labels.csv')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)  

    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--model', type=str, default='resnet') 
    parser.add_argument('--model_path', default='./saved_models')

    args = parser.parse_args()
    
    return args

def get_test_config():
    parser = argparse.ArgumentParser()

    ## 数据集参数
    parser.add_argument('--test_image_dir', default='data/test/')

    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--model', type=str, default='resnet') 

    args = parser.parse_args()
    
    return args
