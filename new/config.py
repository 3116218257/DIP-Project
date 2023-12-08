import os
import sys
import logging
import argparse
from datetime import datetime

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)  
    # 接受多个GPU设备ID，例如：--device 0 1 2 3
    # parser.add_argument('--device', nargs='+', type=int, default=[1]) 
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--model', type=str, default='convnext') 
    parser.add_argument('--model_path', default='./saved_models')
    parser.add_argument('--k_fold',action='store_true')

    args = parser.parse_args()
    
    return args
