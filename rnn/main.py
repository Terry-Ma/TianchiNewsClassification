import argparse
import logging
import sys
sys.path.append('../')

from utils import *

if __name__ == '__main__':
    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--use_gpu_num', type=int)
    parser.add_argument('--log_path', type=str, default='./train_log/model.log')
    args = parser.parse_args()
    # log
    set_logger(args.log_path)
    