import argparse
import logging
import yaml
import sys
sys.path.append('../')

from utils import *
from model import *

def generate_config(args):
    with open('./config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value and key != 'log':
            config['train'][key] = value
    logger.info('train config {}'.format(config))

    return config

if __name__ == '__main__':
    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--use_gpu_num', type=int)
    parser.add_argument('--train_steps', type=int)
    parser.add_argument('--log', type=str, default='train.log')
    args = parser.parse_args()
    # log
    set_logger('./train_log/{}'.format(args.log))
    # config
    config = generate_config(args)
    # train
    model = Model(config)
    model.train()
    model.generate_submit()