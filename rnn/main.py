import argparse
import logging
import yaml
import sys
sys.path.append('../')

from utils import *
from model import *

logger = logging.getLogger()

if __name__ == '__main__':
    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config.yml')
    parser.add_argument('--log_file', type=str, default='train.log')
    with open('./config.yml') as f:
        config_format = yaml.load(f, Loader=yaml.FullLoader)
    for _, param_kv in config_format.items():
        for param_k, param_v in param_kv.items():
            parser.add_argument('--{}'.format(param_k), type=type(param_v))
    args = parser.parse_args()
    # log
    set_logger('./train_log/{}'.format(args.log_file))
    # config
    config = generate_config(args)
    # checkpoint
    checkpoint_process('./checkpoint/{}'.format(args.checkpoint_dir))
    # train
    model = Model(config)
    model.train()
    # # val analyse
    # model.val_analyse()
    # # submit
    # model.generate_submit()