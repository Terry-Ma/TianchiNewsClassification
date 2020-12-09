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
    parser.add_argument('--cuda_visible_devices', type=str)
    parser.add_argument('--multi_gpu', type=int)
    parser.add_argument('--load_checkpoint', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--train_steps', type=int)
    parser.add_argument('--log_file', type=str, default='train.log')
    parser.add_argument('--checkpoint_dir', type=str)
    parser.add_argument('--submit_file', type=str)
    parser.add_argument('--is_demo', type=int)
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
