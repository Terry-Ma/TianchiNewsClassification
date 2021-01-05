import logging
import yaml
import os
import torch

from gensim.models import KeyedVectors
from torch import nn

logger = logging.getLogger()

def set_logger(log_path):
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)

def checkpoint_process(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
        logger.info('checkpoint dir not exist, make dir {}'.format(checkpoint_dir))
    else:
        logger.info('checkpoint dir exist: {}'.format(checkpoint_dir))

def generate_config(args):
    with open('./config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    logger.info('load config from {}'.format('./config.yml'))
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None:
            find_k = False
            for _, config_v in config.items():
                if key in config_v:
                    config_v[key] = value
                    find_k = True
                    break
            if not find_k:
                config['train'][key] = value
    config['train']['checkpoint_path'] = './checkpoint/{}/'.format(args.experiment_name)
    config['train']['submit_path'] = '../submit/{}.csv'.format(args.experiment_name)
    config['train']['tb_path'] = './train_log/tensorboard/{}/'.format(args.experiment_name)
    logger.info('will use config \n{}'.format(config))

    return config

def load_wv_as_embedding(config, embedding_size, vocab):
    logger.info('will load pretrain-embedding')
    if os.path.exists(config['model']['load_wv_path']):
        # load local embedding
        pretrain_weight = torch.FloatTensor(embedding_size)
        torch.nn.init.normal_(pretrain_weight)    # N(0, 1)
        wv = KeyedVectors.load(config['model']['load_wv_path'], mmap='r')
        for embed_i in range(embedding_size[0]):
            if vocab.itos[embed_i] in wv:
                pretrain_weight[embed_i, :] = torch.tensor(wv[vocab.itos[embed_i]])
    else:
        logger.error('word2vec path {} not exists'.format(config['model']['load_wv_path']))
        raise Exception('word2vec path {} not exists'.format(config['model']['load_wv_path']))
    
    return pretrain_weight

if __name__ == '__main__':
    set_logger('./rnn/train_log/train.log')
    logger.info('set logger sucess')