import torch
import logging
import os

from data_process import *
from gensim.models import KeyedVectors
from torch import nn

logger = logging.getLogger()

class BaseModel:
    def __init__(self, config):
        self.config = config
        self.train_X, self.train_y, self.val_X, self.val_y, self.test_X, self.vocab = \
            generate_tensor(config)
        self.train_iter, self.val_iter, self.test_iter = generate_train_val_test_iter(\
            self.train_X, self.train_y, self.val_X, self.val_y, self.test_X, config)
        self.config['model']['vocab_size'] = len(self.vocab)
        self.config['model']['type_num'] = self.train_y.unique().shape[0]

    def init(self, NNModule):
        # gpu device
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config['train']['cuda_visible_devices']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info('use device {}'.format(self.device))
        # init model
        self.model = NNModule(self.config).to(self.device)
        logger.info('init model \n{}'.format(self.model))
        # load pretrain_word2vec
        if self.config['model']['load_wv_path'] != '':
            self.load_wv_as_embedding()
        # load checkpoint
        if self.config['model']['load_checkpoint'] != '':
            self.load_checkpoint()
        # multi-gpu
        if self.config['train']['multi_gpu'] and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info('will train on {} gpus'.format(torch.cuda.device_count()))
        # init loss - mean
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        # init optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['train']['lr'])

    def load_wv_as_embedding(self):
        logger.info('will load pretrain-embedding')
        if os.path.exists(self.config['model']['load_wv_path']):
            # load local embedding
            pretrain_weight = torch.FloatTensor(self.model.embedding.weight.size())
            torch.nn.init.normal_(pretrain_weight)    # N(0, 1)
            wv = KeyedVectors.load(self.config['model']['load_wv_path'], mmap='r')
            for embed_i in range(self.model.embedding.weight.shape[0]):
                if self.vocab.itos[embed_i] in wv:
                    pretrain_weight[embed_i, :] = torch.tensor(wv[self.vocab.itos[embed_i]])
            self.model.embedding = nn.Embedding.from_pretrained(pretrain_weight).to(self.device)
        else:
            logger.error('word2vec path {} not exists'.format(self.config['model']['load_wv_path']))
            raise Exception('word2vec path {} not exists'.format(self.config['model']['load_wv_path']))
    
    def load_checkpoint(self):
        logger.info('will load pretrain-model')
        checkpoint_path = './checkpoint/{}'.format(self.config['model']['load_checkpoint'])
        if os.path.exists(checkpoint_path):
            # load checkpoint
            self.model.load_state_dict(torch.load(checkpoint_path))
            logger.info('load model from {}'.format(checkpoint_path))
        else:
            logger.error('checkpoint path {} not exists'.format(checkpoint_path))
            raise Exception('checkpoint path {} not exists'.format(checkpoint_path))

    def train(self):
        pass

    def generate_submit(self):
        pass