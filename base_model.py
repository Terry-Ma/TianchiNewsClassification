import torch
import logging

from data_process import *

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

    def train(self):
        pass

    def generate_submit(self):
        pass