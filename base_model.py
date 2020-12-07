import torch
import logging

from data_process import *

logger = logging.getLogger()

class BaseModel:
    def __init__(self, config):
        self.config = config
        self.train_X, self.train_y, self.val_X, self.val_y, self.test_X, self.vocab = \
            data_process(config)
        self.train_iter, self.val_iter = generate_train_val_iter(\
            self.train_X, self.train_y, self.val_X, self.val_y, config)

    def train(self):
        pass

    def generate_submit(self):
        pass