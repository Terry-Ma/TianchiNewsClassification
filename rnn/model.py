import torch
import logging
import sys
sys.path.append('../')

from torch import nn
from torch.nn.functional import softmax
from sklearn.metrics import f1_score
from base_model import BaseModel
from utils import *

logger = logging.getLogger()

class Model(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = BiGRU(config)

    def train(self):
        pass

    def generate_submit(self):
        pass

class BiGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config['model']['vocab_size'], config['model']['embed_size'])
        self.bigru = nn.GRU(config['model']['embed_size'], config['model']['hidden_num'], \
            config['model']['layer_num'], bidirectional=True)
        self.linear = nn.Linear(2 * config['model']['vocab_size'], config['model']['type_num'])
    
    def forward(self, X, state_begin=None):
        embed_X = self.embedding(X).permute(1, 0, 2)
        output, hidden_states = self.bigru(embed_X, state_begin)
        linear_input = torch.cat((output[-1, :, :self.hidden_num], output[0, :, self.hidden_num:]), dim=1)
        output = self.linear(linear_input)

        return output