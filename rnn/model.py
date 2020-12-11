import torch
import logging
import sys
import numpy as np
import os
import pandas as pd
sys.path.append('../')

from torch import nn
from torch.nn.functional import softmax
from sklearn.metrics import f1_score, classification_report
from base_model import BaseModel
from torch.nn.functional import softmax
from utils import *

logger = logging.getLogger()

class Model(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.init(BiRNN)

    def train(self):
        logger.info('start training')
        cur_epochs = 1
        cur_train_steps = 1
        max_val_f1 = 0
        best_step = 1
        while cur_train_steps <= self.config['train']['train_steps']:
            check_train_steps = 0
            check_train_loss = 0
            check_train_y = np.array([])
            check_train_pred_y = np.array([])
            # shuffle every epoch
            for batch_X, batch_y in self.train_iter:
                check_train_y = np.concatenate((check_train_y, np.array(batch_y)))
                # train
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_pred_y = self.model(batch_X)
                train_loss = self.loss(batch_pred_y, batch_y)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                # check train steps & loss & pred_y
                check_train_steps += 1
                check_train_loss += train_loss.item()
                batch_pred_y = batch_pred_y.argmax(dim=1).to('cpu')
                check_train_pred_y = np.concatenate((check_train_pred_y, np.array(batch_pred_y)))
                # check & val step
                if cur_train_steps > 0 and cur_train_steps % self.config['train']['steps_per_check'] == 0:
                    self.model.eval()    # dropout...
                    check_val_loss = 0
                    check_val_steps = 0
                    check_val_y = np.array([])
                    check_val_pred_y = np.array([])
                    # train f1
                    check_train_f1 = f1_score(check_train_y, check_train_pred_y, average='macro')
                    with torch.no_grad():
                        for batch_X, batch_y in self.val_iter:
                            check_val_y = np.concatenate((check_val_y, np.array(batch_y)))
                            # val
                            batch_X = batch_X.to(self.device)
                            batch_y = batch_y.to(self.device)
                            batch_pred_y = self.model(batch_X)
                            # check val steps & loss & pred_y
                            check_val_loss += self.loss(batch_pred_y, batch_y).item()
                            check_val_steps += 1
                            batch_pred_y = batch_pred_y.argmax(dim=1).to('cpu')
                            check_val_pred_y = np.concatenate((check_val_pred_y, np.array(batch_pred_y)))
                    # val f1
                    check_val_f1 = f1_score(check_val_y, check_val_pred_y, average='macro')
                    # max f1 model
                    if check_val_f1 > max_val_f1:
                        max_val_f1 = check_val_f1
                        best_step = cur_train_steps
                        torch.save(self.model.state_dict(), './checkpoint/{}/best_model.pkl'.format(
                            self.config['train']['checkpoint_dir']))
                    # log
                    logger.info('epoch {0}, steps {1}, train_loss {2:.4f}, val_loss {3:.4f}, train_f1 {4:.4f}, val_f1 {5:.4f}, max_val_f1 {6:.4f}, best_step {7}'.format(
                        cur_epochs,
                        cur_train_steps,
                        check_train_loss / check_train_steps,
                        check_val_loss / check_val_steps,
                        check_train_f1,
                        check_val_f1,
                        max_val_f1,
                        best_step
                        ))
                    check_train_steps = 0
                    check_train_loss = 0
                    check_train_y = np.array([])
                    check_train_pred_y = np.array([])
                    self.model.train()    # dropout...
                # checkpoint
                if cur_train_steps > 0 and cur_train_steps % self.config['train']['steps_per_checkpoint'] == 0:
                    checkpoint_path = './checkpoint/{}/checkpoint_steps_{}.pkl'.\
                        format(self.config['train']['checkpoint_dir'], cur_train_steps)
                    torch.save(self.model.state_dict(), checkpoint_path)
                    logger.info('save checkpoints {}'.format(checkpoint_path))
                cur_train_steps += 1
                if cur_train_steps > self.config['train']['train_steps']:
                    break   
            cur_epochs += 1 
        logger.info('training complete, training epochs {0}, steps {1}, max_val_f1 {2:.4f}, best_step {3}'.\
            format(cur_epochs, self.config['train']['train_steps'], max_val_f1, best_step))
        # val analyse
        self.val_analyse()

    def generate_submit(self):
        # predict
        test_pred_y = np.array([])
        with torch.no_grad():
            self.model.eval()    # dropout...
            for batch_X, _ in self.test_iter:
                batch_X = batch_X.to(self.device)
                batch_pred_y = self.model(batch_X)
                batch_pred_y = batch_pred_y.argmax(dim=1).to('cpu')
                test_pred_y = np.concatenate((test_pred_y, batch_pred_y))
            self.model.train()    # dropout...
        # DataFrame
        submit = pd.DataFrame(columns=['label'])
        submit['label'] = test_pred_y
        submit['label'] = submit['label'].astype('int')
        # to csv
        submit_path = '../submit/{}'.format(self.config['eval']['submit_file'])
        submit.to_csv(submit_path, index=False)
        logger.info('generate submit {}'.format(submit_path))
    
    def val_analyse(self):
        # predict
        val_y = np.array([])
        val_pred_y = np.array([])
        with torch.no_grad():
            self.model.eval()    # dropout...
            for batch_X, batch_y in self.val_iter:
                batch_X = batch_X.to(self.device)
                batch_pred_y = self.model(batch_X)
                batch_pred_y = batch_pred_y.argmax(dim=1).to('cpu')
                val_pred_y = np.concatenate((val_pred_y, batch_pred_y))
                val_y = np.concatenate((val_y, batch_y.to('cpu')))
            self.model.train()    # dropout...
        # eval
        logger.info('model val analyse \n{}'.format(
            classification_report(val_y.astype(np.int32), val_pred_y.astype(np.int32))))

class BiRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config['model']['vocab_size'], config['model']['embed_size'])
        if self.config['model']['model_type'] == 'GRU':
            self.birnn = nn.GRU(
                input_size=config['model']['embed_size'],
                hidden_size=config['model']['hidden_num'],
                num_layers=config['model']['layer_num'],
                dropout=self.config['model']['dropout'],
                bidirectional=True
                )
        else:
            self.birnn = nn.LSTM(
                input_size=config['model']['embed_size'],
                hidden_size=config['model']['hidden_num'],
                num_layers=config['model']['layer_num'],
                dropout=self.config['model']['dropout'],
                bidirectional=True
                )
        if self.config['model']['use_attention']:
            self.attention = Attention(2 * self.config['model']['hidden_num'], self.config['model']['attention_size'])
        self.linear = nn.Linear(2 * config['model']['hidden_num'], config['model']['type_num'])
    
    def forward(self, X):
        embed_X = self.embedding(X).permute(1, 0, 2)
        output, _ = self.birnn(embed_X)
        if self.config['model']['use_attention']:
            linear_input = self.attention(output.permute(1, 0, 2))
        else:
            linear_input = torch.cat((output[-1, :, :self.config['model']['hidden_num']], \
                output[0, :, self.config['model']['hidden_num']:]), dim=1)
        output = self.linear(linear_input)

        return output

class Attention(nn.Module):
    def __init__(self, input_size, attention_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1)
        )
    
    def forward(self, X):
        '''
        Args:
            X: (batch_size, seq_len, input_size)
        Returns:
            output: (batch_size, input_size)
        '''
        weight = self.attention(X)
        weight = softmax(weight, dim=1)
        output = (X * weight).sum(dim=1).squeeze()

        return output