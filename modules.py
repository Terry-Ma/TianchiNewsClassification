import torch
import logging

from torch import nn
from torch.nn.functional import softmax
from utils import *
from transformers import (
    BertModel,
    BertForSequenceClassification,
    AdamW,
    BertConfig,
    get_linear_schedule_with_warmup
    )

class BiRNN(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        if self.config['model']['load_wv_path'] != '':
            pretrain_weight = load_wv_as_embedding(
                config,
                torch.Size([self.config['model']['vocab_size'], self.config['model']['embed_size']]),
                vocab
                )
            self.embedding = nn.Embedding.from_pretrained(pretrain_weight)
        else:
            self.embedding = nn.Embedding(
                self.config['model']['vocab_size'],
                self.config['model']['embed_size']
                )
        if self.config['model']['model_type'] == 'GRU':
            self.birnn = nn.GRU(
                input_size=self.config['model']['embed_size'],
                hidden_size=self.config['model']['hidden_num'],
                num_layers=self.config['model']['layer_num'],
                dropout=self.config['model']['dropout'],
                bidirectional=True
                )
        else:
            self.birnn = nn.LSTM(
                input_size=self.config['model']['embed_size'],
                hidden_size=self.config['model']['hidden_num'],
                num_layers=self.config['model']['layer_num'],
                dropout=self.config['model']['dropout'],
                bidirectional=True
                )
        if self.config['model']['agg_function'] == 'attention':
            self.attention = Attention(2 * self.config['model']['hidden_num'], self.config['model']['attention_size'])
        elif self.config['model']['agg_function'] == 'max':
            self.max_pool = torch.nn.MaxPool1d(self.config['preprocess']['max_len'])
        elif self.config['model']['agg_function'] == 'mean':
            self.mean_pool = torch.nn.AvgPool1d(self.config['preprocess']['max_len'])
        self.linear = nn.Linear(2 * self.config['model']['hidden_num'], self.config['model']['type_num'])
    
    def forward(self, X):
        '''
        args:
            X: (batch_size, seq_len)
        '''
        embed_X = self.embedding(X).permute(1, 0, 2)
        output, _ = self.birnn(embed_X)
        if self.config['model']['agg_function'] == 'attention':
            linear_input = self.attention(output.permute(1, 0, 2))
        elif self.config['model']['agg_function'] == 'max':
            linear_input = self.max_pool(output.permute(1, 2, 0)).squeeze()
        elif self.config['model']['agg_function'] == 'mean':
            linear_input = self.mean_pool(output.permute(1, 2, 0)).squeeze()
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


class Bert(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        default_config = BertConfig.from_pretrained('bert-base-uncased').to_dict()
        for key, value in self.config['bert_model'].items():
            default_config[key] = value
        self.config['model']['embed_size'] = default_config['hidden_size']
        # vocab
        default_config['vocab_size'] = self.config['model']['vocab_size']
        default_config['pad_token_id'] = self.config['model']['pad_id']
        bert_config = BertConfig.from_dict(default_config)
        self.bert_model = BertModel(bert_config)
        if self.config['model']['load_wv_path'] != '':
            pretrain_weight = load_wv_as_embedding(
                self.config,
                torch.Size([self.config['model']['vocab_size'], self.config['model']['embed_size']]),
                vocab
                )
            self.bert_model.embeddings.word_embeddings = nn.Embedding.from_pretrained(
                pretrain_weight,
                padding_idx=self.config['model']['pad_id']
                )
        self.classifier = nn.Sequential(
            nn.Dropout(bert_config.hidden_dropout_prob),
            nn.Linear(bert_config.hidden_size, self.config['model']['type_num'])
        )
    
    def forward(self, X):
        bert_output = self.bert_model(input_ids=X[:, 0, :], attention_mask=X[:, 1, :])
        pool_output = bert_output.pooler_output   # (batch_size, hidden_num)
        output = self.classifier(pool_output)

        return output


class BaseOptimizer:
    def __init__(self, config, model):
        # init config
        self.config = config
        # init optimizer
        if self.config['train']['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config['train']['lr'],
                weight_decay=self.config['train'].get('weight_decay', 0)
                )
        else:
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.config['train']['lr'],
                weight_decay=self.config['train'].get('weight_decay', 0),
                momentum=self.config['train'].get('momentum', 0)
                )
        # init lr scheduler
        if self.config['train'].get('lr_scheduler') == 'step':
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['train']['lr_step_size'],
                gamma=self.config['train']['lr_step_gamma']
                )
        self.lr = self.optimizer.param_groups[0]['lr']

    def step(self):
        self.optimizer.step()
        if self.config['train']['lr_scheduler'] == 'step':
            self.lr_scheduler.step()
        self.lr = self.optimizer.param_groups[0]['lr']
    
    def zero_grad(self):
        self.optimizer.zero_grad()


class BertOptimizer:
    def __init__(self, config, model):
        self.config = config
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
                'weight_decay': config['train']['weight_decay']
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
                'weight_decay': 0.0
            }
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=config['train']['lr'])
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            self.config['train']['warmup_steps'],
            self.config['train']['train_steps']
            )
        self.lr = self.optimizer.param_groups[0]['lr']
    
    def step(self):
        self.optimizer.step()
        self.lr_scheduler.step()
        self.lr = self.optimizer.param_groups[0]['lr']

    def zero_grad(self):
        self.optimizer.zero_grad()