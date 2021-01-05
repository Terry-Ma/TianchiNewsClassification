import pandas as pd
import torch
import numpy as np
import logging
import yaml
import os

from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
from torchtext.vocab import Vocab

PAD = '<pad>'
UNK = '<unk>'
CLS = '<cls>'
data_path = '~/mayunquan/TianchiNewsClassification/data'
logger = logging.getLogger()

def generate_train_val_test_iter(train_X, train_y, val_X, val_y, test_X, config):
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=config['train']['batch_size'])  # shuffle=False
    test_dataset = torch.utils.data.TensorDataset(test_X, torch.zeros(test_X.shape[0]))
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=config['train']['batch_size'])
    logger.info('train iter & val & test iter generated')

    return train_iter, val_iter, test_iter

def generate_tensor(config):
    train_X, train_y, val_X, val_y, test_X = generate_train_val_test(config['preprocess']['is_demo'])
    train_X, train_y, val_X, val_y, test_X, vocab = data2tensor(
        train_X, train_y, val_X, val_y, test_X, config['preprocess']['max_len'], 
        config['preprocess']['min_freq'], config['model']['model_name'] == 'bert',
        config['preprocess']['vocab_path'])
    
    return train_X, train_y, val_X, val_y, test_X, vocab

def generate_train_val_test(is_demo):
    if is_demo:
        logger.info('generate train & val & test data: load demo')
        raw_train = pd.read_csv('{}/train_demo.csv'.format(data_path), sep='\t')
        test_X = pd.read_csv('{}/test_demo.csv'.format(data_path), sep='\t')['text']
    else:
        logger.info('generate train & val & test data: load total data')
        raw_train = pd.read_csv('{}/train.csv'.format(data_path), sep='\t')
        test_X = pd.read_csv('{}/test.csv'.format(data_path), sep='\t')['text']
    # train & val split
    train_X, val_X, train_y, val_y = train_test_split(
        raw_train['text'], raw_train['label'], test_size=0.1, random_state=2020, shuffle=True)
    for data in (train_X, train_y, val_X, val_y):
        data.reset_index(drop=True, inplace=True)

    return train_X, train_y, val_X, val_y, test_X

def data2tensor(train_X, train_y, val_X, val_y, test_X, max_len, min_freq, is_bert, vocab_path):
    # generate y
    train_y = torch.tensor(train_y)
    val_y = torch.tensor(val_y)
    # X truncated
    train_X, train_mask = truncated(train_X, max_len, is_bert)
    val_X, val_mask = truncated(val_X, max_len, is_bert)
    test_X, test_mask = truncated(test_X, max_len, is_bert)
    # generate vocab
    vocab = generate_vocab(train_X, min_freq, vocab_path)
    # transform
    train_X = train_X.apply(lambda x: [vocab.stoi[word] for word in x])
    val_X = val_X.apply(lambda x: [vocab.stoi[word] for word in x])
    test_X = test_X.apply(lambda x: [vocab.stoi[word] for word in x])
    # to tensor
    if not is_bert:
        train_X = torch.from_numpy(np.array(list(train_X)))
        val_X = torch.from_numpy(np.array(list(val_X)))
        test_X = torch.from_numpy(np.array(list(test_X)))
    else:
        train_X = torch.from_numpy(np.array([list(train_X), list(train_mask)])).permute(1, 0, 2)
        val_X = torch.from_numpy(np.array([list(val_X), list(val_mask)])).permute(1, 0, 2)
        test_X = torch.from_numpy(np.array([list(test_X), list(test_mask)])).permute(1, 0, 2)
    logger.info('data2tensor: train shape {}, val shape {}, test shape {}, vocab size {}'.format(\
        train_X.shape, val_X.shape, test_X.shape, len(vocab)))

    return train_X, train_y, val_X, val_y, test_X, vocab

def truncated(data, max_len, is_bert):
    data = data.apply(lambda x: x.split())
    pad_attention_mask = None
    if not is_bert:
        data = data.apply(lambda x: x[:max_len] if len(x) >= max_len else \
            x + [PAD] * (max_len - len(x)))
    else:
        pad_attention_mask = data.apply(lambda x: [1] * max_len if len(x) >= max_len - 1 else \
            [1] * (len(x) + 1) + [0] * (max_len - len(x) - 1))
        data = data.apply(lambda x: [CLS] + x[:max_len - 1] if len(x) >= max_len - 1 else \
            [CLS] + x + [PAD] * (max_len - len(x) - 1))
    
    return data, pad_attention_mask

def generate_vocab(train_X, min_freq, vocab_path):
    if os.path.exists(vocab_path):
        vocab = torch.load(vocab_path)
        logger.info('load existing vocab {}'.format(vocab_path))
    else:
        word2freq = defaultdict(int)
        for words in train_X:
            for word in words:
                word2freq[word] += 1
        if PAD in word2freq:
            del word2freq[PAD]
        vocab = Vocab(Counter(word2freq), min_freq=min_freq, specials=[UNK, PAD])
        torch.save(vocab, vocab_path)
        logger.info('generate vocab and save: {}'.format(vocab_path))

    return vocab

if __name__ == '__main__':
    with open('./rnn/config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)
    train_X, train_y, val_X, val_y, test_X, vocab = generate_tensor(config)
    train_iter, val_iter = generate_train_val_test_iter(train_X, train_y, val_X, val_y, test_X, config)