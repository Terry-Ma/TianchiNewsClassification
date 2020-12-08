import pandas as pd
import torch
import numpy as np
import logging
import yaml

from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
from torchtext.vocab import Vocab

PAD = '<pad>'
UNK = '<unk>'
data_path = '~/mayunquan/TianchiNewsClassification/data'
logger = logging.getLogger()

def generate_train_val_iter(train_X, train_y, val_X, val_y, config):
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    logger.info('train iter & val iter generated')

    return train_iter, val_iter

def generate_tensor(config):
    train_X, train_y, val_X, val_y, test_X = generate_train_val_test(config['preprocess']['is_demo'])
    train_X, train_y, val_X, val_y, test_X, vocab = data2tensor(
        train_X, train_y, val_X, val_y, test_X, config['preprocess']['max_len'], config['preprocess']['min_freq'])
    
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

def data2tensor(train_X, train_y, val_X, val_y, test_X, max_len, min_freq):
    # generate y
    train_y = torch.tensor(train_y)
    val_y = torch.tensor(val_y)
    # X truncated
    train_X = truncated(train_X, max_len, PAD)
    val_X = truncated(val_X, max_len, PAD)
    test_X = truncated(test_X, max_len, PAD)
    # generate vocab
    vocab = generate_vocab(train_X, min_freq)
    # transform
    train_X = train_X.apply(lambda x: [vocab.stoi[word] for word in x])
    val_X = val_X.apply(lambda x: [vocab.stoi[word] for word in x])
    test_X = test_X.apply(lambda x: [vocab.stoi[word] for word in x])
    # to tensor
    train_X = torch.from_numpy(np.array(list(train_X)))
    val_X = torch.from_numpy(np.array(list(val_X)))
    test_X = torch.from_numpy(np.array(list(test_X)))
    logger.info('data2tensor: train shape {}, val shape {}, test shape {}, vocab size {}'.format(\
        train_X.shape, val_X.shape, test_X.shape, len(vocab)))

    return train_X, train_y, val_X, val_y, test_X, vocab

def truncated(data, max_len, PAD):
    data = data.apply(lambda x: x.split())
    data = data.apply(lambda x: x[:max_len] if len(x) >= max_len else \
        x + [PAD] * (max_len - len(x)))
    
    return data

def generate_vocab(train_X, min_freq):
    word2freq = defaultdict(int)
    for words in train_X:
        for word in words:
            word2freq[word] += 1
    if PAD in word2freq:
        del word2freq[PAD]
    vocab = Vocab(Counter(word2freq), min_freq=min_freq, specials=[UNK, PAD])

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
    train_iter, val_iter = generate_train_val_iter(train_X, train_y, val_X, val_y, config)