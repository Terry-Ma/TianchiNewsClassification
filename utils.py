import pandas as pd
import torch
import numpy as np
import logging

from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
from torchtext.vocab import Vocab

PAD = '<pad>'
UNK = '<unk>'
data_path = '~/mayunquan/news_classification/data'
logger = logging.getLogger()

def data_process(is_demo=False, max_len=1024, min_freq=5):
    train_X, train_y, val_X, val_y, test_X = generate_train_val_test(is_demo)
    train_X, train_y, val_X, val_y, test_X, vocab = data2tensor(
        train_X, train_y, val_X, val_y, test_X, max_len, min_freq)
    
    return train_X, train_y, val_X, val_y, test_X, vocab

def generate_train_val_test(is_demo):
    if is_demo:
        logger.info('load demo')
        raw_train = pd.read_csv('{}/train_demo.csv'.format(data_path), sep='\t')
        test_X = pd.read_csv('{}/test_demo.csv'.format(data_path), sep='\t')['text']
    else:
        logger.info('load total data')
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

if __name__ == '__main__':
    train_X, train_y, val_X, val_y, test_X, vocab = data_process(is_demo=False)
    print(train_X.shape)
    print(train_y.shape)
    print(len(vocab))