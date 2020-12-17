import pandas as pd
import random

from gensim.models import Word2Vec, KeyedVectors

data_path = '../data/'
save_path = './skip_gram.wv'

class SentencesGenerator:
    def __init__(self):
        train_text = pd.read_csv('{}/train.csv'.format(data_path), sep='\t')['text']
        test_text = pd.read_csv('{}/test.csv'.format(data_path), sep='\t')['text']
        self.sentences = pd.concat((train_text, test_text), axis=0).reset_index(drop=True)
    
    def __iter__(self):
        indexs = list(range(self.sentences.shape[0]))
        random.shuffle(indexs)
        for index in indexs:
            yield self.sentences[index].split()

def train_skipgram(sentences_generator):
    model = Word2Vec(
        sentences=sentences_generator,
        size=128,
        window=5,
        min_count=5,
        sg=1,  # skip-gram
        iter=20,  # epoch
        workers=8
        )
    word_vectors = model.wv
    word_vectors.save(save_path)

def load_test():
    sk_wv = KeyedVectors.load(save_path, mmap='r')
    print(type(sk_wv))
    print(sk_wv['3659'])
    # print(sk_wv['myq'])   # raise Exception
    print('myq' in sk_wv)
    print('0' in sk_wv)

if __name__ == '__main__':
    # sentences_generator = SentencesGenerator()
    # train(sentences_generator)
    load_test()