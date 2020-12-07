import pandas as pd
import numpy as np
import fasttext
import sys
sys.path.append('../')

from sklearn.metrics import f1_score
from utils import *

train_path = './fasttext_train'
val_path = './fasttext_val'
model_path = './fasttext_model.bin'
submit_path = '../submit'

def train_fasttext(train_X, train_y, val_X, val_y):
    # generate train & val file
    with open(train_path, 'w') as f,\
        open(val_path, 'w') as g:
        for i in range(train_X.shape[0]):
            f.write('__label__{} '.format(train_y.iloc[i]) + train_X.iloc[i] + '\n')
        for i in range(val_X.shape[0]):
            g.write('__label__{} '.format(val_y.iloc[i]) + val_X.iloc[i] + '\n')
    # train fasttext: default optimization metric is f1
    model = fasttext.train_supervised(
        input=train_path,
        autotuneValidationFile=val_path,
        autotuneDuration=3600,  # 1h
        )
    model.save_model(model_path)
    # val f1
    print('f1 score {}'.format(val_metric(val_X, val_y)))

def val_metric(val_X, val_y):
    model = fasttext.load_model(model_path)
    val_pred_res = model.predict(list(val_X))
    val_pred_y = [int(i[0].split('__')[-1]) for i in val_pred_res[0]]
    f1 = f1_score(val_y, val_pred_y, average='macro')

    return f1

def generate_submit(test_X, name):
    model = fasttext.load_model(model_path)
    test_pred_res = model.predict(list(test_X))
    test_pred_y = [int(i[0].split('__')[-1]) for i in test_pred_res[0]]
    submit = pd.DataFrame(test_pred_y, columns=['label'])
    submit.to_csv('{}/{}'.format(submit_path, name), index=False)

if __name__ == '__main__':
    train_X, train_y, val_X, val_y, test_X = generate_train_val_test(is_demo=False)
    train_fasttext(train_X, train_y, val_X, val_y)
    generate_submit(test_X, 'fasttext_submit.csv')