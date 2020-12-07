import pandas as pd
import lightgbm as lgb
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

data_path = '../data'
submit_path = '../submit'

def generate_freq_features(raw_train_X, raw_test_X):
    raw_text = pd.concat((raw_train_X, raw_test_X), axis=0)
    vectorizer = CountVectorizer(token_pattern='\w+')
    freq_features = vectorizer.fit_transform(raw_text).toarray()
    freq_features = freq_features / freq_features.sum(axis=1).reshape(-1, 1)

    return freq_features[:raw_train_X.shape[0], :], freq_features[raw_train_X.shape[0]:, :]

def generate_tf_idf_features(raw_train_X, raw_test_X):
    raw_text = pd.concat((raw_train_X, raw_test_X), axis=0)
    vectorizer = TfidfVectorizer(token_pattern='\w+')
    freq_features = vectorizer.fit_transform(raw_text).toarray()

    return freq_features[:raw_train_X.shape[0], :], freq_features[raw_train_X.shape[0]:, :]

def train_lgb(train_X, train_y, val_X, val_y, lgb_params):
    train_data = lgb.Dataset(train_X, train_y)
    val_data = lgb.Dataset(val_X, val_y)
    model = lgb.train(lgb_params, train_data, valid_sets=[train_data, val_data], verbose_eval=5, feval=f1_metrics)
    
    return model

def f1_metrics(preds, train_data):
    y_true = train_data.get_label()
    # preds shape (https://github.com/Microsoft/LightGBM/blob/a6e878e2fc6e7f545921cbe337cc511fbd1f500d/python-package/lightgbm/sklearn.py#L80-L81)
    y_preds = preds.reshape(-1, y_true.shape[0]).argmax(axis=0)
    
    return 'f1 score', f1_score(y_true, y_preds, average='macro'), True

def generate_submit(model, test_X, name):
    test_pred_y = model.predict(test_X).argmax(axis=1)
    submit = pd.DataFrame(test_pred_y, columns=['label'])
    submit.to_csv('{}/{}'.format(submit_path, name), index=False)

if __name__ == '__main__':
    raw_train = pd.read_csv('{}/train.csv'.format(data_path), sep='\t')
    raw_test_X = pd.read_csv('{}/test.csv'.format(data_path), sep='\t')['text']
    raw_train_X = raw_train['text']
    train_y = raw_train['label']
    # freq
    # train_freq_X, test_freq_X = generate_freq_features(raw_train_X, raw_test_X)
    # cur_train_X, cur_val_X, cur_train_y, cur_val_y = train_test_split(
    #     train_freq_X, train_y, test_size=0.1, random_state=2020, shuffle=True)
    freq_lgb_params = {
        'objective': 'multiclass',
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.02,
        'early_stopping_rounds': 200,
        'num_class': 14,
        'num_threads': 8
    }
    # freq_lgb_model = train_lgb(cur_train_X, cur_train_y, cur_val_X, cur_val_y, freq_lgb_params)
    # generate_submit(freq_lgb_model, test_freq_X, 'freq_lgb_submit.csv')
    # tf-idf
    train_tf_idf_X, test_tf_idf_X = generate_tf_idf_features(raw_train_X, raw_test_X)
    cur_train_X, cur_val_X, cur_train_y, cur_val_y = train_test_split(
        train_tf_idf_X, train_y, test_size=0.1, random_state=2020, shuffle=True)
    tf_idf_lgb_params = freq_lgb_params
    tf_idf_lgb_model = train_lgb(cur_train_X, cur_train_y, cur_val_X, cur_val_y, tf_idf_lgb_params)
    generate_submit(tf_idf_lgb_model, test_tf_idf_X, 'tf_idf_lgb_submit.csv')