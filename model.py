import torch
import logging
import os

from data_process import *
from torch import nn
from modules import *
from sklearn.metrics import f1_score, classification_report

name2model = {
    'rnn': BiRNN,
    'bert': Bert
}

name2optimizer = {
    'rnn': BaseOptimizer,
    'bert': BertOptimizer
}

logger = logging.getLogger()

class Model:
    def __init__(self, config):
        self.init_data(config)
        self.init_config(config)
        self.init_device()
        self.init_model()
        self.init_loss()
        self.init_optimizer()

    def init_data(self, config):
        self.train_X, self.train_y, self.val_X, self.val_y, self.test_X, self.vocab = \
            generate_tensor(config)
        self.train_iter, self.val_iter, self.test_iter = generate_train_val_test_iter(\
            self.train_X, self.train_y, self.val_X, self.val_y, self.test_X, config)

    def init_config(self, config):
        self.config = config
        self.config['model']['vocab_size'] = len(self.vocab)
        self.config['model']['type_num'] = self.train_y.unique().shape[0]
        self.config['model']['pad_id'] = self.vocab.stoi[PAD]

    def init_device(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info('use device {}'.format(self.device))

    def init_model(self):
        ModelStruct = name2model[self.config['model']['model_name']]
        self.model = ModelStruct(self.config, self.vocab).to(self.device)
        logger.info('init model \n{}'.format(self.model))
        # multi-gpu
        if self.config['train']['multi_gpu'] and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info('will train on {} gpus'.format(torch.cuda.device_count()))

    def init_loss(self):
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        logger.info('init loss \n{}'.format(self.loss))

    def init_optimizer(self):
        Optimizer = name2optimizer[self.config['model']['model_name']]
        self.optimizer = Optimizer(self.config, self.model)
        logger.info('init optimizer\n {}'.format(self.optimizer))

    def train(self):
        self.cur_epochs = 1
        self.cur_train_steps = 1
        self.max_val_f1 = 0
        self.best_step = 1
        # load checkpoint
        if self.config['model']['load_checkpoint'] != '':
            self.load_checkpoint()
        logger.info('start training')
        while self.cur_train_steps <= self.config['train']['train_steps']:
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
                if self.cur_train_steps > 0 and self.cur_train_steps % self.config['train']['steps_per_check'] == 0:
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
                    # log
                    logger.info('epoch {0}, steps {1}, train_loss {2:.4f}, val_loss {3:.4f}, train_f1 {4:.4f}, val_f1 {5:.4f}, max_val_f1 {6:.4f}, best_step {7}, lr {8}'.format(
                        self.cur_epochs,
                        self.cur_train_steps,
                        check_train_loss / check_train_steps,
                        check_val_loss / check_val_steps,
                        check_train_f1,
                        check_val_f1,
                        self.max_val_f1,
                        self.best_step,
                        self.optimizer.lr
                        ))
                    # max f1 model
                    if check_val_f1 > self.max_val_f1:
                        self.max_val_f1 = check_val_f1
                        self.best_step = self.cur_train_steps
                        cpt_path = './checkpoint/{}/best_model.cpt'.format(self.config['train']['checkpoint_dir'])
                        self.save_checkpoint(cpt_path)
                    check_train_steps = 0
                    check_train_loss = 0
                    check_train_y = np.array([])
                    check_train_pred_y = np.array([])
                    self.model.train()    # dropout...
                # checkpoint
                if self.cur_train_steps > 0 and self.cur_train_steps % self.config['train']['steps_per_checkpoint'] == 0:
                    cpt_path = './checkpoint/{}/checkpoint_steps_{}.cpt'.\
                        format(self.config['train']['checkpoint_dir'], self.cur_train_steps)
                    self.save_checkpoint(cpt_path)
                self.cur_train_steps += 1
                if self.cur_train_steps > self.config['train']['train_steps']:
                    break
            self.cur_epochs += 1 
        logger.info('training complete, training epochs {0}, steps {1}, max_val_f1 {2:.4f}, best_step {3}'.\
            format(self.cur_epochs, self.config['train']['train_steps'], self.max_val_f1, self.best_step))
        # val analyse
        self.val_analyse()
        # generate submit
        self.generate_submit()

    def save_checkpoint(self, cpt_path):
        cpt_dict = {  # DataParallel's state_dict is in module
            'model': self.model.module.state_dict() if self.config['train']['multi_gpu'] and\
                torch.cuda.device_count() > 1 else self.model.state_dict(),
            'cur_train_steps': self.cur_train_steps,
            'cur_epochs': self.cur_epochs,
            'max_val_f1': self.max_val_f1,
            'best_step': self.best_step
            }
        for k, v in self.optimizer.state_dict().items():
            cpt_dict[k] = v
        torch.save(cpt_dict, cpt_path)
        logger.info('save checkpoints {}'.format(cpt_path))

    def load_checkpoint(self):
        logger.info('will load checkpoint')
        cpt_path = './checkpoint/{}'.format(self.config['model']['load_checkpoint'])
        if os.path.exists(cpt_path):
            # load checkpoint
            cpt_dict = torch.load(cpt_path)
            self.model.load_state_dict(cpt_dict['model'])
            self.optimizer.load_state_dict(cpt_dict)
            self.cur_train_steps = cpt_dict['cur_train_steps'] + 1
            self.cur_epochs = cpt_dict['cur_epochs'] + 1
            self.max_val_f1 = cpt_dict['max_val_f1']
            self.best_step = cpt_dict['best_step']
            logger.info('load checkpoint from {}'.format(cpt_path))
        else:
            logger.error('checkpoint path {} not exists'.format(cpt_path))
            raise Exception('checkpoint path {} not exists'.format(cpt_path))

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