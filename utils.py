import logging

logger = logging.getLogger()

def set_logger(log_path='./rnn/train_log/train.log'):
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
    set_logger()
    logger.info('set logger sucess')