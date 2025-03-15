import os
import logging
import logging.handlers
from datetime import datetime

def create_log_dir(log_dir):
    log_dir = os.path.expanduser(log_dir)
    if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return

def stream_handler(formats):
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formats)
    return stream_handler

def file_handler(file_name, formats):
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=file_name, when='D', backupCount=7, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formats)
    return file_handler


def get_logger(name):
    # 创建日志文件
    name = name.replace('.py', '')
    root_path, file_name = os.path.split(os.path.realpath(__file__))
    log_dir = os.path.join(root_path, '../logs')
    create_log_dir(log_dir)

    logger_format = '%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(threadName)s - %(message)s'
    formats = logging.Formatter(logger_format)

    _logger = logging.getLogger('{}_{}'.format(name, datetime.now().strftime('%Y%m%d')))
    _logger.setLevel(logging.DEBUG)

    # 按指定格式输出到文件
    file_name = os.path.join(log_dir, '{}.log'.format(name))
    fh = file_handler(file_name, formats)
    _logger.addHandler(fh)

    # 按指定格式输出到控制台
    sh = stream_handler(formats)
    # _logger.addHandler(sh)
    fh.close()
    # sh.close()
    return _logger


dateslogger = get_logger('trade_dates')
stockslogger = get_logger('stocks')
