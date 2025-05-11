import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, shapiro, norm
from statsmodels.stats.diagnostic import acorr_ljungbox
import re

def convert_freq(data,
                 freq,
                 method='sum'):
    '''
    对日内股票分钟数据进行重采样
    Args:
        data: array, 日内股票分钟数据
        freq: str, 重采样频率，如'5T'表示5分钟
        method: str, 重采样方法，'sum'或'mean'
    '''
    if not method or method not in ['sum', 'mean']:
        raise ValueError('Do not support method:%s' % method)
    min = re.match(r'^(\d+)T$', freq)
    if not min or int(min.group(1)) < 1:
        raise ValueError(f"Invalid freq format: {freq}, should be like '5T'")
    k = int(min.group(1))

    data = np.asarray(data)
    if len(data) < k:
        return np.array([np.nan])
    
    n = len(data)
    m, remainder = divmod(n, k)

    resampled = []
    if m > 0:
        main = data[:m*k].reshape(m, k)
        resampled.append(main.sum(1) if method == 'sum' else main.mean(1))
    
    # 处理尾部
    if remainder > 0:
        tail = data[m*k:]
        tail_value = tail.sum() if method == 'sum' else tail.mean()
        resampled.append(np.array([tail_value]))  

    return np.concatenate(resampled) if resampled else np.array([])


def remove_None(array):
    return array[~np.isnan(array)] if len(array) > 0 else None


def mean(data):
    data = remove_None(data)
    return data.mean() if len(data) > 0 else None


def skewness(data):
    data = remove_None(data)
    return skew(data) if len(data) > 0 else None


def delta(data):
    data = remove_None(data)
    return np.diff(data)


def kurt(data):
    data = remove_None(data)
    return kurtosis(data) if len(data) > 0 else None

def realized_var(data):
    return ((delta(data))**2).sum() if len(data) > 0 else None

def filter(data, positive=True):
    if positive:
        return data[data > 0] if len(data) > 0 else None
    return data[data < 0] if len(data) > 0 else None

def Shapiro_Wilk(data):
    data = remove_None(data)
    W, p = shapiro(data)
    sw_value = 1 - W
    return W, p, sw_value

def log(data):
    data = remove_None(data)
    return np.log(data)

def median(data):
    data = remove_None(data)
    return np.median(data) if len(data) > 0 else None

def abs(data):
    data = remove_None(data)
    return np.abs(data)
    




