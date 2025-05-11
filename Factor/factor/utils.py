import numpy as np
import pandas as pd
import re
import statsmodels.api as sm
from numpy_ext import rolling_apply
from joblib import Parallel, delayed
import copy, math
from typing import Union


def calc_first_diff_corr(ser: Union[pd.Series, np.array]):
    """计算一阶差分相关系数
    Args:
        ser (pd.Series | np.array)
        
    Returns:
        pd.Series:一阶差分相关系数
    """
    ser_diff = np.diff(ser)
    correlation = np.corrcoef(ser[:-1], ser_diff)[0,1]
    
    return correlation

def cal_DW_stats(ser):
    diff = np.diff(ser)
    dw = np.sum(diff ** 2) / np.sum(ser ** 2)
    return dw  



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
    

def drop_row(df,
             col,
             value_list):
    drop_all = pd.DataFrame(columns=df.columns)
    for value in value_list:
        if type(value) == type(np.nan):
            if np.isnan(value):
                df = df.reset_index(drop=True)
                deleted = df[df[col].isnull() == True]
                df = df.drop(df.index[df[col].isnull() == True].values)
                df = df.reset_index(drop=True)
            else:
                df = df.reset_index(drop=True)
                deleted = df[df[col] == value]
                df = df.drop(df.index[df[col] == value].values)
                df = df.reset_index(drop=True)

            drop_all = pd.concat([drop_all, deleted], ignore_index=True)
        return df, drop_all
    

def nmerge(df,
          df_add,
          on):
    # 与pd.merge()功能相同, 避免列被重新命名(新加的为a_, 并且保持on的列不变)
    new_name = ['a_' + x for x in df_add.columns]
    new_name = dict(zip(df_add.columns, new_name))
    df_add = df_add.rename(columns = new_name)
    new_name = ['a_' + x for x in on]
    new_name = dict(zip(new_name, on))
    df_add = df_add.rename(columns = new_name)

    return df.merge(df_add, on=on)

def divide_row_in_year(df):
    # 将df的所有行按照年份划分, 返回一个列表包含各年度的行
    df['date'] = pd.to_datetime(df['date'])
    first_year = df['date'].min().year
    last_year = df['date'].max().year
    result = []
    for y in range(first_year, last_year+1):
        select = list(map(lambda x: x.year == y, df['date']))
        df_ = df[select]
        result.append(df_)
    return result


def ar1_residuals(data, T_beta: bool):
    # 检查输入数据是否为一维numpy数组
    if not isinstance(data, np.ndarray) or len(data.shape) != 1:
        raise ValueError("输入数据必须是一维的numpy数组")
    
    nan_inf_count = np.isnan(data).sum() + np.isinf(data).sum()
    ratio = nan_inf_count / len(data)
    if ratio > 0.5:
        return None
    else:
        data = data[~np.isnan(data) & ~np.isinf(data)]

    y = data[1:]
    X = data[:-1]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    residuals = results.resid
    if T_beta == False:
        if len(residuals) == 0:
            return None
        else:
            return residuals
    else:
        y_res = residuals[1:]
        X_res = residuals[:-1]
        X_res = sm.add_constant(X_res)
        model_res = sm.OLS(y_res, X_res)
        results_res = model_res.fit()
        if len(results_res.params) == 0:
            return None
        else:
            return results_res.params[1]


def parallel(df,
             func,
             n_core):
    len_df = len(df)
    sp = list(range(len_df)[::int(len_df / n_core + 0.5)])[:-1]
    sp.append(len_df)
    slc_gen = (slice(*idx) for idx in zip(sp[:-1], sp[1:]))
    res = Parallel(n_jobs=n_core)(delayed(func)(df[slc]) for slc in slc_gen)
    return pd.concat(res)




