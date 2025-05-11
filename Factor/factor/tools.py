import os
os.chdir('/Users/lihaohan/Desktop/quant/毕业设计')
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from collections import defaultdict
from tqdm import tqdm
from utils.factor import *
import re

def process_pv_data(data: pd.DataFrame, fields: list):
    try:
        if 'Unnamed: 0' in data.columns:
            data = data.drop('Unnamed: 0', axis=1)
        data.reset_index(drop=True, inplace=True)
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        #data['time'] = data['datetime'].apply(lambda x: x[-8:].replace(':', '')[:4])
        for field in fields:
            if field not in ['date', 'code', 'time']:
                data[field] = data[field].astype(float)
        return data[fields]
    except:
        code = data['code'][0]
        print(f'{code} has error')


def cal_factors(data_file, 
                factor_names,
                params=None):
    """多因子计算核心函数
    Args:
        data_file: 数据文件名
        factor_names: 需要计算的因子列表

    Returns:
        dict: {因子名称: 对应的DataFrame}
    """
    base_path = '/Users/lihaohan/Desktop/研报复现/测试数据'
    params = params or {}
    try:
        df = pd.read_pickle(os.path.join(base_path, data_file))
        df = process_pv_data(df, ['date', 'code', 'time', 'volume', 'close', 'open', 'high', 'low'])
        if df.empty:
            name = data_file.split('.')[0]
            raise ValueError(f'{name} is empty')

        results = {}
        for factor in factor_names:
            try:
                factor_func = globals()[f'cal_{factor}']
                param = params.get(factor, {})
                args = param.get('args', ())
                kwargs = param.get('kwargs', {})

                factor_df = factor_func(df.copy(), *args, **kwargs)  # 避免数据污染
                results[factor] = factor_df
            except Exception as e:
                print(f'[{data_file}] {factor} calculation failed: {e}')
        
        return results
    
    except Exception as e:
        print(f'File processing failed [{data_file}]: {e}')
        return None

def multi_process_calculate(data_files, 
                            factor_names,
                            params=None):
    """多进程多因子计算
    Args:
        data_files: 数据文件列表
        factor_names: 需要计算的因子名称列表
    Returns:
        dict: {因子名称: 合并后的DataFrame}
    """
    missing_factors = [f for f in factor_names if f'cal_{f}' not in globals()]
    if missing_factors:
        raise ValueError(f"Missing calculation functions for: {missing_factors}")

    # 存储结果
    result_collector = defaultdict(list)
    
    with ProcessPoolExecutor() as executor:
        process_func = partial(cal_factors, 
                             factor_names=factor_names,
                             params=params or {})
        results = list(tqdm(executor.map(process_func, data_files), total=len(data_files)))

    for file_result in results:
        if file_result:  
            for factor_name, df in file_result.items():
                if df is not None and not df.empty:
                    result_collector[factor_name].append(df)

    final_results = {}
    for factor, dfs in result_collector.items():
        if dfs:
            final_results[factor] = pd.concat(dfs, ignore_index=True)
        else:
            final_results[factor] = pd.DataFrame()
    
    return final_results



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
        return np.nan
    
    n = len(data)
    m, remainder = divmod(n, k)

    resampled = []
    if m > 0:
        main = data[:m*k].reshape(m, k)
        resampled.append(main.sum(1) if method == 'sum' else main.mean(1))
    
    # 处理尾部
    if remainder > 0:
        tail = data[m*k:]
        resampled.append(tail.sum() if method == 'sum' else tail.mean())
    
    return np.concatenate(resampled) if resampled else np.array([])
    




