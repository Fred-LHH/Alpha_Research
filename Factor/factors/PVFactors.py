import os
os.chdir('/Users/lihaohan/Alpha_Research')
import pandas as pd
import numpy as np
from Factor.base import BaseFactor, get_rolling_data
from mpire import WorkerPool
from Data.get_data import read_daily
from scipy.stats import skew, kurtosis
import re
from utils import *

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
    


class PVFactors(BaseFactor):
    """
    rtn5_mean: 5分钟收益率均值
    real_var: 已实现方差
    rtn_skew: 收益率偏度
    rtn_kurt: 收益率峰度
    rv_up: 上行收益率已实现方差
    rv_down: 下行收益率已实现方差
    rv_umd: 上下行已实现方差之差除以已实现方差
    对数成交量偏度
    对数成交量后尾(90%、10%占比)
    15min成交量变化率偏度
    15min成交量变化率峰度
    累计成交量均值
    累计成交量标准差
    cvr:收盘15min的成交量占当日比例
    ovr:开盘15min的成交量占当日比例
    pvi:放量时的分钟收益率之和
    vov:风险模糊度

    计算完成后还需对因子进行时序平滑
    """

    def __init__(self, 
                 factor_name='PVFactors', 
                 data_path='/Volumes/T7Shield/ProcessedData', 
                 factor_parameters={}, 
                 save_path='/Volumes/T7Shield/Alpha_Research/Factor'):
        super(PVFactors, self).__init__(factor_name=factor_name,
                                 factor_parameters=factor_parameters,
                                 data_path=data_path, save_path=save_path)
        
    
    def prepare_data(self):
        min_files = os.listdir(self.data_path)
        min_files = [f for f in min_files if f.endswith('.pkl') and not f.startswith('._')]

        return min_files
    
    def generate_factor(self, start_date, end_date):
        files = self.prepare_data()
        
        def calculator(file):
            try:
                code = file.split('.')[0]
                data = pd.read_pickle(os.path.join(self.data_path, file))
                data = data[['date', 'code', 'volume', 'high', 'low', 'close']].assign(
                    ret=data['close'] / data['close'].shift(1) - 1
                )
                data.dropna(inplace=True)

                def _calculate_factors(group):
                    """ 单次分组计算所有因子 """
                    factors = {}

                    # 收益率基础因子
                    ret = group['ret'].values
                    factors['rtn5_mean'] = convert_freq(ret, '5T', 'sum').mean()
                    factors['real_var'] = np.power(ret, 2).sum()
                    factors['rtn_skew'] = skew(ret)
                    factors['rtn_kurt'] = kurtosis(ret)

                    # 上下行波动率
                    up_ret = ret[ret > 0]
                    down_ret = ret[ret < 0]
                    factors['rv_up'] = np.power(up_ret, 2).sum()
                    factors['rv_down'] = np.power(down_ret, 2).sum()
                    factors['rv_umd'] = (factors['rv_up'] - factors['rv_down']) / factors['real_var'] if factors['real_var'] != 0 else np.nan

                    # 成交量相关
                    log_vol = np.log(group['volume'] + 1e-6)
                    factors['log_volume_skew'] = skew(log_vol)
                    thre_90tail = np.percentile(log_vol, 90)
                    thre_10tail = np.percentile(log_vol, 10)
                    factors['log_volume_90tail'] = log_vol[log_vol > thre_90tail].sum() / log_vol.sum()
                    factors['log_volume_10tail'] = log_vol[log_vol < thre_10tail].sum() / log_vol.sum()

                    # 15分钟成交量变化 开收盘15分钟成交占比
                    vol_15T = convert_freq(group['volume'].values, '15T', 'sum')
                    if len(vol_15T) >= 2:
                        factors['ovr'] = vol_15T[0] / vol_15T.sum()
                        factors['cvr'] = vol_15T[-1] / vol_15T.sum()
                        vol_change = np.diff(vol_15T) / vol_15T[:-1]
                        factors['vol_change_skew'] = skew(vol_change[np.isfinite(vol_change)])
                        factors['vol_change_kurt'] = kurtosis(vol_change[np.isfinite(vol_change)])
                    else:
                        factors.update({'ovr': np.nan, 'cvr': np.nan, 'vol_change_skew': np.nan, 'vol_change_kurt': np.nan})
                    
                    cum_vol = group['volume'].cumsum()
                    factors['cumvol_mean'] = cum_vol.mean()
                    factors['cumvol_std'] = cum_vol.std()
                    factors['pvi'] = group['ret'][group['volume'] > group['volume'].mean()].sum()

                    _ret = ret[14:-15]
                    _ret = convert_freq(_ret, '5T', method='sum')
                    intraday_var1 = _ret.std(ddof=1)
                    high_5T = convert_freq(group['high'].values, '5T', method='max')
                    low_5T = convert_freq(group['low'].values, '5T', method='min')
                    constant = 1 / (4 * np.log(2) * len(low_5T))
                    intraday_var2 = np.sqrt(constant * np.sum(np.power(np.log(high_5T / low_5T), 2)))
                    factors['vov1'] = intraday_var1
                    factors['vov2'] = intraday_var2
                    """
                    计算完日内波动率之后, 为了剔除个股日内波动自身量级对VoV的影响, 我们需要在计算日间标准差之前预先对日内标准差在横截面上进行标准化处理, 计算Z-score值。使用40个交易日(约两个月_)内标准化后的计算标准差得到最终的VoV, 标准差使用时间周期基本与5分钟涨跌幅的样本量保持一致.
                    """

                    return pd.Series(factors)
                
                factor_df = data.groupby(['date', 'code']).apply(_calculate_factors).reset_index()
                return factor_df

            except Exception as e:
                print(f"Error processing {file}: {e}")
                return pd.DataFrame()

        with WorkerPool(n_jobs=4) as pool:
            results = pool.map(calculator, files, progress_bar=True)
        
        self.factor = pd.concat(results).reset_index(drop=True)
        
    
    def run(self, pool: bool = False):
        self.generate_factor()
        if pool:
            self.clear_factor()
        self.save()
        



def cal_vov(vov1, vov2, window=40, min_periods=20):
    """
    计算风险模糊度
    Args:
    -------
        vov1: 日内5分钟收益率的波动率
        vov2: 日内5分钟收益率的Parkinson波动率
    """
    vov1 = vov1.pivot(index='date', columns='code', values=vov1.columns[-1])
    vov2 = vov2.pivot(index='date', columns='code', values=vov2.columns[-1])

    vov1, vov2 = standardlize(vov1), standardlize(vov2)

    if min_periods is None:
        min_periods = int(window * 0.5)

    vov_factor1 = vov1.rolling(window=window, min_periods=min_periods).std()
    vov_factor2 = vov2.rolling(window=window, min_periods=min_periods).std()

    return vov_factor1, vov_factor2





