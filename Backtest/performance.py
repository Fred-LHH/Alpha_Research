import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Tuple

from decorators import do_on_dfs
from Data.utils import DButils
from config import *
from Data.get_data import read_daily, read_market

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['axes.unicode_minus'] = False


def bt_perf(
        nets: pd.Series,
        rets: pd.Series,
        counts_one_year: 12,
) -> pd.Series:
    '''
    Params:
    ----------
    nets: pd.Series
        净值序列, index为日期
    rets: pd.Series
        收益率序列, index为日期
    counts_one_year: int
        一年内交易了多少次
    
    Returns:
    ----------
    pd.Series
        评价结果包括年化收益率、总收益率、年化波动率、年化夏普、最大回撤率、胜率
    '''
    n_series = nets.copy()
    r_series = rets.copy()
    duration = (n_series.index[-1] - n_series.index[0]).days
    year = duration / 365
    ret_yearly = n_series.iloc[-1] / year
    max_drawdown = -((n_series + 1) / (n_series+1).expanding(1).max() - 1).min()
    vol = np.std(r_series) * (counts_one_year ** 0.5)
    sharpe = ret_yearly / vol
    wins = r_series[r_series > 0]
    win_rate = len(wins) / len(r_series)
    return pd.Series(
        [n_series.iloc[-1], ret_yearly, vol, sharpe, max_drawdown, win_rate],
        index = ['总收益率', '年化收益率', '年化波动率', '信息比率', '最大回撤率', '胜率']
    )


@do_on_dfs
def bt_relative_perf(
    ret: pd.Series,
    hs300: bool = 0,
    zz500: bool = 0,
    zz1000: bool = 0,
    sz50: bool = 0,
    sz180: bool = 0,
    zxb: bool = 0,
    start_dt: str = START_DATE,
    end_dt: str = DATE2,
    show_ex_nets: bool = 0,
    freq: str = 'W',
    plot: bool = 0,
) -> pd.Series:
    '''
    对于给定投资组合的收益率序列, 计算相对于指定指数的超额表现

    Params:
    ----------
    ret: pd.Series
        投资组合收益率序列, index为日期
    hs300: bool, optional
        是否计算相对于沪深300的超额表现, by default 0
    zz500: bool, optional
        是否计算相对于中证500的超额表现, by default 0
    zz1000: bool, optional
        是否计算相对于中证1000的超额表现, by default 0
    sz50: bool, optional
        是否计算相对于上证50的超额表现, by default 0
    sz180: bool, optional
        是否计算相对于上证180的超额表现, by default 0
    zxb: bool, optional
        是否计算相对于中小板的超额表现, by default 0
    start_dt: str, optional
        起始日期, 如'20180101', by default START_DATE
    show_ex_nets: bool, optional
        是否返回超额净值数据, by default 0

    Returns:
    ----------
    pd.Series
        评价结果包括年化收益率、总收益率、年化波动率、年化夏普、最大回撤率、胜率
    '''
    indices = {
        '399300': hs300,
        '399905': zz500,
        '399852': zz1000,
        '000016': sz50,
        '000010': sz180,
        '399005': zxb
    }

    offsets = {
        'W': pd.DateOffset(weeks=1),
        'M': pd.DateOffset(months=1)}

    frequency = {
        'W': 52,
        'M': 12
    }

    indice = [index for index, value in indices.items() if value]
    if not indice or len(indice) > 1:
        raise ValueError('请指定一个指数')
    index = indice[0]
    net_index = read_market(close=1, index=index, start_date=start_dt, end_date=end_dt)
    start_dt, end_dt = pd.Timestamp(start_dt), pd.Timestamp(end_dt)
    
    net_index.index = pd.to_datetime(net_index.index)
    if freq:
        net_index = net_index.resample(freq).last()
    ret_index = net_index.pct_change()
    if start_dt and end_dt:
        ret_index = ret_index.loc[start_dt:end_dt]
    
    ret = ret - ret_index
    ret = ret.dropna()
    net = ret.cumsum()

    rtop = pd.Series(0, index=[net.index.min() - offsets.get(freq, pd.DateOffset(days=1))])
    net = pd.concat([rtop, net])
    ret = pd.concat([rtop, ret])

    if freq:
        net = net.resample(freq).last()
        ret = ret.resample(freq).last()

    counts = frequency.get(freq, 252)
    com = bt_perf(net, ret, counts)
    if plot:
        p_net = net.ffill()
        p_net.iplot()
    if show_ex_nets:
        return com, net
    else:
        return com


