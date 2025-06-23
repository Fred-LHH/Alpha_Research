import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rcParams['axes.unicode_minus'] = False
rcParams['font.sans-serif'] = ['SimHei']  
rcParams['axes.unicode_minus'] = False 
import os
from .utils import *
from .post import *
from Data.tushare.api import TSDataFetcher
import tushare as ts
ts.set_token('18427aa0a10e23a2bf2bf2de0b240aa0005db0629feea9fa2a3bd6a8')
pro = ts.pro_api()
api = TSDataFetcher()

class Backtest_Factor:

    def __init__(self, market, return_type='open'):
        self.market = market
        # 预期当天触发信号，次日所获得收益
        if return_type =='open':
            self.sr = market.groupby('code')['open'].apply(lambda x: x.shift(-2)/x.shift(-1) - 1).droplevel(0)
        elif return_type == 'close':
            self.sr = market.groupby('code')['close'].apply(lambda x: x.shift(-1)/x - 1).droplevel(0)
        elif return_type == 'overnight':
            self.sr = market.groupby('code').apply(lambda x: x['open'].shift(-1)/x['close'] - 1).droplevel(0)
        elif return_type == 'inday':
            self.sr = market.groupby('code').apply(lambda x: x['close'].shift(-1)/x['open'].shift(-1) - 1).droplevel(0)
        self.sr = self.sr.dropna()

    def get_oi_df(self, factor, cal_type=0, cal_param=[0.3, 0.7,  '9999d','365d']):
        """给出因子值，计算进出场信号
        factor: 因子值, index:date
        cal_type计算类型:
            0: 按照历史分位数, 参数: [0.1, 0.9, '9999d', '200d'], 最小10%做空, 90%后做多, 窗口9999d,最小窗口'200d'
            1: 绝对数值, 参数：[a, b, factor2, c. d], 小于a空开, 小于b多开, factor2平仓因子, None表示没有平仓因子
        """
        factor = factor.dropna()
        self.fname = factor.name
        def fun0(x):  
            x = x.set_index('date')
            min_date = x.index.min() + pd.Timedelta(cal_param[3])
            x['short'] = x[self.fname].rolling(cal_param[2]).quantile(cal_param[0])
            x['long'] = x[self.fname].rolling(cal_param[2]).quantile(cal_param[1])
            x = x.loc[min_date: ].reset_index().set_index(['date', 'code'])
            return  x

        if cal_type == 0:
            factor = factor.dropna()
            f = factor.reset_index().groupby('code')[['date', 'code', self.fname]].apply(lambda x: fun0(x)).droplevel(0)
            io_df = pd.DataFrame(index=self.market.index, columns=['lo', 'so'])
            io_df['lo'] = f['long'] < f[self.fname]
            io_df['so'] = f['short'] > f[self.fname]
        elif cal_type == 1:
            lo = factor >= cal_param[1]
            so = factor <= cal_param[0]
            if cal_param[2] is None:
                io_df = pd.DataFrame(index=self.market.index, columns=['lo', 'so'])
            else:
                io_df = pd.DataFrame(index=self.market.index, columns=['lo', 'so', 'sc', 'lc'])
                factor2 = cal_param[2]
                io_df['sc'] = factor2 <= cal_param[3]
                io_df['lc'] = factor2 >= cal_param[4]
            io_df['so'] = so
            io_df['lo'] = lo
        else:
            print('cal_type格式错误')
        io_df[self.fname] = factor
        io_df = io_df.dropna()
        return io_df
    
    def get_signals(self, io_df, type=0, n_core=10):
        """给出进场出场条件，生成信号
        信号： 当天收盘后触发、次日执行
        oi_df: index: date, columns:[lo, so, lc, sc](多开，空开，多平，空平)
        """
        def fun(x):
            return self.get_one_signal(io_df=x, type=type)
        return parallel_group(io_df, func=fun, n_core=n_core, sort_by='code')

    def get_one_signal(self, io_df, type=0):
        if type == 0:
            if set(io_df.columns) == set(('lo', 'so', self.fname)): # 多空
                io_df['lc'] = io_df['so']
                io_df['sc'] = io_df['lo']
            final_signal = self.compose_io_long_short(df=io_df)
        elif type == 1:
            if set(io_df.columns) == set(('lo', 'so', self.fname)):
                io_df['lc'] = io_df['so']
                io_df['sc'] = io_df['lo']
            final_signal = self.compose_io_long_only(df=io_df)
        elif type == 2:  # 均值回归
            if self.fname not in io_df.columns:
                raise ValueError("io_df must contain 'factor' column for mean reversion signal.")
            final_signal = self.compose_mean_reg(df=io_df)
            
        elif type == 3:  # 概率分布
            if self.fname not in io_df.columns:
                raise ValueError("io_df must contain factor name column for probability distribution signal.")
            final_signal = self.compose_Probdistribution(df=io_df)
        else:
            print('输入止损格式错误，程序终止')
            return None
        final_signal = final_signal[final_signal.index.get_level_values(0) >= pd.to_datetime('2020-01-01')]
        return final_signal
    # 满仓
    def compose_io_long_short(self, df):
        last_signal = 0
        for t in df.index:
            if last_signal == 0:
                if df.loc[t, 'lo']:
                    df.loc[t, 'signal'] = 1
                elif df.loc[t, 'so']:
                    df.loc[t, 'signal'] = -1
                else:
                    df.loc[t, 'signal'] = 0
            elif last_signal == 1:
                if df.loc[t, 'lc']:
                    if df.loc[t, 'so']:
                        df.loc[t, 'signal'] = -1
                    else:
                        df.loc[t, 'signal'] = 0
                else:
                    df.loc[t, 'signal'] = 1
            elif last_signal == -1:
                if df.loc[t, 'sc']:
                    if df.loc[t, 'lo']:
                        df.loc[t, 'signal'] = 1
                    else:
                        df.loc[t, 'signal'] = 0
                else:
                    df.loc[t, 'signal'] = -1
            last_signal = df.loc[t, 'signal']
        return df['signal']
    # 满仓
    def compose_io_long_only(self, df):
        last_signal = 0
        for t in df.index:
            if last_signal == 0:
                if df.loc[t, 'lo']:
                    df.loc[t, 'signal'] = 1
                else:
                    df.loc[t, 'signal'] = 0  
            
            elif last_signal == 1:
                if df.loc[t, 'lc']:
                    df.loc[t, 'signal'] = 0  # 平仓后保持空仓
                else:
                    df.loc[t, 'signal'] = 1  # 继续持有多头
        
            last_signal = df.loc[t, 'signal']
        return df['signal']


    # 均值回归假设, 仅多头
    def compose_mean_reg(self, df):
        """假设指标会经历均值回归, 以该指标超出其近5年均值的百分比作为仓位比例
        """
        df['rolling_5y_mean'] = df[self.fname].rolling(window=252*5).mean()
        df['signal'] = (df[self.fname] - df['rolling_5y_mean']) / df['rolling_5y_mean']
        signal = df['signal']
        pos = pd.Series(np.where(signal >= 0, 1, np.where(signal <= -1, 0, 1+signal)), 
                        index=signal.index, name='position')
        return pos
        
    
    def compose_Probdistribution(self, df):
        """假定指标的值遵从一定的概率分布,以该值在
           近5年值域所处的位置作为仓位比例,考虑指标在历史数据中的分布特征
           r = F / R - 1
           R = max_5y - min_5y
        """
        df['rolling_5y_max'] = df[self.fname].rolling(window=252*5).max()
        df['rolling_5y_min'] = df[self.fname].rolling(window=252*5).min()
        df['R'] = df['rolling_5y_max'] - df['rolling_5y_min']
        df['signal'] = df[self.fname] / df['R'] - 1

        signal = df['signal']
        pos = pd.Series(np.where(signal >= 0, 1, np.where(signal <= -1, 0, 1+signal)), 
                        index=signal.index, name='position')
        return pos


    def fast_post(self, factor, cal_type=0, cal_param=[0.3, 0.7,  '9999d','365d'], n_core=10, signal_type=3, is_full=True):
        if is_full:
            io_df = self.get_oi_df(factor=factor, cal_type=cal_type, cal_param=cal_param)
            signals = self.get_signals(io_df=io_df, type=signal_type, n_core=n_core)
        else:
            signals = self.get_signals(io_df=factor, type=signal_type, n_core=n_core)
        pos = SignalPost(signals=signals, sr=self.sr)
        pos.position_post(compose_type=0, benchmark=self.market)


class SignalPost():
    def __init__(self, signals, sr, comm=2/1e4):
        self.signals = signals #信号
        self.sr = sr  #收益率
        self.comm = comm

    # 根据信号计算持仓df
    # compose_type信号组合方式
    # 0: 默认满仓等权
    def get_position(self, compose_type=0):
        if compose_type == 0:
            postion_df = self.signals.unstack().fillna(0)
            postion_df = postion_df.div(postion_df.abs().sum(axis=1), axis=0)
        else:
            print('输入compose_type错误')
        self.position_df = postion_df

    def position_post(self, compose_type=0, benchmark=None):
        self.get_position(compose_type=compose_type)
        self.turnover = (self.position_df - self.position_df.shift(1)).fillna(0)
        sr_df = self.sr.unstack()*self.position_df - self.turnover*self.comm
        sr_compose = sr_df.loc[self.turnover.index].sum(axis=1)

        self.post = ReturnsPost(returns=sr_compose, benchmark=benchmark)
        self.post.pnl()

