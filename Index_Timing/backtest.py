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
        df['rolling_5y_mean'] = df[self.fname].rolling(window=252*1).mean()
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
        df['rolling_5y_max'] = df[self.fname].rolling(window=252*1).max()
        df['rolling_5y_min'] = df[self.fname].rolling(window=252*1).min()
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





class Bt:
    def __init__(self, 
                 index_data: pd.DataFrame):
        """
        Args:
        index_data: pd.DataFrame
            指数数据 columns: ['date', 'code', 'open', 'close']
        """
        self.index_data = self.process_index(index_data)

    def process_index(self, df):
        df.reset_index(inplace=True)
        df.rename(columns={'order_book_id': 'code'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
    
    def cal_mkt_effect(self, window: int=20, 
                       pos_range: float=0.5,
                       neg_range: float=-0.3,
                       pos_num: int=50,
                       neg_num: int=50):
        """计算市场赚/亏钱效应

        Args:
        window: int
            计算窗口大小
        pos_range: float
            累计涨幅阈值
        neg_range: float
            累计跌幅阈值
        pos_num: int
            赚钱效应股票数阈值
        neg_num: int
            亏钱效应股票数阈值
        """
        rets = self.stock_data.pct_change(window)
    
        pos_mask = rets > pos_range
        neg_mask = rets < neg_range
        self.actual_pos_num = pos_mask.sum(axis=1)
        self.actual_neg_num = neg_mask.sum(axis=1)

        self.market_effect = pd.Series(
            np.where(self.actual_pos_num > pos_num, 1,
                    np.where(self.actual_neg_num > neg_num, -1, 0)),
            index=rets.index,
            name='market_effect'
        )

    def generate_signals(self):
        """生成买入卖出信号
        """
        con = self.index_data.join(self.market_effect)
        con['pos_signal'] = (con['close'] > con['ma20']) & (con['market_effect'] == 1)
        con['neg_signal'] = con['market_effect'] == -1
        con['signal'] = 0

        for date in con.index:
            is_pos_signal = con.loc[date, 'pos_signal']
            is_neg_signal = con.loc[date, 'neg_signal']
            
            if is_pos_signal and is_neg_signal:
                con.loc[date, 'signal'] = -1
                
            elif is_neg_signal:
                con.loc[date, 'signal'] = -1
            elif is_pos_signal:
                con.loc[date, 'signal'] = 1
        
        self.signal = con

    def backtest(self, initial_cap=1e6, is_plot=True):
        df = self.signal.copy()
        df['position'] = 0.0 # 持仓数量
        df['cash'] = initial_cap # 初始现金
        df['total'] = initial_cap # 总资产
        df['entry'] = 0
        df['exit'] = 0
        cash = initial_cap
        position = 0.0


        for i, date in enumerate(df.index):
            row = df.loc[date]
        
            prev_signal = df.iloc[i-1]['signal'] if i > 0 else 0
            if prev_signal == 1 and position == 0:
                open_price = row['open']
                if open_price > 0:
                    shares = int(cash // open_price)
                    if shares > 0:
                        cost = shares * open_price
                        position = shares
                        cash -= cost
                        df.at[date, 'entry'] = 1
            elif prev_signal == -1 and position > 0:
                open_price = row['open']
                sale = position * open_price
                cash += sale
                position = 0
                df.at[date, 'exit'] = 1
        
        
            df.at[date, 'position'] = position
            df.at[date, 'cash'] = cash
            df.at[date, 'total'] = cash + (position * row['close'])

        if is_plot:
            neg_num = self.actual_neg_num
            pos_num = self.actual_pos_num
            num = pd.concat([neg_num, pos_num], axis=1)
            num.columns = ['neg_num', 'pos_num']
            num.dropna(inplace=True)

            fig, ax1 = plt.subplots(figsize=(12, 8))
            ax1.bar(num.index, num['neg_num'], label='neg_num', color='blue')
            ax1.bar(num.index, num['pos_num'], label='pos_num', color='green')
            ax1.set_ylabel('number', fontsize=12)
    
            ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='y=50')  
            ax1.axhline(y=100, color='black', linestyle='--', linewidth=1, label='y=100')  
            ax1.legend(loc='upper left')

            ax2 = ax1.twinx()
            ax2.plot(df.index, df['total'] / initial_cap, label='Strategy Net Value', color='red', linestyle='-')
            ax2.plot(df.index, df['close'] / df['close'].iloc[0], label='Benchmark ', color='grey', linestyle='-')
            buy_signals = df[df['entry'] == 1]
            sell_signals = df[df['exit'] == 1]
            if not buy_signals.empty:
                ax2.scatter(buy_signals.index, buy_signals['open'] / df['close'].iloc[0], color='magenta', marker='^', s=100, label='Buy Entry', zorder=5)
    
            if not sell_signals.empty:
                ax2.scatter(sell_signals.index, sell_signals['open'] / df['close'].iloc[0], color='cyan', marker='v', s=100, label='Sell Exit', zorder=5)
                    
            ax2.set_ylabel('Value / Price', fontsize=12)
            ax2.legend(loc='upper right')
            
            plt.title('ZZ500 Strategy', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.grid(alpha=0.3)
    
            plt.show()
    
        
        entry_dates = df[df['entry'] == 1].index
        exit_dates = df[df['exit'] == 1].index
    
        holding_returns_list = []
    
        if not entry_dates.empty:
            current_entry_idx = 0
            while current_entry_idx < len(entry_dates):
                entry_date = entry_dates[current_entry_idx]
            
                exits_after_entry = exit_dates[exit_dates > entry_date]
            
                if not exits_after_entry.empty:
                    exit_date = exits_after_entry[0]
                
                    entry_p = df.loc[entry_date, 'open']
                    exit_p = df.loc[exit_date, 'open']
                
                    trade_return = (exit_p - entry_p) / entry_p 
                    holding_returns_list.append(trade_return)
                
                    next_entry_candidates = entry_dates[entry_dates > exit_date]
                    if not next_entry_candidates.empty:
                        current_entry_idx = entry_dates.get_loc(next_entry_candidates[0])
                    else:
                        break 
                else:
                    # 如果有入场但没有对应的出场 (可能持有到最后)
                    entry_p = df.loc[entry_date, 'open']
                    exit_p = df['close'].iloc[-1] 
                    trade_return = (exit_p - entry_p) / entry_p
                    holding_returns_list.append(trade_return)
                    break 

        holding_returns = np.array(holding_returns_list)

        if len(holding_returns) > 0:
            pos_returns = holding_returns[holding_returns > 0]
            neg_returns = holding_returns[holding_returns < 0]
            profit_loss_ratio = np.mean(pos_returns) / abs(np.mean(neg_returns)) if len(neg_returns) > 0 else np.nan
            win_rate = len(pos_returns) / len(holding_returns)
            mean_holding_return = np.mean(holding_returns)
        else:
            profit_loss_ratio = np.nan
            win_rate = np.nan
            mean_holding_return = np.nan

 
        df['daily_ret'] = df['total'].pct_change().fillna(0)
        total_return = df['total'].iloc[-1] / initial_cap - 1

        num_years = (df.index[-1] - df.index[0]).days / 365
        ret_yearly = total_return / num_years

        cumulative = df['total']
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
    
    
        sharpe_ratio = (df['daily_ret'].mean() / df['daily_ret'].std()) * np.sqrt(252) if df['daily_ret'].std() != 0 else 0

        perf = pd.Series([total_return, ret_yearly, sharpe_ratio, max_drawdown, win_rate, profit_loss_ratio, mean_holding_return, len(holding_returns)],
                     index=[' 总收益率', '年化收益率', '夏普比率', '最大回撤', '胜率', '盈亏比', '单笔交易的平均收益', '总交易次数'])

        return df, perf

    def optimize_params(self, p_range: List[int],
                        n_range: List[int]):

        """
        通过网格搜索优化赚钱效应和亏钱效应的股票数量阈值。

        Args:
        p_range: List[int]
            赚钱效应股票数量阈值的列表，例如 [30, 50, 70]
        n_range: List[int]
            亏钱效应股票数量阈值的列表，例如 [30, 50, 70]

        这里采用总收益率比作为优化目标。
        """
        from tqdm import tqdm
        import warnings
        warnings.filterwarnings("ignore")
        results = []
        best_ret = -np.inf
        best_params = {}

        for pos_num_val in tqdm(p_range):
            for neg_num_val in n_range:
                
                self.cal_mkt_effect(pos_num=pos_num_val, neg_num=neg_num_val)
                self.generate_signals()
                _, perf = self.backtest(is_plot=False)  

                current_ret = perf[' 总收益率']
                if pd.isna(current_ret): 
                    current_ret = -np.inf

                results.append({
                    'pos_num': pos_num_val,
                    'neg_num': neg_num_val,
                    **perf.to_dict() 
                })

                if current_ret > best_ret:
                    best_ret = current_ret
                    best_params = {'pos_num': pos_num_val, 'neg_num': neg_num_val}
        
        results_df = pd.DataFrame(results)
        print(f"\nBest parameters: {best_params} with 总收益率: {best_ret:.4f}")
        
        return results_df, best_params








