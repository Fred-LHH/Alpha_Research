import os
os.chdir('/Users/lihaohan/Alpha_Research')
import pandas as pd
import numpy as np
from Factor.base import BaseFactor
from mpire import WorkerPool
from Data.get_data import read_daily

class Reverse_M(BaseFactor):
    """
    开源(1):反转因子的W式切割
    将过去20日的涨跌幅分解出动量(M_low)和反转(M_high)
    理想反转因子的交易行为逻辑是A股反转之力的微观来源是大单成交
    """

    def __init__(self, 
                 factor_name='Reverse_M', 
                 data_path='/Volumes/T7Shield/ProcessedData', 
                 factor_parameters={'quantile': 15 / 16}, 
                 save_path='/Volumes/T7Shield/Alpha_Research/Factor'):
        super(Reverse_M, self).__init__(factor_name=factor_name,
                                 factor_parameters=factor_parameters,
                                 data_path=data_path, save_path=save_path)
        
    
    def prepare_data(self):
        min_files = os.listdir(self.data_path)
        min_files = [f for f in min_files if f.endswith('.pkl') and not f.startswith('._')]

        return min_files
    
    def generate_factor(self, start_date, end_date):
        files = self.prepare_data()
        close = read_daily(close=1, start_date=start_date, end_date=end_date, freq='D')
        close = close.astype(np.float32)
        ret = close / close.shift(1) - 1
        ret.index = pd.to_datetime(ret.index, format='%Y-%m-%d', errors='coerce')
        ret = ret.stack().reset_index()
        ret.columns = ['date', 'code', 'ret']
        def calculator(file):
            try:
                code = file.split('.')[0]
                data = pd.read_pickle(os.path.join(self.data_path, file))
                data = data[['date', 'code']].assign(mean_amount=data['amount'] / data['volume'])
                data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d', errors='coerce')
                mask = (data['date'] >= pd.to_datetime(start_date)) & (data['date'] <= pd.to_datetime(end_date))
                data = data[mask]
                if data.empty:
                    return pd.DataFrame()
                
                def M(group):
                    if group['mean_amount'].dropna().empty:
                        return np.nan
                    return np.quantile(group['mean_amount'].dropna(), 15 / 16)
                
                quantiles = data.groupby(['date', 'code']).apply(M).reset_index()
                quantiles.columns = ['date', 'code', 'M']
                merged = pd.merge(quantiles, ret, on=['date', 'code'], how='left')
                res = []
                for i in range(19, len(merged)):
                    rolling_window = merged.iloc[i-19:i+1]
                    high_quantile = rolling_window.nlargest(10, 'M')
                    low_quantile = rolling_window.nsmallest(10, 'M')

                    M_high = high_quantile['ret'].sum()
                    M_low = low_quantile['ret'].sum()
                    M = M_high - M_low
                    res.append({'date': rolling_window['date'].iloc[-1], 'code': code, 'Reverse_M': M, 'M_high': M_high, 'M_low': M_low})

                factor = pd.DataFrame(res)
                return factor
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                return pd.DataFrame()
            
        with WorkerPool(n_jobs=4) as pool:
            results = pool.map(calculator, files, progress_bar=True)

        self.factor = pd.concat(results).reset_index(drop=True)
        
    
    def run(self, pool: bool = False):
        self.generate_factor()
        if pool:
            self.clear_factor()
        self.save()
        










