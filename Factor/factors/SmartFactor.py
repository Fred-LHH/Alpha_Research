import os
os.chdir('/Users/lihaohan/Alpha_Research')
import pandas as pd
import numpy as np
from Factor.base import BaseFactor, get_rolling_data
from mpire import WorkerPool
from Data.get_data import read_daily

class SmartFactor(BaseFactor):
    """
    开源(3):聪明钱因子
    聪明钱在交易过程中往往呈现单笔订单数量更大、订单报价更为激进

    Q1切割标准:S = |R| / V^0.5
    Q2切割标准:S = |R| / V^0.1
    Q3切割标准:S = V
    Q4切割标准:S = rank(|R|) + rank(V)
    Q5切割标准:S = |R| / ln(V)
    """

    def __init__(self, 
                 factor_name='SmartFactor', 
                 data_path='/Volumes/T7Shield/ProcessedData', 
                 factor_parameters={'window': 10, 'vol_percent': 0.2}, 
                 save_path='/Volumes/T7Shield/Factor/factor'):
        super(SmartFactor, self).__init__(factor_name=factor_name,
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
                data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d', errors='coerce')
                mask = (data['date'] >= pd.to_datetime(start_date)) & (data['date'] <= pd.to_datetime(end_date))
                data = data[mask]
                if data.empty:
                    return pd.DataFrame()
                data = data[['date', 'code', 'time', 'volume', 'close']].assign(
                    S1=(np.abs(data['close'] / data['close'].shift(1) - 1)) / np.sqrt(data['volume']),
                    S2 = (np.abs(data['close'] / data['close'].shift(1) - 1)) / np.power(data['volume'], 0.1),
                    S3 = data['volume'],
                    S5 = np.abs(data['close'] / data['close'].shift(1) - 1) / np.log(data['volume']),
                    price=(data['close'] + data['high'] + data['low']) / 3)
                data = data.sort_values(by=['date', 'time'])    
                
                days = self.factor_parameters['window']
                rows_per_day = 240

                def Smart_money1(group):
                    group = group.copy()
                    group = group.sort_values(by='S1', ascending=False)
                    group['cumsum_vol'] = group['volume'].cumsum()
                    total_vol = group['volume'].sum()
                    group['cumsum_volratio'] = group['cumsum_vol'] / total_vol
                    smart_money_time = group[group['cumsum_volratio'] <= self.factor_parameters['vol_percent']]
                    vwaps = (smart_money_time['price'] * smart_money_time['volume']).sum() / smart_money_time['volume'].sum()
                    vwap_total = (group['price'] * group['volume']).sum() / group['volume'].sum()
                    Q1 = vwaps / vwap_total
                    return Q1
                
                def Smart_money2(group):
                    group = group.copy()
                    group = group.sort_values(by='S2', ascending=False)
                    group['cumsum_vol'] = group['volume'].cumsum()
                    total_vol = group['volume'].sum()
                    group['cumsum_volratio'] = group['cumsum_vol'] / total_vol
                    smart_money_time = group[group['cumsum_volratio'] <= self.factor_parameters['vol_percent']]
                    vwaps = (smart_money_time['price'] * smart_money_time['volume']).sum() / smart_money_time['volume'].sum()
                    vwap_total = (group['price'] * group['volume']).sum() / group['volume'].sum()
                    Q2 = vwaps / vwap_total
                    return Q2
                
                def Smart_money3(group):
                    group = group.copy()
                    group = group.sort_values(by='S3', ascending=False)
                    group['cumsum_vol'] = group['volume'].cumsum()
                    total_vol = group['volume'].sum()
                    group['cumsum_volratio'] = group['cumsum_vol'] / total_vol
                    smart_money_time = group[group['cumsum_volratio'] <= self.factor_parameters['vol_percent']]
                    vwaps = (smart_money_time['price'] * smart_money_time['volume']).sum() / smart_money_time['volume'].sum()
                    vwap_total = (group['price'] * group['volume']).sum() / group['volume'].sum()
                    Q3 = vwaps / vwap_total
                    return Q3
                
                def Smart_money4(group):
                    group = group.copy()
                    group['S4'] = (np.abs(group['close'] / group['close'].shift(1) - 1)).rank() + group['volume'].rank()
                    group = group.sort_values(by='S4', ascending=False)
                    group['cumsum_vol'] = group['volume'].cumsum()
                    total_vol = group['volume'].sum()
                    group['cumsum_volratio'] = group['cumsum_vol'] / total_vol
                    smart_money_time = group[group['cumsum_volratio'] <= self.factor_parameters['vol_percent']]
                    vwaps = (smart_money_time['price'] * smart_money_time['volume']).sum() / smart_money_time['volume'].sum()
                    vwap_total = (group['price'] * group['volume']).sum() / group['volume'].sum()
                    Q4 = vwaps / vwap_total
                    return Q4
                
                def Smart_money5(group):
                    group = group.copy()
                    group = group.sort_values(by='S5', ascending=False)
                    group['cumsum_vol'] = group['volume'].cumsum()
                    total_vol = group['volume'].sum()
                    group['cumsum_volratio'] = group['cumsum_vol'] / total_vol
                    smart_money_time = group[group['cumsum_volratio'] <= self.factor_parameters['vol_percent']]
                    vwaps = (smart_money_time['price'] * smart_money_time['volume']).sum() / smart_money_time['volume'].sum()
                    vwap_total = (group['price'] * group['volume']).sum() / group['volume'].sum()
                    Q5 = vwaps / vwap_total
                    return Q5
                
                res = []
                for batch in get_rolling_data(data, days, rows_per_day):
                    batch = batch.dropna()
                    if batch.empty:
                        continue
                    date, code = batch['date'].iloc[-1], batch['code'].iloc[-1]
                    Q1 = Smart_money1(batch)
                    Q2 = Smart_money2(batch)
                    Q3 = Smart_money3(batch)
                    Q4 = Smart_money4(batch)
                    Q5 = Smart_money5(batch)
                    res.append({'date': date, 'code': code, 'SmartFactor1': Q1, 'SmartFactor2': Q2, 'SmartFactor3': Q3, 'SmartFactor4': Q4, 'SmartFactor5': Q5})

                factor = pd.DataFrame(res)
                return factor

            except Exception as e:
                print(f"Error processing file {file}: {e}")
                return pd.DataFrame()
            
        with WorkerPool(n_jobs=2) as pool:
            results = pool.map(calculator, files, progress_bar=True)

        self.factor = pd.concat(results).reset_index(drop=True)
        
    
    def run(self, pool: bool = False):
        self.generate_factor()
        if pool:
            self.clear_factor()
        self.save()
        










