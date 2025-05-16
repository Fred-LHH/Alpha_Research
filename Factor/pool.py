import pandas as pd
import numpy as np
import os
os.chdir('/Users/lihaohan/Alpha_Research')
from Data.get_data import *
from tqdm import tqdm
from utils import boom_one
from Data.tushare.api import *

api = TSDataFetcher()


class PoolFactor:
    def __init__(self, 
                 factor: pd.DataFrame,
                 factor_name: str):
        """因子过滤器: 过滤st、涨跌停、停牌、次新股

        Parameters
        ----------
        factor: pd.DataFrame
            index为日期, columns为股票;
        factor_name: str
            因子名称
        """
        self.factor = factor.stack().reset_index()
        self.factor.columns = ['date', 'code', factor_name]
        self.factor['date'] = pd.to_datetime(self.factor['date']).dt.strftime('%Y-%m-%d')
        self.name = factor_name

    def remove_newstock(self, periods=365):
        stocks_list = api.all_stocks()
        stocks_list['list_date'] = pd.to_datetime(stocks_list['list_date'], format='%Y-%m-%d')
        current_date = self.factor['date'].max()
        remove_date = current_date - pd.Timedelta(days=365)
        new_codes = stocks_list[stocks_list['list_date'] > remove_date]['symbol'].tolist()
        self.factor = self.factor[~self.factor['code'].isin(new_codes)]


    def remove_st(self):
        st = read_filter_con(st=1)
        self.factor = pd.merge(self.factor, st, on=['date', 'code'], how='left')
        self.factor['tmp_factor'] = np.where(self.factor['st'] == 0, np.nan, self.factor[self.name])
        self.factor.drop(columns=self.name, inplace=True)

    def remove_suspend(self):
        suspend = read_filter_con(suspend=1)
        self.factor = pd.merge(self.factor, suspend, on=['date', 'code'], how='left')
        self.factor['tmp_factor'] = np.where(self.factor['suspend'] == 0, np.nan, self.factor[self.name])
        self.factor.drop(columns=self.name, inplace=True)

    def remove_limit(self):
        limit = read_filter_con(limit=1)
        self.factor = pd.merge(self.factor, limit, on=['date', 'code'], how='left')
        self.factor['tmp_factor'] = np.where(self.factor['limit_status'] == 0, np.nan, self.factor[self.name])
        self.factor.drop(columns=self.name, inplace=True)


    def smooth_factor(self, 
                      backsee: int=20,
                      daily: bool=1,
                      min_periods: int=None):
        
        self.factor = self.factor.pivot(index='date', columns='code', values=self.name)
        return boom_one(self.factor,
                        backsee=backsee,
                        daily=daily,
                        min_periods=min_periods)
    
    def run(self):
        self.remove_newstock()
        self.remove_st()
        self.remove_suspend()
        self.remove_limit()
        self.smooth_factor()


