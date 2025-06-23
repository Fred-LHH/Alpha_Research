import pandas as pd
import numpy as np
from Data.get_data import *

class BaseTimingFactor(object):
    """
    因子基类，用于因子的计算和存取
    """

    def __init__(self, 
                 factor_name, 
                 factor_parameters, 
                 data_path, 
                 save_path):
        """
        Parameters
        ----------
        factor_name: str
            因子名, 必须唯一
        factor_parameters: dict
            因子计算使用到的自定义参数
        data_path: 数据源地址
        save_path: 数据存储地址
        """

        self.factor_name = factor_name
        self.factor_parameters = factor_parameters
        self.factor = pd.DataFrame()
        # data路径
        if not save_path:
            self.save_path = os.path.abspath('./factor/{}.pickle')
        else:
            self.save_path = save_path + '/factor/{}.pickle'

        if data_path:
            self.data_path = data_path
        else:
            raise ValueError('请输入因子计算的源数据地址')

    def get_factor_name(self):
        """
        获取因子唯一名称
        """
        return self.factor_name

    def prepare_data(self):
        """
        用于获取需要的数据。
        """
        raise NotImplementedError

    def generate_factor(self, start_date, end_date):
        """
        用于计算因子并返回因子的值。
        
        Parameters:
        -----------
        file: 指数的量价数据
        """
        raise NotImplementedError
        
    def _get_trading_days(self, start_date, end_date):
        """
        获取计算因子的交易日历

        Parameters:
        -----------
        start_date: (str)起始时间 '20180103'
        end_date: (str)结束时间 
        """
        return read_trade_dates(start_date=start_date, end_date=end_date)

    def save(self):
        """
        存入数据
        """
        if self.factor is None or len(self.factor) == 0:
            return
        if self.save_path:
            save_dir = os.path.dirname(self.save_path)  
            if not os.path.exists(save_dir):
                os.makedirs(save_dir) 
            
            if isinstance(self.factor, list):
                for factor in self.factor:
                    name = factor.columns[-1]
                    factor.to_pickle(self.save_path.format(name))  
            else:
                self.factor.to_pickle(self.save_path.format(self.factor_name))















