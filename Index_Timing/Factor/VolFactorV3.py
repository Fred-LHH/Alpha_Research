import pandas as pd
import numpy as np
from .Base import BaseTimingFactor
import os


class VolumeFactorV3(BaseTimingFactor):
    """牛市让利, 熊市得益, 价量共振择时之二:如何规避放量下跌？
    
    本文构造了价格效率指标和动量指标
    价格效率指标取值范围为0~100, 当价格效率指标较大(比如大于 50), 且市场动量小于0, 市场有可能处于趋势
    较强的下跌状态, 价量共振择时V3希望能在V2的信号中过滤掉这个状态

    研报给出的信号如下:
    1. 成交量使用的移动平均线为 AMA, 收盘价使用的移动平均线为 BMA(移动平均线长度为 50).
    2. 价能=BMA(Today)/BMA(Today-3), 当日的 BMA 除以前 3 个交易日的 BMA.
    3. 量能=AMA5/AMA100, 5 日的 AMA 除以 100 日的 AMA.
    4. 价量共振指标=价能×量能, 价能乘以量能.
    5. 当 5 日均线高于 90 日均线, 市场划分为多头市场;当 5 日均线小于 90 日均线, 市场划分为空头市场.
    6. 当前为多头市场下, 若价量共振指标大于 1.125 则做多, 否则以 1.125 平仓.当前为空头市场下, 若价量共
    振指标大于 1.275 则做多, 否则以 1.275 平仓, 最终计算得到资产的持仓序列.
    7. 将收盘价简单平滑，这里使用 4 日的加权移动平均线进行平滑(John F. Ehlers 在构造技术指标的过程中, 经
    常使用 4 日的 WMA 构造 Smooth, 因为 WMA 的滞后阶数为(N-1)/3, 4 日的 WMA 滞后阶数为 1 日), 再计算平滑后收盘价 10 日的价格效率指标和 10 日的动量指标.那么趋势较强的下跌市场状态定义为, 10 日的价格效率指标大于 50, 且 10 日的动量指标小于 0.
    8. 将步骤 6 得到的持仓序列排除步骤 7 得到的下跌市场状态, 得到了最终的持仓序列.
    """
    def __init__(self, **kwargs):
        default_params = {
            'L': 50,     # BMA的移动平均周期
            'N': 3,      # 价能计算的时间窗口
            'ama_short': 5,    # 短期AMA周期
            'ama_long': 100,    # 长期AMA周期
            'er_period': 10,   # AMA效率比率计算周期
            'fast_constant': 2,   # AMA快速平滑常数（对应于短期EMA）
            'slow_constant': 30,   # AMA慢速平滑常数（对应于长期EMA）
            'Threshold1': 1.125,
            'Threshold2': 1.275,
            'WMA_period': 4,  # 平滑收盘价的WMA周期
            }
        
        params = {**default_params, **kwargs.get('factor_parameters', {})}
        super().__init__(
            factor_name='VolumeFactorsV3',
            factor_parameters=params,
            data_path=kwargs.get('data_path'),
            save_path=kwargs.get('save_path')
            )
    
    def prepare_data(self):
      """返回多个指数的日频量价数据文件
      """
      daily_files = os.listdir(self.data_path)
      daily_files = [f for f in daily_files if f.endswith('.pkl') and not f.startswith('._')]

      return daily_files
    
    def generate_factor(self, start_date, end_date):
      files = self.prepare_data()
      fast_sc = 2 / (self.factor_parameters['fast_constant'] + 1)
      slow_sc = 2 / (self.factor_parameters['slow_constant'] + 1)
      WMA_period = self.factor_parameters['WMA_period']

      def _cal_OBV(data):
        """计算OBV的改进版VA
        """
        hc_diff = data['high'] - data['close']
        hc_diff = hc_diff.replace(0, np.nan) 
        va = data['vol'] * (
          (data['close'] - data['low']) -
          (data['high'] - data['close'])
        ) / hc_diff
        va_adjust = np.where(
            data['close'] > data['pre_close'], va,
            np.where(data['close'] < data['pre_close'], -va, 0)
        )
        # 累计OBV
        return va_adjust.cumsum()
      
      def _cal_pvi(data):
        """计算正成交量指标
        """
        pvi = pd.Series(1000, index=data.index, dtype=float)
        vol_increase = data['vol'] > data['vol'].shift(1)
        for i in range(1, len(pvi)):
            if vol_increase.iloc[i]:
                pvi.iloc[i] = pvi.iloc[i-1] * (1 + data['pct_change'].iloc[i])
        return pvi
      
      def _cal_nvi(data):
        """计算负成交量指标
        """
        nvi = pd.Series(1000, index=data.index, dtype=float)
        vol_decrease = data['volume'] < data['volume'].shift(1)
    
        for i in range(1, len(nvi)):
            if vol_decrease.iloc[i]:
                nvi.iloc[i] = nvi.iloc[i-1] * (1 + data['pct_change'].iloc[i])
        return nvi
      
      def _calculate_ama(ser, period):
        """计算适应性移动平均线(AMA)
        """
        direction = ser - ser.shift(period)
        
        # 波动性计算
        volatility = ser.diff().abs().rolling(period).sum()
        volatility = volatility.replace(0, np.nan)  # 避免除零错误
        
        # 效率比率
        er = (direction / volatility).abs()
        
        # 平滑系数计算
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # AMA计算
        ama = pd.Series(0.0, index=ser.index)
        ama.iloc[:period] = ser.iloc[:period].mean()  
        
        # 递归计算AMA
        for i in range(period, len(ser)):
            ama.iloc[i] = ama.iloc[i-1] + sc.iloc[i] * (ser.iloc[i] - ama.iloc[i-1])
        
        return ama
      
      def _filter_down_market(price):
        """过滤掉大跌的市场状态
        """
        weights = np.arange(1, WMA_period+1)
        wma_price = np.comvolve(price, weights / weights.sum(), mode='full')[:len(price)]
        wma_price = pd.Series(wma_price, index=price.index)
        wma_price.iloc[:WMA_period-1] = np.nan
        direction = (wma_price - wma_price.shift(10)).abs()
        volatility = wma_price.diff().abs().rolling(10).sum()
        volatility = volatility.replace(0, np.nan)
        efficiency = (direction / volatility) * 100
        momentum = wma_price - wma_price.shift(10)
        strong_downtrend = (efficiency > 50) & (momentum < 0)
        market_status = pd.Series(1, index=price.index)
        market_status[strong_downtrend] = 0
        market_status.iloc[:10] = 1
    
        return market_status
    
      try:
        for file in files:
          df = pd.read_pickle(os.path.join(self.data_path, file))
          df = self.data.copy()
      
          df['OBV_VA'] = self._calculate_obv()
          df['PVI'] = self._calculate_pvi()
          df['NVI'] = self._calculate_nvi()
        
          # 计算AMA（成交量适应性移动平均）
          df['AMA_vol'] = self._calculate_ama(df['volume'], self.factor_parameters['er_period'])
        
          # 计算BMA（价格移动平均）
          L = self.factor_parameters['L']
          df['BMA'] = df['close'].rolling(L).mean()
        
          # 计算量能 (AMA5/AMALong)
          df['AMA5'] = df['AMA_vol'].rolling(self.factor_parameters['ama_short']).mean()
          df['AMALong'] = df['AMA_vol'].rolling(self.factor_parameters['ama_long']).mean()
          df['volume_energy'] = df['AMA5'] / df['AMALong']
        
          # 计算价能 (BMA_t / BMA_{t-N})
          N = self.factor_parameters['N']
          df['price_energy'] = df['BMA'] / df['BMA'].shift(N)
        
          # 计算价量共振因子
          df['resonance'] = df['price_energy'] * df['volume_energy']
        
          self.factor = df
        
          self.save()
      except Exception as e:
        print(f"Error processing file {file}: {e}")
        return pd.DataFrame()



      

    









