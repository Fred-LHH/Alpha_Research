import pandas as pd
import numpy as np
from .Base import BaseTimingFactor
import os

class VolumeFactorV2(BaseTimingFactor):
    """华创金工 成交量的奥秘：另类价量共振指标的择时 https://mp.weixin.qq.com/s/BcEsr9VU8f3v_K2odpJndA
    周频指数择时

    从普通价量共振系统构建可以看出, 如果在持续下跌的市场遇到了放量的大阴线, 
    当量能远超过价能的时候, 比较容易做出抄底的操作
  
    本类构建的指标:
    OBJ: 量在价先, 认为成交量变化能提前预示价格趋势
      背离分析​:价格创新高但OBV未同步上,暗示趋势可能反转
      突破确认​:OBV突破前高配合价格突破,增强趋势延续信号
    PVI:识别散户资金主导市场, 主力可能反向操作
    NVI:侦测大户行为, 可能蕴含大户暗中吸筹的信息
      趋势判断​:PVI/NVI位于其移动平均线上方为多头市场,反之为空头
      组合信号​:PVI与NVI同步上穿均线,预示“大多头行情”
    AMA: 适应性移动平均线
      一种动态调整权重的移动平均线，根据市场波动性自动优化平滑系数，更适合捕捉成交量或价格的短期异动
      量能计算​:AMA5(5日适应性均线)与 AMALong(长期均线)比值,反映短期成交量活跃度
    BMA: 收盘价移动平均线
      普通移动平均线(如SMA或EMA)应用于收盘价序列,用于平滑价格波动并识别趋势方向
      价能计算​:当日BMA除以前N日 BMA, 衡量价格趋势强度

    研报给出的信号如下:
    1. 当5日均线高于90日均线, 市场划分为多头市场;当5日均线小于90日均线, 市场划分为空头市场。
    2. 当前为多头市场下, 若价量共振指标大于Threshold1则做多, 否则以Threshold1平仓.当前为空头市场下, 若价量共振指标大于Threshold2则做多, 否则以 Threshold2 平仓。
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
          'Threshold2': 1.275
          }
        
        params = {**default_params, **kwargs.get('factor_parameters', {})}
        super().__init__(
          factor_name='VolumeFactorsV2',
          factor_parameters=params,
          data_path=kwargs.get('data_path'),
          save_path=kwargs.get('save_path')
          )
    
    def prepare_data(self):
        """返回多个指数的日频量价数据文件
        """
        daily_files = os.listdir(self.data_path)
        daily_files = [f for f in daily_files if f.endswith('.parquet') and not f.startswith('._')]

        return daily_files
    def process_ricequant_data(self, df):
        df = df.copy()
        df = df.reset_index()
        df.rename(columns={'order_book_id': 'code', 'volume': 'vol', 'prev_close': 'pre_close'}, inplace=True)
        df['pct_change'] = df['close'].pct_change() 
        return df
    
    def generate_factor(self, start_date, end_date):
        files = self.prepare_data()
        fast_sc = 2 / (self.factor_parameters['fast_constant'] + 1)
        slow_sc = 2 / (self.factor_parameters['slow_constant'] + 1)

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
            vol_decrease = data['vol'] < data['vol'].shift(1)
    
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

        try:
            for file in files:
                df = pd.read_parquet(os.path.join(self.data_path, file))
                df = self.process_ricequant_data(df)
      
                df['OBV_VA'] = _cal_OBV(df)
                df['PVI'] = _cal_pvi(df)
                df['NVI'] = _cal_nvi(df)
        
                # 计算AMA（成交量适应性移动平均）
                df['AMA_vol'] = _calculate_ama(df['vol'], self.factor_parameters['er_period'])
        
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



      

    









