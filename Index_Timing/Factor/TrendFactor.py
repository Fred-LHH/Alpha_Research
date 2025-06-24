import pandas as pd
import numpy as np
from Index_Timing.Factor.Base import BaseTimingFactor
import os

class TrendFactor(BaseTimingFactor):
    """捕捉拐点的蛛丝马迹：趋势与加速度共振的择时

    VolFactor中提出的择时模型在牛转熊的捕捉上较快, 但是在上涨趋势的持续跟踪能力上较弱
    本文试图在牛转熊的捕捉与趋势的跟踪追求一个平衡, 通过构建一类模型, 相比于价量共振模型能够提升上涨趋势的持续跟踪能力, 
    又不失能较快地把握牛转熊的拐点.

    趋势与加速度共振系统:
    1. 定义收盘价 close 使用的移动平均线为 EMA(指数平滑移动平均线)。
    2. 定义趋势指标:TrendInd=EMA(close)short/EMA(close)long,收盘价的短期 EMA 除以收盘价的长期 EMA,
    short=10, long=26。
    3. 我们希望构造一个指标能够衡量趋势变化的速度，即定义加速度指标=趋势指标/趋势指标的指数平滑移动平
    均线，即 AcceleratorInd=(TrendInd)/EMA(TrendInd)compare, compare=12。
    4. 定义趋势集:当趋势指标大于 1,即 TrendInd>1 的时候，市场处于趋势市场。
    5. 定义加速度集:当加速度指标大于 1, 即 AcceleratorInd>1 的时候，市场处于加速阶段。
    6, 趋势与加速度共振系统的核心思想在于, 我们既希望能够在上涨的趋势市场下做多, 但是随着运行速度逐渐
    下降的时候, 又希望能够尽早离开这个市场,进而在牛转熊的关键时点上迅速逃离.反过来说,我们希望能
    够在加速运行的趋势市场中持仓,否则空仓.因此定义趋势与加速度共振集合={趋势集∩加速度集},就在
    趋势指标大于 1 且加速度指标大于 1 的情况下做多,否则空仓.
    7. 在《牛市让利,熊市得益,价量共振择时之二:如何规避放量下跌？》一文中, 定义 5 日均线高于 90 日均
    线, 市场划分为多头市场；当 5 日均线小于 90 日均线, 市场划分为空头市场.定义动量指标为 N 天的收益
    率, 趋势较强的下跌市场状态定义为, 10 日的价格效率指标大于 50, 且 10 日的动量指标小于 0。我们希望
    在空头市场规避过于频繁的抄底风险, 因此我们定义空头市场下跌状态为, 5 日均线小于 90 日均线并且 10
    日的动量指标小于 0. 
    8, 将步骤 6 得到的持仓序列排除步骤 7 得到的空头市场下跌状态, 得到了最终的持仓序列。

    改进:
    1. 5日均线高于90日均线, 市场划分为多头市场;当5日均线小于90日均线, 市场划分为空头市场.
    2. 当市场处于多头的时候, 价量共振模型V3或者趋势加速度共振模型做多的时候持有多头仓位, 否则空仓.
    3, 当市场处于空头的时候, 当价量共振模型V3做多的时候持有多头仓位, 否则空仓.
    """
    def __init__(self, **kwargs):
        default_params = {
        'EMA_short': 10,  # 短期EMA周期
        'EMA_long': 26,  # 长期EMA周期
        'compare': 12,  # 加速度指标比较周期
        'WMA_period': 4,  # 平滑收盘价的WMA周期
        }
        
        params = {**default_params, **kwargs.get('factor_parameters', {})}
        super().__init__(
            factor_name='TrendFactors',
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
        return df
    
    def generate_factor(self, start_date, end_date):
        files = self.prepare_data()
        WMA_period = self.factor_parameters['WMA_period']

        def _cal_Trend_ind(data):
            """计算趋势指标
            """
            short_ema = data['close'].ewm(span=self.factor_parameters['EMA_short'], adjust=False).mean()
            long_ema = data['close'].ewm(span=self.factor_parameters['EMA_long'], adjust=False).mean()
            TrendInd = short_ema / long_ema
            AcceleratorInd = TrendInd / TrendInd.ewm(span=self.factor_parameters['compare'], adjust=False).mean()
            long_signal = (TrendInd > 1) & (AcceleratorInd > 1)
            return TrendInd, AcceleratorInd, long_signal
      
        def _cal_short_state(price):
            """计算市场下跌状态 
            """
            weights = np.arange(1, WMA_period+1)
            wma_price = np.convolve(price, weights / weights.sum(), mode='full')[:len(price)]
            wma_price = pd.Series(wma_price, index=price.index)
            wma_price.iloc[:WMA_period-1] = np.nan
            momentum = wma_price - wma_price.shift(10)
            c5_mean = price.rolling(5).mean()
            c90_mean = price.rolling(90).mean()
            short_state = (c5_mean < c90_mean) & (momentum < 0)
            return short_state


        try:
            for file in files:
              df = pd.read_parquet(os.path.join(self.data_path, file))
              df = self.process_ricequant_data(df)

              TrendInd, AcceleratorInd, long_signal = _cal_Trend_ind(df)
              short_state = _cal_short_state(df['close'])
              long_signal = long_signal & ~short_state
          
              df['TrendInd'], df['AcceleratorInd'], df['long'], df['short'] = TrendInd, AcceleratorInd, long_signal, short_state
              self.factor = df
        
              self.save()
        except Exception as e:
          print(f"Error processing file {file}: {e}")
          return pd.DataFrame()



      

    














