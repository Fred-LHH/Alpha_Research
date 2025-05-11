import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

from functools import reduce, lru_cache
from loguru import logger
from plotly.tools import FigureFactory as FF
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Data.get_data import read_daily, read_market
from decorators import do_on_dfs
from Data.utils import *
from config import *
from utils import *

db = DButils()

class frequency_controller(object):
    def __init__(self, freq: str):
        offsets = {
        'W': pd.DateOffset(weeks=1),
        'M': pd.DateOffset(months=1),
        'D': pd.DateOffset(days=1)
        }
        frequency = {
        'W': 52,
        'M': 12,
        'D': 252
        }
        freq_name = {
        'W': '周',
        'M': '月',
        'D': '日'
        }
        days_in_freq = {
        'W': 5,
        'M': 22,
        'D': 1
        }
        self.freq = freq
        self.counts_one_year = frequency[freq]
        self.time_shift = offsets[freq]
        self.bt_name = freq_name[freq]
        self.days_in = days_in_freq[freq]


class Alpha(object):

    @classmethod
    @lru_cache(maxsize=None)
    def __init__(
        cls,
        freq: str='W',
    ):
        cls.freq = freq
        cls.freq_controller = frequency_controller(freq)

    @property
    def factors_out(self):
        return self.__factors_out

    def __call__(self):
        """调用对象则返回因子值"""
        return self.factors_out

    @classmethod
    @lru_cache(maxsize=None)
    def set_basic_data(
        cls,
    ):
        open = read_daily(open=1, freq=cls.freq)
        #close = read_daily(close=1, freq=cls.freq)
        zz500_open = read_market(open=1, index='000802', freq=cls.freq)

        cls.open = open.apply(pd.to_numeric, errors='coerce')
        #cls.close = close.apply(pd.to_numeric, errors='coerce')
        zz500_open = zz500_open.apply(pd.to_numeric, errors='coerce')
        #cls.index_open = zz500_open
        cls.market_ret = zz500_open / zz500_open.shift(1) - 1
        cls.open = cls.open.replace(0, np.nan)
        #cls.close = cls.close.replace(0, np.nan)

    def set_factor_date_as_index(self, df: pd.DataFrame):
        """
        index为时间, columns为股票代码
        """
        '''
        db.cursor.execute('SELECT * FROM st')
        st = pd.DataFrame(db.cursor.fetchall(), columns=[i[0] for i in db.cursor.description])
        st.replace(0, np.nan, inplace=True)
        db.cursor.execute('SELECT * FROM limit_status')
        limit_status = pd.DataFrame(db.cursor.fetchall(), columns=[i[0] for i in db.cursor.description])
        limit_status.replace(0, np.nan, inplace=True)
        db.cursor.execute('SELECT * FROM suspend')
        suspend = pd.DataFrame(db.cursor.fetchall(), columns=[i[0] for i in db.cursor.description])
        suspend.replace(0, np.nan, inplace=True)
        '''
        if self.freq is not 'D':
            #st = st.resample(self.freq).last()
            #limit_status = limit_status.resample(self.freq).last()
            #suspend = suspend.resample(self.freq).last()
            self.factors = df.resample(self.freq).last().dropna(how='all')
        else:
            self.factors = df.dropna(how='all')
        #self.factors = (self.factors * suspend).dropna(how='all')
        #self.factors = (self.factors * st).dropna(how='all')
        #self.factors = (self.factors * limit_status).dropna(how='all')
        self.factor_cover = self.factors.count().sum()
        total = self.open.reindex(self.factors.index).count().sum()
        self.factor_cover = min(self.factor_cover / total, 1)
        self.factor_cross_skew = self.factors.skew(axis=1).mean()
        pos_num = ((self.factors > 0) + 0).sum().sum()
        neg_num = ((self.factors < 0) + 0).sum().sum()
        self.pos_neg_rate = pos_num / (pos_num + neg_num)
        self.corr_itself = show_corr(self.factors, self.factors.shift(1), plt_plot=0)

    @classmethod
    @lru_cache(maxsize=None)
    def get_rets(cls):
        cls.rets = cls.open / cls.open.shift(1) - 1
        cls.rets = cls.rets.stack().reset_index()
        cls.rets.columns = ['date', 'code', 'ret']

    def get_ic_rankic(cls, df):
        """
        计算IC和RankIC
        """
        df1 = df[['ret', 'fac']]
        rankic = df1.rank().corr().iloc[0, 1]
        small_ic = df1[df1.fac <= df1.fac.median()].rank().corr().iloc[0, 1]
        big_ic = df1[df1.fac >= df1.fac.median()].rank().corr().iloc[0, 1]
        df2 = pd.DataFrame({
            'rankic': [rankic],
            'small_rankic': [small_ic],
            'big_rankic': [big_ic]
        })
        return df2

    def get_icir_rankicir(cls, df):
        """
        计算ICIR和RankICIR
        """
        rankic = df.rankic.mean()
        small_rankic = df.small_rankic.mean()
        big_rankic = df.big_rankic.mean()
        
        return pd.DataFrame({'IC': [rankic]}, index=['评价指标']), pd.DataFrame({'1-5IC': [small_rankic], '6-10IC': [big_rankic]}, index=['评价指标']).T


    def get_ic_icir_and_rank(cls, df):
        df1 = df.groupby('date').apply(cls.get_ic_rankic)
        cls.rankics = df1.rankic
        cls.rankics = cls.rankics.reset_index(drop=True, level=1).to_frame()
        cls.small_rankics = df1.small_rankic.reset_index(drop=True, level=1).to_frame()
        cls.big_rankics = df1.big_rankic.reset_index(drop=True, level=1).to_frame()
        df2, df5 = cls.get_icir_rankicir(df1)  
        df2 = df2.T
        return df2, df5

        
    @classmethod
    def get_groups(cls, df, groups_num):
        """
        根据因子值, 判断在第几组
        """
        if 'group' in list(df.columns):
            df = df.drop(columns=['group'])
        df = df.sort_values(['fac'], ascending=True)
        each_group = round(df.shape[0] / groups_num)
        l = list(
            map(
                lambda x, y: [x] * y,
                list(range(1, groups_num + 1)),
                [each_group] * groups_num,
            )
        )
        l = reduce(lambda x, y: x + y, l)
        if len(l) < df.shape[0]:
            l = l + [groups_num] * (df.shape[0] - len(l))
        l = l[:df.shape[0]]
        df.insert(0, 'group', l)
        return df

    def get_data(self, groups_num):
        """
        拼接因子数据和收益率数据
        """
        self.data = pd.merge(self.rets, self.factors, how='inner', on=['date', 'code'])
        self.ic_icir_and_rank, self.big_small_rankic = self.get_ic_icir_and_rank(self.data)
        self.data = self.data.groupby('date').apply(
            lambda x: self.get_groups(x, groups_num)
        )
        self.data = self.data.reset_index(drop=True)

    def make_start_to_one(self, l):
        """
        让净值序列的第一个数变成1
        """
        min_date = self.factors.date.min()
        add_date = min_date - self.freq_controller.time_shift
        add_l = pd.Series([0], index=[add_date])
        l = pd.concat([add_l, l])
        return l

    def get_group_rets_net_values(self, groups_num=10):
        """
        计算组内每一期的平均收益, 生成每日收益率序列和净值序列
        """

        self.group_rets = self.data.groupby(['date', 'group']).apply(
            lambda x: x.ret.mean()
        )
        self.rets_all = self.data.groupby(['date']).apply(lambda x: x.ret.mean())
        self.group_rets = self.group_rets.unstack()
        self.group_rets = self.group_rets[
            self.group_rets.index <= self.factors.date.max()
        ]
        self.group_rets.columns = list(map(str, list(self.group_rets.columns)))
        self.group_rets = self.group_rets.add_prefix('group')
        self.rets_all = self.rets_all.dropna()
        self.long_short_rets = (
            self.group_rets['group1'] - self.group_rets['group' + str(groups_num)]
        )
        self.inner_rets_long = self.group_rets.group1 - self.rets_all
        self.inner_rets_short = (
            self.rets_all - self.group_rets['group' + str(groups_num)]
        )
        self.long_short_net_values = self.make_start_to_one(
            self.long_short_rets.cumsum()
        )
        self.market = self.market_ret.reindex(self.group_rets.index)
        self.long_minus_market_rets = self.group_rets.group1 - self.market_ret['000802']

        if self.long_short_net_values[-1] <= self.long_short_net_values[0]:
            self.long_short_rets = (
                self.group_rets['group' + str(groups_num)] - self.group_rets['group1']
            )
            self.long_short_net_values = self.make_start_to_one(
                self.long_short_rets.cumsum()
            )
            self.inner_rets_long = (
                self.group_rets['group' + str(groups_num)] - self.rets_all
            )
            self.inner_rets_short = (
                self.rets_all - self.group_rets['group1']
            )
            self.long_minus_market_rets = self.group_rets['group' + str(groups_num)] - self.market_ret['000802']

        self.long_minus_market_nets = self.make_start_to_one(self.long_minus_market_rets.dropna().cumsum())
        self.inner_long_net_values = self.make_start_to_one(
            self.inner_rets_long.cumsum()
        )
        self.inner_short_net_values = self.make_start_to_one(
            self.inner_rets_short.cumsum()
        )  
        self.group_rets = self.group_rets.assign(long_short=self.long_short_rets)
        self.group_net_values = self.group_rets.cumsum()
        self.group_net_values = self.group_net_values.apply(self.make_start_to_one)

    def get_long_short_backtest(self):
        """
        计算多空对冲的回测指标
        """
        self.long_short_ret_yearly = self.long_short_net_values[-1] * (
            self.freq_controller.counts_one_year / len(self.long_short_net_values)
        )
        self.inner_long_ret_yearly = self.inner_long_net_values[-1] * (
            self.freq_controller.counts_one_year / len(self.inner_long_net_values)
        )
        self.inner_short_ret_yearly = self.inner_short_net_values[-1] * (
            self.freq_controller.counts_one_year / len(self.inner_short_net_values)
        )

        self.long_short_vol_yearly = np.std(self.long_short_rets) * (
            self.freq_controller.counts_one_year ** 0.5
        )
        self.long_short_info_ratio = (
            self.long_short_ret_yearly / self.long_short_vol_yearly
        )
        self.long_short_comments = pd.DataFrame(
            {
                '评价指标': [
                    self.long_short_ret_yearly,
                    self.long_short_vol_yearly,
                    self.long_short_info_ratio,
                ]
            },
            index=['年化收益', '年化波动', '信息比率'],
        )

    def get_total_comments(self):

        self.group_mean_rets = self.group_rets.drop(columns=['long_short']).mean()
        mar = self.market_ret.reindex(self.factors_out.index)
        self.group_mean_rets = (
            self.group_mean_rets - mar.mean().values
        ) * self.freq_controller.counts_one_year
        self.group1_ret_yearly = self.group_mean_rets.loc['group1']
        self.group10_ret_yearly = self.group_mean_rets.loc['group10']
        if self.group1_ret_yearly > self.group10_ret_yearly:
            self.longside_ret = self.group_rets.group1  #- mar.values
            self.index_ret = mar
        else:
            self.longside_ret = self.group_rets.group10 #- mar.values
            self.index_ret = mar
        self.longside_ret_eachyear = self.longside_ret.resample('Y').mean() * self.freq_controller.counts_one_year

        self.total_comments = pd.concat(
            [
                self.ic_icir_and_rank,
                self.long_short_comments,

                pd.DataFrame(
                    {
                        '评价指标': [
                            self.pos_neg_rate,
                            self.factor_cross_skew,
                            self.corr_itself,
                            self.factor_cover,
                        ]
                    },
                    index=[
                        '正值占比',
                        '因子截面偏度',
                        '自相关系数',
                        '因子覆盖率',
                    ],
                ),
                self.big_small_rankic,
                pd.DataFrame(
                    {
                        '评价指标': [
                            self.group1_ret_yearly,
                            self.group10_ret_yearly,
                        ]
                    },
                    index=['1组收益', '10组收益'],
                )
            ]
        )

    def plot_net_values(self, ilegend=1, without_breakpoint=0):

        tris = self.group_net_values.drop(columns=['long_short'])
        if without_breakpoint:
            tris = tris.dropna()
        net_value_fig = go.Figure()
        for col in tris.columns:
            net_value_fig.add_trace(
                go.Scatter(
                    x=tris.index,
                    y=tris[col],
                    mode='lines',
                    name=col,
                )
            )
        comments = (
            self.total_comments.applymap(lambda x: round(x, 4))
            .reset_index()
        )
        here = pd.concat(
            [
                comments.iloc[1:7, :].reset_index(drop=True),
                comments.iloc[[7,0,8,9,10,11], :].reset_index(drop=True),
            ],
            axis=1,
        )
        here.columns = ["绩效", "结果1", "多与空", "结果2"]
        # here=here.to_numpy().tolist()+[['信息系数','结果','绩效指标','结果']]
        table = go.Figure(data=[go.Table(
            header = dict(values=list(here.columns)),
            cells = dict(values=[here[col] for col in here.columns])
        )])
        
        group_returns = go.Bar(
            y = list(self.group_mean_rets),
            x = [i.replace('roup', '') for i in list(self.group_mean_rets.index)],
            name = '各组收益'
        )
        # table=go.Figure([go.Table(header=dict(values=list(here.columns)),cells=dict(values=here.to_numpy().tolist()))])
        if self.group1_ret_yearly > self.group10_ret_yearly:
            long_ic_bar = go.Bar(y=list(self.small_rankics.small_rankic), x=list(self.small_rankics.index),marker_color="red")
    
            long_ic_line = go.Scatter(
                y=list(self.small_rankics.small_rankic.cumsum()),
                x=list(self.small_rankics.index),
                name="多头ic",
                yaxis="y2",
                mode="lines",
                line=dict(color="blue"),
            )
        else:
            long_ic_bar = go.Bar(y=list(self.big_rankics.big_rankic), x=list(self.big_rankics.index),marker_color="red")
            
            long_ic_line = go.Scatter(
                y=list(self.big_rankics.big_rankic.cumsum()),
                x=list(self.big_rankics.index),
                name="多头ic",
                yaxis="y2",
                mode="lines",
                line=dict(color="blue"),
            )
        all_ic_line = go.Scatter(
            y=list(self.rankics.rankic.cumsum()),
            x=list(self.rankics.index),
            mode="lines",
            name="rankic",
            yaxis="y2",
            line=dict(color="red"),
        )

        fig = make_subplots(
            rows=3, cols=9,
            specs = [
                [{'rowspan': 3, 'colspan': 3}, None, None,
                 {'rowspan': 3, 'colspan': 3}, None, None,
                 {'rowspan': 3, 'colspan': 3}, None, None
                ],
                 [None] * 9,
                 [None] * 9
            ],
            subplot_titles = ['净值曲线', '各组超均收益', 'RankIC 时序图'],
            vertical_spacing = 0.15,
            horizontal_spacing = 0.045
        )

        for trace in net_value_fig.data:
            fig.add_trace(trace, row=1, col=1)
        
        #fig.add_trace(table, row=1, col=4)
        fig.add_trace(group_returns, row=1, col=4)
        fig.add_trace(long_ic_bar, row=1, col=7)
        fig.add_trace(long_ic_line.update(yaxis='y2'), row=1, col=7)
        fig.add_trace(all_ic_line, row=1, col=7)


        fig.update_layout(
            {
                'yaxis2':{
                    'title':'累计IC',
                    'overlaying':'y',
                    'side':'right',
                    'anchor': 'x3'
                }
            }
        )
        table.update_layout(width=800, height=400)
        table.show()

        fig.update_layout(
            showlegend=ilegend,
            width=1400,
            height=600,
            margin=dict(l=0, r=0, t=30, pad=0),
            font=dict(size=12),
            legend=dict(
                yanchor='top',
                y=0.99,
                xanchor='right',
                x=0.99,
            )
        )

        fig.show()


        

    @classmethod
    @lru_cache(maxsize=None)
    def prepare(cls):
        """通用数据准备"""
        cls.get_rets()

    def run(
        self,
        groups_num=10,
        ilegend=1,
        without_breakpoint=0,
        show_more_than=0.025,
    ):
        """运行回测部分"""
        self.__factors_out = self.factors.copy()
        self.factors = self.factors.shift(1)
        self.factors = self.factors.stack().reset_index()
        self.factors.columns = ["date", "code", "fac"]
        self.get_data(groups_num)
        self.get_group_rets_net_values(groups_num=groups_num)
        self.get_long_short_backtest()
        self.get_total_comments()

        if (show_more_than is None) or (show_more_than < max(self.group1_ret_yearly,self.group10_ret_yearly)):
            self.plot_net_values(
                ilegend=bool(ilegend),
                without_breakpoint=without_breakpoint,
            )
        else:
            logger.info(f'多头收益率为{round(max(self.group1_ret_yearly,self.group10_ret_yearly),3)}, ic为{round(self.rankics.rankic.mean(),3)}，表现太差，不展示了')
            # plt.show()



@do_on_dfs
class pure_Alpha(object):
    """封装选股框架"""

    def __init__(
        self,
        factors: pd.DataFrame,
        groups_num: int = 10,
        freq: str = "W",
        time_start: str = DATE1,
        time_end: str = DATE2,
        ilegend: bool = 1,
        without_breakpoint: bool = 0,
        show_more_than: float = 0.025,
    ) -> None:
        """一键回测框架，测试单因子的月频调仓的分组表现
        每月月底计算因子值，月初第一天开盘时买入，月末收盘最后一天收盘时卖出
        剔除上市不足60天的, 停牌天数超过一半的, st天数超过一半的
        月末收盘跌停的不卖出，月初开盘涨停的不买入
        由最好组和最差组的多空组合构成多空对冲组

        Parameters
        ----------
        factors : pd.DataFrame
            要用于检测的因子值, index是时间, columns是股票代码
        groups_num : int, optional
            分组数量, by default 10
        freq : str, optional
            回测频率, by default 'M'
        time_start : int, optional
            回测起始时间, by default None
        time_end : int, optional
            回测终止时间, by default None
        ilegend : bool, optional
            使用cufflinks绘图时, 是否显示图例, by default 1
        without_breakpoint : bool, optional
            画图的时候是否去除间断点, by default 0
        show_more_than : float, optional
            展示收益率大于多少的因子, 默认展示大于0.025的因子, by default 0.025
        """
        if time_start is not None:
            factors = factors[factors.index >= pd.Timestamp(str(time_start))]
        if time_end is not None:
            factors = factors[factors.index <= pd.Timestamp(str(time_end))]
        self.shen = Alpha(freq=freq)
        self.shen.set_basic_data()
        self.shen.set_factor_date_as_index(factors)
        self.shen.prepare()
        self.shen.run(
            groups_num=groups_num,
            ilegend=ilegend,
            without_breakpoint=without_breakpoint,
            show_more_than=show_more_than,
        )







