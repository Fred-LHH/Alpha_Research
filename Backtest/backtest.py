import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

from functools import reduce, lru_cache
from loguru import logger

from Data.get_data import read_daily, read_market
from decorators import do_on_dfs
from Data.utils import *
from config import *
from utils import *
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']  
rcParams['axes.unicode_minus'] = False  

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
        freq: str='D',
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
        zz500_open = read_market(open=1, index='000802', freq=cls.freq)

        cls.open = open.apply(pd.to_numeric, errors='coerce')
        zz500_open = zz500_open.apply(pd.to_numeric, errors='coerce')
        cls.market_ret = zz500_open / zz500_open.shift(1) - 1
        cls.open = cls.open.replace(0, np.nan)

    def set_factor_date_as_index(self, df: pd.DataFrame):
        """
        index为时间, columns为股票代码
        """
        if self.freq is not 'D':
            self.factors = df.resample(self.freq).last().dropna(how='all')
        else:
            self.factors = df.dropna(how='all')
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
        
        return pd.DataFrame({'IC': [rankic]}, index=['Performance']), pd.DataFrame({'1-5IC': [small_rankic], '6-10IC': [big_rankic]}, index=['Performance']).T


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
                'Performance': [
                    self.long_short_ret_yearly,
                    self.long_short_vol_yearly,
                    self.long_short_info_ratio,
                ]
            },
            index=['Annual Ret', 'Annual Vol', 'IR'],
        )

    def get_total_comments(self):
        ####
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
                        'Performance': [
                            self.pos_neg_rate,
                            self.factor_cross_skew,
                            self.corr_itself,
                            self.factor_cover,
                        ]
                    },
                    index=[
                        'Positive Ratio',
                        'Factor Skewness',
                        'Autocorrelation Coefficient',
                        'Factor Coverage',
                    ],
                ),
                self.big_small_rankic,
                pd.DataFrame(
                    {
                        'Performance': [
                            self.group1_ret_yearly,
                            self.group10_ret_yearly,
                        ]
                    },
                    index=['G1 ret', 'G10 ret'],
                )
            ]
        )

    def plot_net_values(self, ilegend=1, without_breakpoint=0):

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        fig.suptitle('factor analysis', fontsize=16, y=0.98)

        ax1 = axes[0]
        tris = self.group_net_values.drop(columns=['long_short'], errors='ignore')
        if without_breakpoint:
            tris = tris.dropna(how='all', axis=0).dropna(how='all', axis=1)

        if not tris.empty:
            for col in tris.columns:
                ax1.plot(tris.index, tris[col], label=col, linewidth=1.5)
            ax1.set_title('Net Value Curves', fontsize=14)
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Net Value', fontsize=12)
            if ilegend:
                ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, linestyle='--', alpha=0.7)
            fig.autofmt_xdate()
        else:
            ax1.text(0.5, 0.5, 'Insufficient Net Value Data', horizontalalign='center', verticalalignment='center', transform=ax1.transAxes)
            ax1.set_title('Net Value Curves', fontsize=14)


        ax2 = axes[1]
        if hasattr(self, 'group_mean_rets') and not self.group_mean_rets.empty:
            group_names = [name.replace('group', 'G') for name in self.group_mean_rets.index]
            bars = ax2.bar(group_names, self.group_mean_rets.values, color='skyblue', edgecolor='black')
            ax2.set_title('Group Annualized Excess Returns', fontsize=14)
            ax2.set_xlabel('Group', fontsize=12)
            ax2.set_ylabel('Annualized Excess Return', fontsize=12)
            ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
            for bar in bars:
                yval = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', va='bottom' if yval >= 0 else 'top', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'Insufficient Group Return Data', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
            ax2.set_title('Group Annualized Excess Returns', fontsize=14)

        ax3 = axes[2]
        plot_rank_ic_data = False
        if hasattr(self, 'group1_ret_yearly') and hasattr(self, 'group10_ret_yearly') and \
           not pd.isna(self.group1_ret_yearly) and not pd.isna(self.group10_ret_yearly):
            if self.group1_ret_yearly > self.group10_ret_yearly:
                ic_bar_data = self.small_rankics.small_rankic if hasattr(self, 'small_rankics') else pd.Series(dtype=float)
                bar_label_suffix = "Long RankIC"
                line_label_suffix = "CumLong RankIC"
            else:
                ic_bar_data = self.big_rankics.big_rankic if hasattr(self, 'big_rankics') else pd.Series(dtype=float)
                bar_label_suffix = "Long RankIC"
                line_label_suffix = "CumLong RankIC"
            
            if not ic_bar_data.empty and not ic_bar_data.isna().all():
                plot_rank_ic_data = True
                ax3.bar(ic_bar_data.index, ic_bar_data, color='lightcoral', alpha=0.7, label=bar_label_suffix)
                ax3.set_xlabel('Date', fontsize=12)
                ax3.set_ylabel('RankIC', color='lightcoral', fontsize=12)
                ax3.tick_params(axis='y', labelcolor='lightcoral')
                
                ax3_twin = ax3.twinx()
                ax3_twin.plot(ic_bar_data.index, ic_bar_data.cumsum(), color='mediumblue', linestyle='--', linewidth=1.5, label=line_label_suffix)
                
                if hasattr(self, 'rankics') and not self.rankics.rankic.empty and not self.rankics.rankic.isna().all():
                    ax3_twin.plot(self.rankics.index, self.rankics.rankic.cumsum(), color='darkgreen', linestyle='-', linewidth=1.5, label='Cumulative Total RankIC')
                
                ax3_twin.set_ylabel('Cumulative RankIC', color='mediumblue', fontsize=12)
                ax3_twin.tick_params(axis='y', labelcolor='mediumblue')
                ax3_twin.grid(True, axis='y', linestyle=':', alpha=0.5)

                if ilegend:
                    lines, labels = ax3.get_legend_handles_labels()
                    lines2, labels2 = ax3_twin.get_legend_handles_labels()
                    ax3.legend(lines + lines2, labels + labels2, loc='best', fontsize=10)
            
        if not plot_rank_ic_data:
            ax3.text(0.5, 0.5, "Insufficient RankIC Data", 
                     horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
        
        ax3.set_title('RankIC Time Series', fontsize=14)
        if plot_rank_ic_data: fig.autofmt_xdate()


        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
        plt.show()


        comments = self.total_comments.applymap(lambda x: round(x, 4) if isinstance(x, (int, float)) else x).reset_index()

        idx_1_to_7 = [i for i in range(1,7) if i < len(comments)]
        idx_others = [i for i in [7,0,8,9,10,11] if i < len(comments)]

        part1 = comments.iloc[idx_1_to_7, :].reset_index(drop=True) if idx_1_to_7 else pd.DataFrame(columns=comments.columns)
        part2 = comments.iloc[idx_others, :].reset_index(drop=True) if idx_others else pd.DataFrame(columns=comments.columns)
        
        max_rows = max(len(part1), len(part2))
        if len(part1) < max_rows:
            padding = pd.DataFrame(index=range(max_rows - len(part1)), columns=part1.columns)
            part1 = pd.concat([part1, padding]).reset_index(drop=True)
        if len(part2) < max_rows:
            padding = pd.DataFrame(index=range(max_rows - len(part2)), columns=part2.columns)
            part2 = pd.concat([part2, padding]).reset_index(drop=True)

        here = pd.concat([part1, part2], axis=1)
        here.columns = ["Performance1", "res1", "Performance2", "res2 "] 

        fig_table, ax_table = plt.subplots(figsize=(8, 3.5)) 
        ax_table.axis('tight')
        ax_table.axis('off')
        
        if not here.empty:
            table_obj = ax_table.table(cellText=here.values, 
                                     colLabels=here.columns, 
                                     loc='center', 
                                     cellLoc='center',
                                     colWidths=[0.25, 0.15, 0.25, 0.15]) 

            table_obj.auto_set_font_size(False)
            table_obj.set_fontsize(10)
            table_obj.scale(1.1, 1.1)

            
            for (i, j), cell in table_obj.get_celld().items():
                if i == 0: 
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('royalblue')
                cell.set_edgecolor('gray') 
            
            fig_table.suptitle("Performance Metrics Summary", fontsize=14)
        else:
            ax_table.text(0.5,0.5, "Insufficient Metrics Data",
                          horizontalalignment='center', verticalalignment='center', transform=ax_table.transAxes)
            fig_table.suptitle("Performance Metrics Summary", fontsize=14)

        plt.show()




        

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







