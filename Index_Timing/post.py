import numpy as np
import pandas as pd
import datetime
import statsmodels.api as sm
from statsmodels.sandbox.stats.runs import runstest_1samp
from plottable import ColumnDefinition, ColDef, Table
import os
from .plot import *

def check_output():
    if 'output' in os.listdir():
        pass
    else:
        os.mkdir('output')

riskfreerate = 0.02

class ReturnsPost():
    # returns,简单收益率序列  type:pd.Series index:pd.DatetimeIndex 
    # benchmark,基准收益率序列（可以多个） pd.DataFrame index:pd.DatetimeIndex, 0表示不设基准
    def __init__(self, returns, benchmark=None, stratname='strategy', freq='day',\
                  rf=riskfreerate, show=True):
        self.stratname = stratname
        self.returns = returns.fillna(0)
        # returns频率， 目前支持day, week
        if freq not in ['day', 'week']:
            print('输入频率错误')
            return
        else:
            self.freq = freq
            # 一年多少个bar
            if self.freq == 'day':
                self.annual_num = 250
            elif self.freq == 'week':
                self.annual_num = 48
        # 无风险利率
        self.rf = rf
        # 基准指数
        if type(benchmark)==type(None):
            benchmark = pd.DataFrame(index = self.returns.index)
            benchmark['zero'] = 0
            self.benchmark = benchmark
        self.benchmark = benchmark.reindex(self.returns.index, fill_value=0).fillna(0)
        self.sigma_benchmark = np.exp(np.log(self.benchmark[\
            self.benchmark.columns[0]]+1).std())-1
        self.cal_detail(show)
        if show:
            self.detail()

        self.daily_sharpe_ratio = self.cal_rolling_sharpe(window=35)

    def cal_rolling_sharpe(self, window: int):
        excess_ret = self.returns - self.rf
        rolling_mean = excess_ret.rolling(window=window).mean()
        rolling_std = excess_ret.rolling(window=window).std(ddof=1)
        sharpe_ratio = rolling_mean / rolling_std
        return sharpe_ratio
    # 详细评价表
    def cal_detail(self, show=False):
        # 策略绝对表现
        self.net = (1+self.returns).cumprod()
        self.lr = np.log(self.returns + 1)
        self.bars = len(self.returns)  
        self.years = (self.returns.index[-1]-self.returns.index[0]).days/365  
        self.return_total = self.net.iloc[-1]-1                    
        self.return_annual = (self.return_total+1)**(self.annual_num/self.bars)-1   
        self.sigma = (np.exp(self.lr.std())-1)*np.sqrt(self.annual_num)
        self.sharpe = (self.return_annual - self.rf)/self.sigma
        a = np.maximum.accumulate(self.net)
        self.drawdown = (a-self.net)/a
        # 超额表现
        self.excess_lr = self.lr-np.log(self.benchmark[self.benchmark.columns[0]]+1)
        self.excess_net = np.exp(self.excess_lr.cumsum())
        self.excess_total = self.excess_net.iloc[-1]/self.excess_net.iloc[0]
        self.excess_return_annual = self.excess_total**(1/self.years)-1
        self.excess_sigma = (np.exp(self.excess_lr.std())-1)*np.sqrt(self.annual_num)
        self.excess_sharpe = self.excess_return_annual/self.excess_sigma
        a = np.maximum.accumulate(self.excess_net)
        self.excess_drawdown = (a-self.excess_net)/a
        # CAPM (无风险收益为0)
        y = self.returns.fillna(0)
        x = self.benchmark[self.benchmark.columns[0]].fillna(0)
        x = sm.add_constant(x)
        model = sm.OLS(y, x.values).fit()
        # 市场风险暴露
        self.beta = model.params['x1']
        # 市场波动无法解释的截距项
        self.alpha = model.params['const']
        #model.summary()

        if show:
            col0 = pd.DataFrame(columns=['col0'])
            if self.freq == 'day':
                col0.loc[0] = 'strategy time(year, day)'
            elif self.freq == 'week':
                col0.loc[0] = 'strategy time(year, week)'
            col0.loc[1] = '%s, %s'%(round(self.years,1), len(self.net))
            col1 = pd.DataFrame(columns=['col1'])
            col1.loc[0] = 'total return(%)'
            col1.loc[1] = round(self.net.iloc[-1]*100-100, 1)
            col1.loc[2] = 'annual return(%)'
            col1.loc[3] = round(self.return_annual*100,1)
            col1.loc[4] = 'annual excess return(%)'
            col1.loc[5] = round(self.excess_return_annual*100,1)
            col2 = pd.DataFrame(columns=['col2'])
            col2.loc[0] = 'daily win rate(%)'  
            col2.loc[1] = round(100*(self.returns>=0).mean(),1)
            col2.loc[2] = 'daily excess win rate(%)'
            col2.loc[3] = round(100*(self.excess_lr>0).mean(),1)
            col3 = pd.DataFrame(columns=['col3'])
            col3.loc[0] = 'max drawdown(%)'
            col3.loc[1] = round(max(self.drawdown)*100, 1)
            col3.loc[2] = 'excess max drawdown(%)'
            col3.loc[3] = round(max(self.excess_drawdown)*100, 1)
            col3.loc[4] = 'volatility(%)'
            col3.loc[5] = round(self.sigma*100, 1)
            col4 = pd.DataFrame(columns=['col4'])
            col4.loc[0] = 'beta'
            col4.loc[1] = round(self.beta,2)
            col4.loc[2] = 'alpha(%)'
            col4.loc[3] = round(self.alpha*self.annual_num*100,1)
            col5 = pd.DataFrame(columns=['col5'])
            col5.loc[0] = 'sharpe'
            col5.loc[1] = round(self.sharpe,2)
            col5.loc[2] = 'excess sharpe'
            col5.loc[3] = round(self.excess_sharpe,2)
            col5.loc[4] = 'calmar ratio'
            col5.loc[5] = round(self.return_annual/max(self.drawdown),2)
            col6 = pd.DataFrame(columns=['col6'])
            col6.loc[0] = ''
            col6.loc[1] = ''
            col7 = pd.DataFrame(columns=['col7'])
            col7.loc[0] = 'youcheng test'   # 拒绝随机假设的概率
            col7.loc[1] = round(100*runstest_1samp(self.returns>0)[1],2)
            df_details = pd.concat([col0, col1, col2, col3, \
                    col4, col5, col6, col7], axis=1).fillna('')
            self.df_details = df_details
    def detail(self):
        plt, fig, ax = matplot(w=22)
        column_definitions = [ColumnDefinition(name='col0', group="basic params"), \
                              ColumnDefinition(name='col1', group="ret ab"), \
                            ColumnDefinition(name='col2', group='ret ab'), \
                            ColumnDefinition(name='col3', group='risk level'), \
                            ColumnDefinition(name="col4", group='risk adj'), \
                            ColumnDefinition(name="col5", group='risk adj'), \
                            ColumnDefinition(name="col6", group='strategy performance'),
                            ColumnDefinition(name="col7", group='performance analysis')] + \
                             [ColDef("index", title="", width=0, textprops={"ha":"right"})]
        tab = Table(self.df_details, row_dividers=False, col_label_divider=False, 
                    column_definitions=column_definitions,
                    odd_row_color="#e0f6ff", even_row_color="#f0f0f0", 
                    textprops={"ha": "center"})
        #ax.set_xlim(2,5)
        # 设置列标题文字和背景颜色(隐藏表头名)
        tab.col_label_row.set_facecolor("white")
        tab.col_label_row.set_fontcolor("white")
        # 设置行标题文字和背景颜色
        tab.columns["index"].set_facecolor("white")
        tab.columns["index"].set_fontcolor("white")
        tab.columns["index"].set_linewidth(0)
        check_output()
        plt.savefig('./output/details.png')
        plt.show()
# 时间起止（默认全部），是否显示细节,是否自定义输出图片名称，是否显示对数，是否显示超额
    def pnl(self, timerange=None, detail=True, filename=None, log=False, excess=False, title='pnl'):
        plt, fig, ax = matplot()
        # 只画一段时间内净值（用于展示局部信息,只列出sharpe）
        if type(timerange) != type(None):
            # 时间段内净值与基准
            net = self.net.loc[timerange[0]:timerange[1]]
            returns = self.returns.loc[timerange[0]:timerange[1]]
            # 计算年化夏普
            return_annual = (returns+1).prod()**(self.annual_num/len(returns))-1   
            sigma = (np.exp(np.log(1+returns).std())-1)*np.sqrt(self.annual_num)
            sharpe = (return_annual - self.rf)/sigma

            if detail:
                # 回撤
                a = np.maximum.accumulate(net)
                drawdown = (a-net)/a 
                ax.text(0.7,0.05,'annual return: {}%\nsharpe ratio:   {}\nmax drawdown:   {}%\n'.format(
                    round(100*return_annual,2), round(sharpe,2), 
                        round(100*max(drawdown),2)), transform=ax.transAxes)
                # 回撤
                ax2 = ax.twinx()
                ax2.fill_between(drawdown.index,-100*drawdown, 0, color='C1', alpha=0.1)
                if excess:
                    ax.plot(np.exp(self.excess_lr.loc[timerange[0]:timerange[1]].cumsum()), 
                            c='C3', label='excess return')
            else:
                ax.text(0.7,0.05,'Sharpe:  {}'.format(round(sharpe,2)), transform=ax.transAxes)
            ax.plot(net/net.iloc[0], c='C0', linewidth=2, label=self.stratname)
            # 如果基准是0就不绘制了
            if not type(self.benchmark)==type(None):
                # benchmark 匹配回测时间段, 基准从0开始
                benchmark = self.benchmark.loc[self.net.index[0]:self.net.index[-1]].copy()
                #benchmark.loc[self.net.index[0]] = 0
                benchmark.loc[self.net.index[0], :]  = 0
                # colors of benchmark
                colors_list = ['C4','C5','C6','C7']
                for i in range(len(benchmark.columns)):
                    ax.plot((benchmark[benchmark.columns[i]]+1).cumprod(), \
                            c=colors_list[i], label=benchmark.columns[i])
                if excess:
                    ax.plot(np.exp(self.excess_lr.cumsum()), c='C3', label='excess return')
                #plt.legend(loc='upper left')
            if log:
                # 对数坐标显示
                ax.set_yscale("log")
            ax.set_xlim(returns.index[0], returns.index[-1])
            plt.title(title)
            plt.gcf().autofmt_xdate()
        else:
    #评价指标
            ax.text(0.7,0.05,'annual return: {}%\nsharpe ratio:   {}\nmax drawdown:   {}%\n'.format(
            round(100*self.return_annual,2), round(self.sharpe,2), 
            round(100*max(self.drawdown),2)), transform=ax.transAxes)
        # 净值与基准
            ax.plot(self.net, c='C0', linewidth=2, label=self.stratname)
            # 如果基准是0就不绘制了
            if not (self.benchmark==0).all().values[0]:
                # benchmark 匹配回测时间段, 基准从0开始
                benchmark = self.benchmark.loc[self.net.index[0]:self.net.index[-1]].copy()
                #benchmark.loc[self.net.index[0]] = 0
                # colors of benchmark
                colors_list = ['C4','C5','C6','C7', 'C8', 'C9']*10
                for i in range(len(benchmark.columns)):
                    ax.plot((benchmark[benchmark.columns[i]]+1).cumprod(), alpha=0.8,\
                            c=colors_list[i], label=benchmark.columns[i])
                if excess:
                    ax.plot(np.exp(self.excess_lr.cumsum()), c='C3', linewidth=1.5, label='excess return')
                plt.legend(loc='upper left')
            if log:
                # 对数坐标显示
                ax.set_yscale("log")
            # 回撤
            ax2 = ax.twinx()
            ax2.fill_between(self.drawdown.index,-100*self.drawdown, 0, color='C1', alpha=0.1)
            ax.set_ylabel('cum net')
            ax.set_xlabel('date')
            ax2.set_ylabel('drawdown (%)')
            ax.set_xlim(self.net.index[0], self.net.index[-1])
            plt.title(title)
            plt.gcf().autofmt_xdate()
        check_output()
        if type(filename) == type(None):
            plt.savefig('./output/pnl.png')
        else:
            plt.savefig('./output/'+filename)
        plt.show()
# 滚动收益与夏普
    def rolling_return(self, key='return'):
        plt, fig, ax = matplot()
        if key=='return':
            ax.plot((self.net/self.net.shift(self.annual_num//2)-1)*100, c='C0', label='rolling half year return')
            ax2 = ax.twinx()
            ax2.plot((self.net/self.net.shift(self.annual_num)-1)*100, c='C3', label='rolling annual return')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.set_ylabel('(%)')
            ax2.set_ylabel('(%)')
        elif key=='sharpe':
            halfyearly_sharpe = (np.exp(self.lr.rolling(self.annual_num//2).mean()*self.annual_num)-1)/\
            ((np.exp(self.lr.rolling(self.annual_num//2).std())-1)*np.sqrt(self.annual_num))
            yearly_sharpe = (np.exp(self.lr.rolling(self.annual_num).mean()*self.annual_num)-1)/\
            ((np.exp(self.lr.rolling(self.annual_num).std())-1)*np.sqrt(self.annual_num))
            ax.plot(halfyearly_sharpe, c='C0', label='rolling yearly sharpe')
            ax2 = ax.twinx()
            ax2.plot(yearly_sharpe, c='C3', label='rolling yearly sharpe')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.set_ylim(-3, 10)
        ax.set_xlim(self.returns.index[0], self.returns.index[-1]) 
        plt.gcf().autofmt_xdate()
        check_output()
        plt.savefig('./output/rolling.png')
        plt.show()
# 年度与月度收益
    def pnl_yearly(self):
        lr = self.lr
        lr.name = 'lr'
        bench = np.log(self.benchmark[self.benchmark.columns[0]].fillna(0)+1)
        bench.name = 'bench'
        year = pd.Series(dict(zip(self.returns.index, self.returns.index.map(lambda x: x.year))))
        year.name = 'year'
        yearly_returns = pd.concat([year, lr, bench], axis=1)
        yearly_returns = (np.exp(yearly_returns.groupby('year').sum())*100-100)

        plt, fig, ax = matplot()

        len_years = len(yearly_returns)
        plot_x = range(len_years)
        plot_index = yearly_returns.index
        plot_height = yearly_returns['lr'].values
        plot_height1 = yearly_returns['bench'].values
        max_height = max(np.hstack([plot_height, plot_height1]))
        min_height = min(np.hstack([plot_height, plot_height1]))
        height = max_height-min_height
        # 如果benchmark是0的话就不画对比了
        if not (self.benchmark==0).any().values[0]:
            ax.bar([i-0.225  for i in plot_x], plot_height, width=0.45, color='C0', label='strategy')
            ax.bar([i+0.225  for i in plot_x], plot_height1, width=0.45, color='C4',\
                    label=self.benchmark.columns[0])
            for x, contri in zip(plot_x, plot_height):
                if contri>0:
                    plt.text(x-0.225, contri+height/30, round(contri,1), ha='center', color='C3', fontsize=8)
                else:
                    plt.text(x-0.225, contri-height/20, round(contri,1), ha='center', color='C2', fontsize=8)
            for x, contri in zip(plot_x, plot_height1):
                if contri>0:
                    plt.text(x+0.225, contri+height/30, round(contri,1), ha='center', color='C3', fontsize=8)
                else:
                    plt.text(x+0.225, contri-height/20, round(contri,1), ha='center', color='C2', fontsize=8)
        else:
            ax.bar([i  for i in plot_x], plot_height, width=0.45, color='C0', label='strategy')
            for x, contri in zip(plot_x, plot_height):
                if contri>0:
                    plt.text(x, contri+height/30, round(contri,1), ha='center', color='C3', fontsize=8)
                else:
                    plt.text(x, contri-height/20, round(contri,1), ha='center', color='C2', fontsize=8)
        plt.ylim(min_height-0.1*height, max_height+0.1*height)

        plt.legend()
        plt.title('年度收益')
        plt.xticks(plot_x, labels=plot_index)
        check_output()
        plt.ylabel('(%)')
        plt.savefig('./output/pnl_yearly.png')
        plt.show()
    def pnl_monthly(self, excess=False):
        if excess:
            df = self.excess_lr
        else:
            df = self.lr
        name = (lambda x: 0 if x==None else x)(df.name) 
        df = df.reset_index()
        # 筛出同月数据
        df['month'] = df['date'].apply(lambda x: x - datetime.timedelta(days=x.day-1,\
                                    hours=x.hour, minutes=x.minute, \
                                        seconds=x.second, microseconds=x.microsecond))
        df = df[['month', name]]
        df = df.set_index('month')[name]
        # 月度收益 %
        period_return = (np.exp(df.groupby('month').sum()) - 1)*100
        plt, fig, ax = month_thermal(period_return)
        check_output()
        plt.savefig('./output/pnl_monthly.png')
        plt.show()



