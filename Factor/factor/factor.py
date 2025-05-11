import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, shapiro, norm
import os
os.chdir('/Users/lihaohan/Desktop/quant/毕业设计')
from utils.utils import *
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Union
from operators import *

###################################兴业证券###################################
# 由于兴业证券研报因子的计算方法说的实在是不清楚，故止步于兴业金工的第四篇
##### 收益率分布因子
def cal_return_factors(df):
    #### 收益分布因子
    # rtn5_mean: 5分钟收益率均值
    # real_var: 已实现方差
    # rtn_skew: 收益率偏度
    # rtn_kurt: 收益率峰度
    # rv_up: 上行收益率已实现方差
    # rv_down: 下行收益率已实现方差
    # rv_umd: 上下行已实现方差之差除以已实现方差
    def rv_umd_factor(returns):
        returns = returns[~np.isnan(returns)]
        real_var = returns.var(ddof=1)

        up_returns = returns[returns > 0]
        down_returns = returns[returns < 0]

        rv_up = up_returns.var(ddof=1) if len(up_returns) > 0 else 0
        rv_down = down_returns.var(ddof=1) if len(down_returns) > 0 else 0
    
        if real_var > 0:
            rv_umd = (rv_up - rv_down) / real_var
        else:
            rv_umd = None

        rtn_skew = skew(returns) if len(returns) > 0 else None
        rtn_kurt = kurtosis(returns) if len(returns) > 0 else None

        return_5min = convert_freq(returns, '5T', method='sum')
        rtn5_mean = return_5min.mean() if len(return_5min) > 0 else None

        return {
            'real_var': real_var,
            'rv_up': rv_up,
            'rv_down': rv_down,
            'rv_umd': rv_umd,
            'rtn_skew': rtn_skew,
            'rtn_kurt': rtn_kurt,
            'rtn5_mean': rtn5_mean
        }
    
    df['return'] = df.groupby(['code', 'date'])['close'].pct_change()
    factors = df.groupby(['code', 'date']).apply(lambda x:pd.Series(rv_umd_factor(x['return'].values))).reset_index()

    return factors


def cal_nos(df,
            method=['sw']):
    '''
    计算收益率噪音偏离因子nos
    Args:
        df: DataFrame
        method: str, default 'sw'
    '''
    def nos_factors(returns):
        returns = returns[~np.isnan(returns)]
        mu_hat = np.mean(returns)
        sigma_hat = np.std(returns, ddof=1)
        # 标准化噪声epsilon_t
        epsilon = (returns - mu_hat) / sigma_hat
        '''
        factors = {}
        for m in method:
            if m == 'sw':
                # Shapiro-Wilk检验（样本量3~5000）
                if 3 <= len(epsilon) <= 5000:
                    W, _ = shapiro(epsilon)
                    factors['nos_sw'] = 1 - W  # 值越大偏离越严重
                else:
                    factors['nos_sw'] = np.nan
                    
            elif m == 'ks':
                # Kolmogorov-Smirnov检验
                D, _ = kstest(epsilon, 'norm', args=(0, 1))  
                factors['nos_ks'] = D  # D值越大偏离越严重
                
            elif m == 'ad':
                # Anderson-Darling检验
                try:
                    result = anderson(epsilon, dist='norm')
                    factors['nos_ad'] = result.statistic  # 统计量越大偏离越严重
                except:
                    factors['nos_ad'] = np.nan
        '''
        W, p_value = shapiro(epsilon)
        # nos_sw定义为1-W,即值越大偏离越严重
        nos_sw = 1 - W

        return {
            'nos_sw': nos_sw,
            'sigma_hat': sigma_hat,
            'epsilon': epsilon
        }
    
    valid_methods = ['sw']      #['sw', 'ks', 'ad']
    method = list(set([m.lower() for m in method]))
    for m in method:
        if m not in valid_methods:
            raise ValueError(f"Invalid method: {m}. Valid options: {valid_methods}")


    df['return'] = df.groupby(['code', 'date'])['close'].pct_change()
    factors = df.groupby(['code', 'date']).apply(lambda x:pd.Series(nos_factors(x['return'].values))).reset_index()

    return factors


def cal_exRtn(df,
              alpha=0.05):

    def exRtn_factor(returns):
        returns = returns[~np.isnan(returns)]
        # 计算左尾VaR和ES（极小值）
        if len(returns) > 0:
            left_var = np.percentile(returns, alpha * 100)
            left_exceedances = returns[returns <= left_var]
            exRtn_minVal = left_exceedances.mean() if len(left_exceedances) > 0 else None
            exRtn_minFre = len(left_exceedances)
    
            # 计算右尾VaR和ES（极大值）
            right_var = np.percentile(returns, (1 - alpha) * 100)
            right_exceedances = returns[returns >= right_var]
            exRtn_maxVal = right_exceedances.mean() if len(right_exceedances) > 0 else None
            exRtn_maxFre = len(right_exceedances)
        
            return {
                'exRtn_minVal': exRtn_minVal,
                'exRtn_minFre': exRtn_minFre,
                'exRtn_maxVal': exRtn_maxVal,
                'exRtn_maxFre': exRtn_maxFre
            }
        else:
            return {
                'exRtn_minVal': None,
                'exRtn_minFre': None,
                'exRtn_maxVal': None,
                'exRtn_maxFre': None
            }
    df['return'] = df.groupby(['code', 'date'])['close'].pct_change()
    factors = df.groupby(['code', 'date']).apply(lambda x:pd.Series(exRtn_factor(x['return'].values))).reset_index()

    return factors


def cal_gmm_mean(df,
                 max_iter=100,
                 tol=1e-6):
    '''
    计算混合高斯模型因子
    Args:
        df: DataFrame
        max_iter: int, default 100, EM最大迭代次数
        tol: float, default 1e-6 收敛阈值
    '''
    def em_gmm(returns):
        returns = returns[~np.isnan(returns)]
        if len(returns) < 10:
            return pd.Series([np.nan], index='mu_j')
        # 初始化参数
        pi = 0.8
        mu_s = np.median(returns)
        mu_j = np.percentile(returns, 75)
        sigma_s = np.std(returns) * 0.8
        sigma_j = np.std(returns) * 1.2

        log_likelihood_old = -np.inf
        for _ in range(max_iter):
            # === E步 ===
            prob_s = pi * norm.pdf(returns, mu_s, sigma_s)
            prob_j = (1 - pi) * norm.pdf(returns, mu_j, sigma_j)
            total_prob = prob_s + prob_j + 1e-10  # 防止除零
            
            gamma_s = prob_s / total_prob
            gamma_j = prob_j / total_prob
            
            # === M步 ===
            n1 = gamma_s.sum()
            n2 = gamma_j.sum()
            
            # 更新权重
            pi_new = n1 / (n1 + n2)
            
            # 更新均值（约束主成分接近0）
            mu_s_new = np.sum(gamma_s * returns) / n1
            mu_j_new = np.sum(gamma_j * returns) / n2
            
            # 更新方差（防止过拟合）
            sigma_s_new = np.sqrt(np.sum(gamma_s * (returns - mu_s_new)**2) / n1)
            sigma_j_new = np.sqrt(np.sum(gamma_j * (returns - mu_j_new)**2) / n2)
            
            # 收敛判断
            log_likelihood = np.sum(np.log(pi_new * prob_s + (1 - pi_new) * prob_j))
            if abs(log_likelihood - log_likelihood_old) < tol:
                break
            log_likelihood_old = log_likelihood

        gmm_mean = mu_j_new
        gmm_mean2wgt = mu_j_new / (1 - pi_new)
        gmm_meandif = mu_s_new - mu_j_new
        gmm_meandif2wgtdif = (mu_s_new - mu_j_new) / (2*pi_new - 1)

        return {
            'gmm_mean': gmm_mean,
            'gmm_mean2wgt': gmm_mean2wgt,
            'gmm_meandif': gmm_meandif,
            'gmm_meandif2wgtdif': gmm_meandif2wgtdif
        }
    df['next_close'] = df.groupby(['code', 'date'])['close'].shift(-1)
    df['log_return'] = np.log(df['next_close'] / df['close'])
    factors = df.groupby(['code', 'date']).apply(lambda x:pd.Series(em_gmm(x['log_return'].values))).reset_index()
    return factors


##### 成交量分布因子
def cal_vol_factors(df):
    # 对数成交量偏度
    # 对数成交量后尾(90%、10%占比)
    # 15min成交量变化率偏度
    # 15min成交量变化率峰度
    # 累计成交量均值
    # 累计成交量标准差
    def vol_factors(volume):
        volume = np.asarray(volume)
        volume = volume[~np.isnan(volume)]
        assert len(volume) > 0, '输入的成交量数据不能为空'
        log_vol = np.log(volume + 1e-6)

        logvol_skew = skew(log_vol)
        threshold_90 = np.percentile(log_vol, 90)
        threshold_10 = np.percentile(log_vol, 10)
        logvol_90tail = log_vol[log_vol > threshold_90].sum() / log_vol.sum()
        logvol_10tail = log_vol[log_vol < threshold_10].sum() / log_vol.sum()

        window = 15
        starts = np.arange(0, len(volume), window)
        vol_15min = np.add.reduceat(volume, starts)
        volroc = np.diff(vol_15min) / vol_15min[:-1]
        volroc_skew = skew(volroc[np.isfinite(volroc)])
        volroc_kurt = kurtosis(volroc[np.isfinite(volroc)])

        cum_vol = np.cumsum(volume)
        cumsumvol_mean = np.mean(cum_vol)
        cumsumvol_std = np.std(cum_vol, ddof=1)

        return {
            'logvol_skew': logvol_skew,
            'logvol_90tail': logvol_90tail,
            'logvol_10tail': logvol_10tail,
            'volroc_skew': volroc_skew,
            'volroc_kurt': volroc_kurt,
            'cumsumvol_mean': cumsumvol_mean,
            'cumsumvol_std': cumsumvol_std
        }
    factors = df.groupby(['code', 'date']).apply(lambda x:pd.Series(vol_factors(x['volume'].values))).reset_index()
    return factors


def cal_vol_entropy_factor(df,
                           n_bins=6,
                           bin_type='equal'):
    def vol_entropy_factor(volume,
                           n_bins,
                           bin_type):
        '''
        计算成交量分桶熵因子
        Args:
            volume: array, 股票日内1min成交量
            n_bins: int, 分桶数量
            bin_type: str, 分桶方式
        '''
        volume = np.asarray(volume)
        assert len(volume) > 0, '输入的成交量数据不能为空'
        if bin_type == 'quantile':
            percentiles = np.linspace(0, 100, n_bins + 1)
            bins = np.percentile(volume, percentiles)
        else:
            bins = np.linspace(volume.min(), volume.max(), n_bins + 1)

        hist, _ = np.histogram(volume, bins=bins)
        prob = (hist + 1e-8) / (hist.sum() + 1e-8 * n_bins)  # Laplace平滑
        entropy = -np.sum(prob * np.log(prob)) 
        max_entropy = np.log(n_bins)
        vol_entropy = entropy / max_entropy
        return {
            'vol_entropy': vol_entropy
        }
    factors = df.groupby(['code', 'date']).apply(lambda x:pd.Series(vol_entropy_factor(x['volume'].values, n_bins, bin_type))).reset_index()
    return factors
    
##### 时序信息因子
def cal_foc_comb_factor(df):

    def cal_foc_comb(close, volume):
        close, volume = np.asarray(close), np.asarray(volume)
        m_rtn = close[1:] / close[:-1] - 1
        m_rtn, volume = m_rtn[10:-10], volume[10:-10]
        vol_per = volume / np.sum(volume)
        rtn_foc = calc_first_diff_corr(m_rtn)
        vol_foc = calc_first_diff_corr(vol_per)
        foc_comb = (rtn_foc + vol_foc) / 2
        return {
            'rtn_foc': rtn_foc,
            'vol_foc': vol_foc,
            'foc_comb': foc_comb
        }
    factors = df.groupby(['code', 'date']).apply(lambda x:pd.Series(cal_foc_comb(x['close'].values, x['volume'].values))).reset_index()
    return factors

def cal_DW_comb_factor(df):
    def cal_DW_comb(close, volume):
        close, volume = np.asarray(close), np.asarray(volume)
        m_rtn = close[1:] / close[:-1] - 1
        m_rtn, volume = m_rtn[10:-10], volume[10:-10]
        vol_per = volume / np.sum(volume)
        rtn_DW = cal_DW_stats(m_rtn)
        vol_DW = cal_DW_stats(vol_per)
        DW_comb = (rtn_DW + vol_DW) / 2
        return {
            'rtn_DW': rtn_DW,
            'vol_DW': vol_DW,
            'DW_comb': DW_comb
        }
    factors = df.groupby(['code', 'date']).apply(lambda x:pd.Series(cal_DW_comb(x['close'].values, x['volume'].values))).reset_index()
    return factors


def cal_rho_comb_factor(df):
    df['return'] = df.groupby(['code', 'date'])['close'].pct_change()
    rtn_rho = df.groupby(['code', 'date']).apply(lambda x:ar1_residuals(x['return'].values, True)).reset_index(name='rtn_rho')
    vol_rho = df.groupby(['code', 'date']).apply(lambda x:ar1_residuals(x['volume'].values, True)).reset_index(name='vol_rho')
    rho_comb = pd.merge(rtn_rho, vol_rho, on=['code', 'date'], how='outer')
    rho_comb['rho_comb'] = (rho_comb['rtn_rho'] + rho_comb['vol_rho']) / 2
    return rho_comb


def cal_LBQ_comb_factor(df):
    def cal_LBQ(data, 
            lags: Union[int, list, None] = None):
    
        if len(data) == 0:
            return np.nan
        if isinstance(lags, int):
            q_stats = acorr_ljungbox(data, lags=lags)
            if q_stats.empty:
                return np.nan
            q_stats_std = np.std(q_stats['lb_stat'].values)
        elif isinstance(lags, list):
            q_stats = []
            for lag in lags:
                lb_test = acorr_ljungbox(data, lags=[lag])
                q_stat = lb_test['lb_stat'].values[0]
                q_stats.append(q_stat)
            if len(q_stats) == 0:
                return np.nan
            q_stats_std = np.std(q_stats)
        elif lags is None:
            if len(data) < 10:
                return np.nan
            q_stats = acorr_ljungbox(data, lags=None)
            if q_stats.empty:
                return np.nan
            q_stats_std = np.std(q_stats['lb_stat'].values)
        return q_stats_std
    lags = 20
    df['return'] = df.groupby(['code', 'date'])['close'].pct_change()
    df.dropna(inplace=True)
    rtn_LBQ = df.groupby(['code', 'date']).apply(lambda x:cal_LBQ(x['return'].values, lags)).reset_index(name='rtn_LBQ')
    df['volume_sum'] = df.groupby(['code', 'date'])['volume'].transform('sum')
    df['volume'] = df['volume'] / df['volume_sum']
    df.drop(columns=['volume_sum'], inplace=True)
    df.dropna(inplace=True) 
    vol_LBQ = df.groupby(['code', 'date']).apply(lambda x:cal_LBQ(x['volume'].values, lags)).reset_index(name='vol_LBQ')
    rho_comb = pd.merge(rtn_LBQ, vol_LBQ, on=['code', 'date'], how='outer')
    rho_comb['LBQ_Comb'] = (rho_comb['rtn_LBQ'] + rho_comb['vol_LBQ']) / 2
    return rho_comb


def cal_highStdRtn_mean_factor(df):
    def cal_highStdRtn_mean(returns):
        returns = pd.Series(returns)
        return_5min = returns.rolling(window=5, min_periods=1).sum()
        #计算各个分钟节点过去 30分钟的 5 分钟滚动收益率标准差
        return_std = return_5min.rolling(window=30, min_periods=5).std()
        #筛选出标准差处于日内该股 80%分位数以上的时间节点
        threshold = return_std.quantile(0.8)
        highStdRtn_mean = return_5min[return_std > threshold]
        highStdRtn_mean = highStdRtn_mean.mean()
        return {
            'highStdRtn_mean': highStdRtn_mean
        }
    df['return'] = df.groupby(['code', 'date'])['close'].pct_change()
    factors = df.groupby(['code', 'date']).apply(lambda x:pd.Series(cal_highStdRtn_mean(x['return'].values))).reset_index()
    return factors

def cal_pv_factors(df):
    def hpos(highest_price, kl_score):
        #hpos:日内最高价出现的分钟时间戳
        kl_score = kl_score.rank(method='first').astype(float)
        hpos = kl_score.loc[highest_price.idxmax()]
        return hpos

    def tvr(volume):
        #tvr:收盘15min的成交量占当日比例
        if volume.sum() == 0:
            return np.nan
        tvr = volume[-15:].sum() / volume.sum()
        return tvr

    def pvi(close, volume):
        #pvi:放量时的分钟收益率之和
        returns = close.pct_change()
        pvi = returns.loc[volume > volume.mean()].sum()
        return pvi

    def varr(volume):
        #varr:5min相较于10min成交量比例
        var_5min = np.var(volume.rolling(window=5).mean())
        var_10min = np.var(volume.rolling(window=10).mean())
        if var_10min == 0:
            return np.nan
        varr = (var_5min / 5) / (var_10min / 10)
        return varr

    def pvcorr(close, volume):
        #pvcorr:分钟收益率与成交量变化相关性
        close_diff = close.diff()
        volume_diff = volume.diff()
        close_x = np.sum(np.square(close_diff)) / (len(close) - 1)
        volume_x = np.sum(np.square(volume_diff)) / (len(volume) - 1)
        close_volume_xy = np.sum(close_diff * volume_diff) / (len(close) - 1)
    
        pvcorr = close_volume_xy / np.sqrt(close_x * volume_x)
    
        return pvcorr

    def icp(close, kl_score):
        #icp:日内快速上涨和快速下跌出现时间的中位数之差
        kl_score = kl_score.rank(method='first').astype(float)
        returns = close.pct_change()
        mean = np.mean(returns)
        std = np.std(returns)

        up = kl_score.loc[returns > (mean + std)]
        down = kl_score.loc[returns < (mean - std)]
        icp = np.median(up) - np.median(down)
        return icp

    def r1min_skew(close):
        #r1min_skew:分钟收益率日内分布偏度
        returns = close.pct_change()
        returns = returns[~np.isnan(returns)]
        r1min_skew = skew(returns)
        return r1min_skew

    def trendratio(open, close):
        #trendratio:当日收益率与分钟收益率绝对值之和的比值
        returns = close.pct_change()
        returns = returns[~np.isnan(returns)]
        trendratio = (close.iloc[-1] - open.iloc[0]) / np.abs(returns).sum()
        return trendratio

    def vov(low, high):
        #vov:日内波动率Parkinson估计
        constant = 1 / (4 * np.log(2) * len(low))
        ext = np.sum(np.square(np.log(high / low)))

        vov = np.sqrt(constant * ext)
        return vov

    _hpos = df.groupby(['code', 'date']).apply(lambda x: hpos(x['high'], x['time'])).reset_index().rename(columns={0: 'hpos'})
    _tvr = df.groupby(['code', 'date']).apply(lambda x: tvr(x['volume'])).reset_index().rename(columns={0: 'tvr'})
    _pvi = df.groupby(['code', 'date']).apply(lambda x: pvi(x['close'], x['volume'])).reset_index().rename(columns={0: 'pvi'})
    _varr = df.groupby(['code', 'date']).apply(lambda x: varr(x['volume'])).reset_index().rename(columns={0: 'varr'})
    _pvcorr = df.groupby(['code', 'date']).apply(lambda x: pvcorr(x['close'], x['volume'])).reset_index().rename(columns={0: 'pvcorr'})
    _icp = df.groupby(['code', 'date']).apply(lambda x: icp(x['close'], x['time'])).reset_index().rename(columns={0: 'icp'})
    _r1min_skew = df.groupby(['code', 'date']).apply(lambda x: r1min_skew(x['close'])).reset_index().rename(columns={0: 'r1min_skew'})
    _trendratio = df.groupby(['code', 'date']).apply(lambda x: trendratio(x['open'], x['close'])).reset_index().rename(columns={0: 'trendratio'})
    _vov = df.groupby(['code', 'date']).apply(lambda x: vov(x['low'], x['high'])).reset_index().rename(columns={0: 'vov'})

    result = _hpos.merge(_tvr, on=['code', 'date'], how='outer').merge(_pvi, on=['code', 'date'], how='outer').merge(_varr, on=['code', 'date'], how='outer').merge(_pvcorr, on=['code', 'date'], how='outer').merge(_icp, on=['code', 'date'], how='outer').merge(_r1min_skew, on=['code', 'date'], how='outer').merge(_trendratio, on=['code', 'date'], how='outer').merge(_vov, on=['code', 'date'], how='outer')
    return result

def cal_Smart_factor(df):
    #开源证券-切割

    #聪明钱因子改进版 ps：截至值均为15%
    #(1)S = |R| / V^0.1
    #(2)S = V
    #(3)S = rank(|R|) + rank(V)
    #(4)S = |R| / ln(V)
    def Smart_factor_1(close, volume):
        #S = |R| / V^0.1
        returns = close.pct_change()
        #returns = returns[~np.isnan(returns)]
        S = np.abs(returns) / np.power(volume, 0.1)
        S = S.replace([np.inf, -np.inf], None)
        #将分钟数据按S从大到小排序，取成交量累计占比前20%的分钟
        sorted_data = pd.DataFrame({'S': S, 'close': close, 'volume': volume}).sort_values(by='S', ascending=False)
        sorted_data['cumulative_volume'] = sorted_data['volume'].cumsum()
        total_volume = sorted_data['volume'].sum()
        Smart_trade = sorted_data[sorted_data['cumulative_volume'] <= total_volume * 0.15].index

        smart_vwap = (sorted_data.loc[Smart_trade, 'close'] * sorted_data.loc[Smart_trade, 'volume']).sum() / sorted_data.loc[Smart_trade, 'volume'].sum()

        total_vwap = (sorted_data['close'] * sorted_data['volume']).sum() / sorted_data['volume'].sum()

        Q= smart_vwap / total_vwap

        return Q

    def Smart_factor_2(close, volume):
        #S = V
        S = volume
        #将分钟数据按S从大到小排序，取成交量累计占比前20%的分钟
        sorted_data = pd.DataFrame({'S': S, 'close': close, 'volume': volume}).sort_values(by='S', ascending=False)
        sorted_data['cumulative_volume'] = sorted_data['volume'].cumsum()
        total_volume = sorted_data['volume'].sum()
        Smart_trade = sorted_data[sorted_data['cumulative_volume'] <= total_volume * 0.15].index

        smart_vwap = (sorted_data.loc[Smart_trade, 'close'] * sorted_data.loc[Smart_trade, 'volume']).sum() / sorted_data.loc[Smart_trade, 'volume'].sum()

        total_vwap = (sorted_data['close'] * sorted_data['volume']).sum() / sorted_data['volume'].sum()

        Q= smart_vwap / total_vwap

        return Q

    def Smart_factor_3(close, volume):
        #S = rank(|R|) + rank(V)
        returns = close.pct_change()
        #returns = returns[~np.isnan(returns)]
        S = np.abs(returns).rank() + volume.rank()
        #将分钟数据按S从大到小排序，取成交量累计占比前20%的分钟
        sorted_data = pd.DataFrame({'S': S, 'close': close, 'volume': volume}).sort_values(by='S', ascending=False)
        sorted_data['cumulative_volume'] = sorted_data['volume'].cumsum()
        total_volume = sorted_data['volume'].sum()
        Smart_trade = sorted_data[sorted_data['cumulative_volume'] <= total_volume * 0.15].index

        smart_vwap = (sorted_data.loc[Smart_trade, 'close'] * sorted_data.loc[Smart_trade, 'volume']).sum() / sorted_data.loc[Smart_trade, 'volume'].sum()

        total_vwap = (sorted_data['close'] * sorted_data['volume']).sum() / sorted_data['volume'].sum()

        Q= smart_vwap / total_vwap

        return Q

    def Smart_factor_4(close, volume):
        #S = |R| / ln(V)
        returns = close.pct_change()
        #returns = returns[~np.isnan(returns)]
        S = np.abs(returns) / np.log(volume)
        S = S.replace([np.inf, -np.inf], None)
        #将分钟数据按S从大到小排序，取成交量累计占比前20%的分钟
        sorted_data = pd.DataFrame({'S': S, 'close': close, 'volume': volume}).sort_values(by='S', ascending=False)
        sorted_data['cumulative_volume'] = sorted_data['volume'].cumsum()
        total_volume = sorted_data['volume'].sum()
        Smart_trade = sorted_data[sorted_data['cumulative_volume'] <= total_volume * 0.15].index

        smart_vwap = (sorted_data.loc[Smart_trade, 'close'] * sorted_data.loc[Smart_trade, 'volume']).sum() / sorted_data.loc[Smart_trade, 'volume'].sum()

        total_vwap = (sorted_data['close'] * sorted_data['volume']).sum() / sorted_data['volume'].sum()

        Q= smart_vwap / total_vwap

        return Q
    
    SM_factor_1 = df.groupby(['code', 'date']).apply(lambda x:Smart_factor_1(x['close'], x['volume'])).reset_index()
    SM_factor_1.columns = ['code', 'date', 'SM_1']
    SM_factor_2 = df.groupby(['code', 'date']).apply(lambda x:Smart_factor_2(x['close'], x['volume'])).reset_index()
    SM_factor_2.columns = ['code', 'date', 'SM_2']
    SM_factor_3 = df.groupby(['code', 'date']).apply(lambda x:Smart_factor_3(x['close'], x['volume'])).reset_index()
    SM_factor_3.columns = ['code', 'date', 'SM_3']
    SM_factor_4 = df.groupby(['code', 'date']).apply(lambda x:Smart_factor_4(x['close'], x['volume'])).reset_index()
    SM_factor_4.columns = ['code', 'date', 'SM_4']
    results = SM_factor_1.merge(SM_factor_2, on=['code', 'date'], how='outer').merge(SM_factor_3, on=['code', 'date'], how='outer').merge(SM_factor_4, on=['code', 'date'], how='outer')
    return results

def cal_extreme_ret_rev(df):

    def cal_intra_extreme_ret_rev(ret):
        S = abs(ret - median(ret))
        max_id = np.argmax(S)
        max_value = S[max_id]

        if max_id > 0:
            pre_value = S[max_id - 1]
        else:
            pre_value = None
        return {
            'extreme_ret': max_value,
            'pre_extreme_ret': pre_value
        }
    df['return'] = df.groupby(['code', 'date'])['close'].pct_change()
    factors = df.groupby(['code', 'date']).apply(lambda x:pd.Series(cal_intra_extreme_ret_rev(x['return'].values))).reset_index()
    return factors








