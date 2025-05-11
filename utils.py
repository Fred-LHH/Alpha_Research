import os
import pandas as pd
import numpy as np
import os
os.chdir('/Users/lihaohan/Alpha_Research')
from decorators import do_on_dfs
from functools import reduce
#import polars as pl
import mpire
from typing import Callable, Union, Dict, List, Tuple
import matplotlib.pyplot as plt
import datetime

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False, nb_workers=10)



@do_on_dfs
def standardlize(df: pd.DataFrame, all_pos: bool = 0) -> pd.DataFrame:
    """对因子dataframe做横截面z-score标准化

    Parameters
    ----------
    df : pd.DataFrame
        要做标准化的因子值, index是时间, columns是股票代码
    all_pos : bool, optional
        是否要将值都变成正数，通过减去截面的最小值实现, by default 0

    Returns
    -------
    pd.DataFrame
        标准化之后的因子
    """
    df = ((df.T - df.T.mean()) / df.T.std()).T
    if all_pos:
        df = (df.T - df.T.min()).T
    return df

@do_on_dfs
def boom_one(
    df: pd.DataFrame, 
    backsee: int = 5, 
    daily: bool = 0, 
    min_periods: int = None
    ) -> pd.DataFrame:

    if min_periods is None:
        min_periods = int(backsee * 0.5)
    if not daily:
        df_mean = (
            df.rolling(backsee, min_periods=min_periods).mean().resample("W").last()
        )
    else:
        df_mean = df.rolling(backsee, min_periods=min_periods).mean()
    return df_mean

@do_on_dfs
def to_percent(x: float) -> Union[float, str]:
    """把小数转化为2位小数的百分数

    Parameters
    ----------
    x : float
        要转换的小数

    Returns
    -------
    Union[float,str]
        空值则依然为空，否则返回带%的字符串
    """
    if np.isnan(x):
        return x
    else:
        x = str(round(x * 100, 2)) + "%"
        return x
    
@do_on_dfs
def clip_mad(
    df: pd.DataFrame, 
    n: float = 5, 
    replace: bool = 1, 
    keep_trend: bool = 0
) -> pd.DataFrame:
    if keep_trend:
        df = df.stack().reset_index()
        df.columns = ["date", "code", "fac"]

        def clip_sing(x: pd.DataFrame, n: float = 3):
            median = x.fac.quantile(0.5)
            diff_median = ((x.fac - median).abs()).quantile(0.5)
            max_range1 = median + n * diff_median
            min_range1 = median - n * diff_median
            max_range2 = median + (n + 0.5) * diff_median
            min_range2 = median - (n + 0.5) * diff_median
            x = x.sort_values(["fac"])
            x_min = x[x.fac <= min_range1]
            x_max = x[x.fac >= max_range1]
            x_middle = x[(x.fac > min_range1) & (x.fac < max_range1)]
            x_min.fac = np.nan
            x_max.fac = np.nan
            if x_min.shape[0] >= 1:
                x_min.fac.iloc[-1] = min_range1
                if x_min.shape[0] >= 2:
                    x_min.fac.iloc[0] = min_range2
                    x_min.fac = x_min.fac.interpolate()
            if x_max.shape[0] >= 1:
                x_max.fac.iloc[-1] = max_range2
                if x_max.shape[0] >= 2:
                    x_max.fac.iloc[0] = max_range1
                    x_max.fac = x_max.fac.interpolate()
            x = pd.concat([x_min, x_middle, x_max]).sort_values(["code"])
            return x

        df = df.groupby(["date"]).apply(lambda x: clip_sing(x, n))
        try:
            df = df.reset_index()
        except Exception:
            ...
        df = df.drop_duplicates(subset=["date", "code"]).pivot(
            index="date", columns="code", values="fac"
        )
        return df
    elif replace:

        def clip_sing(x: pd.Series, n: float = 3):
            median = x.quantile(0.5)
            diff_median = ((x - median).abs()).quantile(0.5)
            max_range = median + n * diff_median
            min_range = median - n * diff_median
            x = x.where(x < max_range, max_range)
            x = x.where(x > min_range, min_range)
            return x

        df1 = df.T.apply(lambda x: clip_sing(x, n)).T
        df = np.abs(np.sign(df)) * df1
        return df
    else:
        df0 = df.T
        median = df0.quantile(0.5)
        diff_median = ((df0 - median).abs()).quantile(0.5)
        max_range = median + n * diff_median
        min_range = median - n * diff_median
        mid1 = (((df0 - min_range) >= 0) + 0).replace(0, np.nan)
        mid2 = (((df0 - max_range) <= 0) + 0).replace(0, np.nan)
        return (df0 * mid1 * mid2).T


@do_on_dfs
def clip_three_sigma(df: pd.DataFrame, n: float = 3) -> pd.DataFrame:
    df0 = df.T
    mean = df0.mean()
    std = df0.std()
    max_range = mean + n * std
    min_range = mean - n * std
    mid1 = (((df0 - min_range) >= 0) + 0).replace(0, np.nan)
    mid2 = (((df0 - max_range) <= 0) + 0).replace(0, np.nan)
    return (df0 * mid1 * mid2).T


@do_on_dfs
def clip_percentile(
    df: pd.DataFrame, min_percent: float = 0.025, max_percent: float = 0.975
) -> pd.DataFrame:
    df0 = df.T
    max_range = df0.quantile(max_percent)
    min_range = df0.quantile(min_percent)
    mid1 = (((df0 - min_range) >= 0) + 0).replace(0, np.nan)
    mid2 = (((df0 - max_range) <= 0) + 0).replace(0, np.nan)
    return (df0 * mid1 * mid2).T


@do_on_dfs
def clip(
    df: pd.DataFrame,
    mad: bool = 0,
    three_sigma: bool = 0,
    percentile: bool = 0,
    parameter: Union[float, tuple] = None,
) -> pd.DataFrame:
    """对因子值进行截面去极值的操作

    Parameters
    ----------
    df : pd.DataFrame
        要处理的因子表, columns为股票代码, index为时间
    mad : bool, optional
        使用mad法去极值, 先计算所有因子与平均值之间的距离总和来检测离群值, by default 0
    three_sigma : bool, optional
        根据均值和几倍标准差做调整, by default 0
    percentile : bool, optional
        根据上下限的分位数去极值, by default 0
    parameter : Union[float,tuple], optional
        参数, mad和three_sigma默认参数为3, 输入float形式;而percentile默认参数为(0.025,0.975), 输入tuple形式, by default None

    Returns
    -------
    pd.DataFrame
        去极值后的参数

    Raises
    ------
    ValueError
        不指定方法或参数类型错误，将报错
    """
    if mad and ((isinstance(parameter, float)) or (parameter is None)):
        return clip_mad(df, parameter)
    elif three_sigma and ((isinstance(parameter, float)) or (parameter is None)):
        return clip_three_sigma(df, parameter)
    elif percentile and ((isinstance(parameter, tuple)) or (parameter is None)):
        return clip_percentile(df, parameter[0], parameter[1])
    else:
        raise ValueError("参数输入错误")
'''
def de_cross_polars(
    y: Union[pd.DataFrame, pl.DataFrame],
    xs: Union[list[pd.DataFrame], list[pl.DataFrame]],
) -> pd.DataFrame:
    """因子正交函数, 使用polars库实现

    Parameters
    ----------
    y : Union[pd.DataFrame, pl.DataFrame]
        要研究的因子, 形式与h5存数据的形式相同, index是时间, columns是股票
    xs : Union[list[pd.DataFrame], list[pl.DataFrame]]
        要被正交掉的干扰因子们, 传入一个列表, 每个都是h5存储的那种形式的df, index是时间, columns是股票

    Returns
    -------
    pd.DataFrame
        正交后的残差, 形式与y相同, index是时间, columns是股票
    """
    if isinstance(y, pd.DataFrame):
        y.index.name='date'
        y = pl.from_pandas(y.reset_index())
    if isinstance(xs[0], pd.DataFrame):
        for i in range(len(xs)):
            xs[i].index.name='date'
        xs = [pl.from_pandas(x.reset_index()) for x in xs]
    y = y.unpivot(index="date", variable_name="code").drop_nulls()
    xs = [x.unpivot(index="date", variable_name="code").drop_nulls() for x in xs]
    for num, i in enumerate(xs):
        y = y.join(i, on=["date", "code"], suffix=f"_{num}")
    y = (
        y.select(
            "date",
            "code",
            pl.col("value")
            .least_squares.ols(
                *[pl.col(f"value_{i}") for i in range(len(xs))],
                add_intercept=True,
                mode="residuals",
            )
            .over("date")
            .alias("resid"),
        )
        .pivot("code", index="date", values="resid")
        .sort('date')
        .to_pandas()
        .set_index("date")
    )
    return y
'''
def np_ols(X, y):
    """
    参数:
    X: numpy.ndarray, 自变量矩阵
    y: numpy.ndarray, 因变量向量
    """
    X = np.column_stack((np.ones(X.shape[0]), X))
    betas = np.linalg.inv(X.T @ X) @ X.T @ y
    return betas

def de_cross(y, x_list):
    """
    因子正交化函数。

    参数：
    y: pd.DataFrame, 因变量
    x_list: 包含 pd.DataFrame 的列表, 自变量

    返回：
    pd.DataFrame,正交化后的残差。
    """
    
    dfs = [y] + x_list
    dfs = same_index(same_columns(dfs))
    dates = dfs[0].index
    def one(timestamp):
        y_series = y.loc[timestamp]
        xs = pd.concat([x.loc[timestamp].to_frame(str(num)) for num,x in enumerate(x_list)],axis=1)
        yxs = pd.concat([y_series.to_frame('factor'),xs],axis=1).dropna()
        betas = np_ols(yxs[xs.columns].to_numpy(dtype=float),yxs['factor'].to_numpy(dtype=float))
        yresi = y_series-sum([betas[i+1]*xs[str(i)] for i in range(len(betas)-1)])-betas[0]
        yresi = yresi.to_frame('f_res').T
        yresi.index=[timestamp]
        return yresi
        
    with mpire.WorkerPool(n_jobs=10) as pool:
        residual_df=pd.concat(pool.map(one, dates)).sort_index()
    return residual_df


@do_on_dfs
def convert_to_daily_(
    df: pd.DataFrame, code: str, entry: str, exit: str, kind: str
) -> pd.DataFrame:
    """
    df是要包含任意多列的表格, 为Dataframe格式, 主要内容为, 每一行是
    一只股票或一只基金的代码、分类、进入该分类的时间、移除该分类的时间,
    除此之外,还可以包含很多其他内容
    code是股票代码列的列名,为字符串格式;
    entry是股票进入该分类的日期的列名,为字符串格式
    exit是股票退出该分类的日期的列名,为字符串格式
    kind是分类列的列名,为字符串格式
    """
    df = df[[code, entry, exit, kind]]
    df = df.fillna(int(datetime.datetime.now().strftime("%Y%m%d")))
    try:
        if type(df[entry].iloc[0]) == str:
            df[entry] = df[entry].astype(str)
            df[exit] = df[exit].astype(str)
        else:
            df[entry] = df[entry].astype(int).astype(str)
            df[exit] = df[exit].astype(int).astype(str)
    except Exception:
        print("输入数据的进入日期和推出日期, 既不是字符串,也不是数字格, 请检查一下")
    df = df.set_index([code, kind])
    df = df.stack().to_frame(name="date")

    def fill_middle(df1):
        min_time = df1.date.min()
        max_time = df1.date.max()
        df2 = pd.DataFrame({"date": pd.date_range(min_time, max_time)})
        return df2

    ff = df.reset_index().groupby([code, kind]).apply(fill_middle)
    ff = ff.reset_index()
    ff = ff[[code, kind, "date"]]
    #ff = ff[ff.date >= pd.Timestamp("2004-01-01")]
    return ff


@do_on_dfs
def add_cross_standardlize(*args: list) -> pd.DataFrame:
    """将众多因子横截面做z-score标准化之后相加

    Returns
    -------
    `pd.DataFrame`
        合成后的因子
    """
    res = reduce(lambda x, y: x + y, [standardlize(i) for i in args])
    return res


def show_corr(
    fac1: pd.DataFrame,
    fac2: pd.DataFrame,
    method: str = "pearson",
    plt_plot: bool = 1,
    show_series: bool = 0,
) -> float:
    """展示两个因子的截面相关性

    Parameters
    ----------
    fac1 : pd.DataFrame
        因子1
    fac2 : pd.DataFrame
        因子2
    method : str, optional
        计算相关系数的方法, by default "pearson"
    plt_plot : bool, optional
        是否画出相关系数的时序变化图, by default 1
    show_series : bool, optional
        返回相关性的序列，而非均值

    Returns
    -------
    `float`
        平均截面相关系数
    """
    corr = fac1.corrwith(fac2, axis=1, method=method)
    if show_series:
        return corr
    else:
        if plt_plot:
            corr.plot(rot=60)
            plt.show()
        return corr.mean()


def show_corrs(
    factors: list[pd.DataFrame],
    factor_names: list[str] = None,
    print_bool: bool = True,
    show_percent: bool = True,
    method: str = "pearson",
) -> pd.DataFrame:
    """展示很多因子两两之间的截面相关性

    Parameters
    ----------
    factors : list[pd.DataFrame]
        所有因子构成的列表, by default None
    factor_names : list[str], optional
        上述因子依次的名字, by default None
    print_bool : bool, optional
        是否打印出两两之间相关系数的表格, by default True
    show_percent : bool, optional
        是否以百分数的形式展示, by default True
    method : str, optional
        计算相关系数的方法, by default "pearson"

    Returns
    -------
    `pd.DataFrame`
        两两之间相关系数的表格
    """
    corrs = []
    for i in range(len(factors)):
        main_i = factors[i]
        follows = factors[i + 1 :]
        corr = [show_corr(main_i, i, plt_plot=False, method=method) for i in follows]
        corr = [np.nan] * (i + 1) + corr
        corrs.append(corr)
    if factor_names is None:
        factor_names = [f"fac{i}" for i in list(range(1, len(factors) + 1))]
    corrs = pd.DataFrame(corrs, columns=factor_names, index=factor_names)
    np.fill_diagonal(corrs.to_numpy(), 1)
    corrs=pd.DataFrame(corrs.fillna(0).to_numpy()+corrs.fillna(0).to_numpy().T-np.diag(np.diag(corrs)),index=corrs.index,columns=corrs.columns)
    if show_percent:
        pcorrs = corrs.applymap(to_percent)
    else:
        pcorrs = corrs.copy()
    if print_bool:
        return pcorrs
    else:
        return corrs


def same_columns(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """保留多个dataframe共同columns的部分

    Parameters
    ----------
    dfs : List[pd.DataFrame]
        多个dataframe

    Returns
    -------
    List[pd.DataFrame]
        保留共同部分后的结果
    """
    dfs = [i.T for i in dfs]
    res = []
    for i, df in enumerate(dfs):
        others = dfs[:i] + dfs[i + 1 :]

        for other in others:
            df = df[df.index.isin(other.index)]
        res.append(df.T)
    return res


def same_index(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """保留多个dataframe共同index的部分

    Parameters
    ----------
    dfs : List[pd.DataFrame]
        多个dataframe

    Returns
    -------
    List[pd.DataFrame]
        保留共同部分后的结果
    """
    res = []
    for i, df in enumerate(dfs):
        others = dfs[:i] + dfs[i + 1 :]

        for other in others:
            df = df[df.index.isin(other.index)]
        res.append(df)
    return res




def ind_neutralize(factor_df, factor_name, industry_df):
    """
    Description
    ----------
    对每期因子进行行业中性化
    方法: 先用pd.get_dummies生成行业虚拟变量, 然后用带截距项回归得到残差作为因子

    Parameters
    ----------
    factor_df: pandas.DataFrame,因子值,格式为trade_date,stock_code,factor
    factor_name: str. 因子名称
    industry_df: pandas.DataFrame, 股票所属行业, 格式为trade_date,stock_code,ind_code

    Return
    ----------
    pandas.DataFrame.行业中性化后的因子数据
    """
    df = pd.merge(factor_df, industry_df, on=["date", "code"])
    g = df.groupby("date", group_keys=False)
    df = g.apply(_single_ind_neutralize, factor_name)
    df = df[["date", "code", factor_name]].copy()
    return df


def _single_ind_neutralize(df, factor_name):
    """
    Description
    ----------
    对单期因子进行行业中性化

    Parameters
    ----------
    df: pandas.DataFrame, 因子值和行业的df, 格式为trade_date,stock_code,'factor_name',dummy_ind_code
    factor_name: str. 因子名称

    Return
    ----------
    pandas.DataFrame.行业中性化后的因子数据
    """
    x = df.iloc[:, 3:].values
    y = df[factor_name].values
    X = np.hstack([np.ones((x.shape[0], 1)), x])
    # 计算回归残差
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    df[factor_name] = y - X @ beta
    return df