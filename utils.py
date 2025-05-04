import os
import pandas as pd
import numpy as np
import os
os.chdir('/Users/lihaohan/Alpha_Research')
from decorators import do_on_dfs
from functools import reduce
import polars as pl
import mpire
from typing import Union
import datetime

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False, nb_workers=10)



do_on_dfs
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
    keep_trend: bool = 1
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
    index_list = [df.index for df in [y]+x_list]
    dates = reduce(lambda x, y: x.intersection(y), index_list)
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

