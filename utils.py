import os
import pandas as pd
import numpy as np
import os
os.chdir('/Users/lihaohan/Alpha_Research')
from decorators import do_on_dfs
from functools import reduce
import mpire
from typing import Callable, Union, Dict, List, Tuple
import matplotlib.pyplot as plt
import datetime
import scipy.stats as ss
import tqdm.auto
import joblib

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




@do_on_dfs
def add_suffix(code: str) -> str:
    """给股票代码加上后缀

    Params
    ----------
    code : str
        纯数字组成的字符串类型的股票代码,如000001

    Returns
    -------
    str
        添加完后缀后的股票代码, 如000001.SZ
    """
    if not isinstance(code, str):
        code = str(code)
    if len(code) < 6:
        code = "0" * (6 - len(code)) + code
    if code.startswith("0") or code.startswith("3"):
        code = ".".join([code, "SZ"])
    elif code.startswith("6"):
        code = ".".join([code, "SH"])
    elif code.startswith("8"):
        code = ".".join([code, "BJ"])
    return code

@do_on_dfs
def get_value(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """很多因子计算时，会一次性生成很多值，使用时只取出一个值

    Params
    ----------
    df : pd.DataFrame
        每个value是一个列表或元组的pd.DataFrame
    n : int
        取第n个值

    Returns
    -------
    `pd.DataFrame`
        仅有第n个值构成的pd.DataFrame
    """

    def get_value_single(x, n):
        try:
            return x[n]
        except Exception:
            return np.nan

    df = df.applymap(lambda x: get_value_single(x, n))
    return df


def merge_many(
    dfs: List[pd.DataFrame], names: list = None, how: str = "outer"
) -> pd.DataFrame:
    """将多个宽dataframe依据columns和index, 拼接在一起, 拼成一个长dataframe

    Params
    ----------
    dfs : List[pd.DataFrame]
        将所有要拼接的宽表放在一个列表里
    names : list, optional
        拼接后，每一列宽表对应的名字, by default None
    how : str, optional
        拼接的方式, by default 'outer'

    Returns
    -------
    pd.DataFrame
        拼接后的dataframe
    """
    num = len(dfs)
    if names is None:
        names = [f"fac{i+1}" for i in range(num)]
    dfs = [i.stack().reset_index() for i in dfs]
    dfs = [i.rename(columns={list(i.columns)[-1]: j}) for i, j in zip(dfs, names)]
    dfs = [
        i.rename(columns={list(i.columns)[-2]: "code", list(i.columns)[0]: "date"})
        for i in dfs
    ]
    df = reduce(lambda x, y: pd.merge(x, y, on=["date", "code"], how=how), dfs)
    return df

@do_on_dfs
def drop_duplicates_index(new: pd.DataFrame) -> pd.DataFrame:
    """对dataframe依照其index进行去重, 并保留最上面的行

    Parames
    ----------
    new : pd.DataFrame
        要去重的dataframe

    Returns
    -------
    pd.DataFrame
        去重后的dataframe
    """
    pri_name = new.index.name
    new = new.reset_index()
    new = new.rename(
        columns={
            list(new.columns)[0]: "tmp_name_for_this_function_never_same_to_others"
        }
    )
    new = new.drop_duplicates(
        subset=["tmp_name_for_this_function_never_same_to_others"], keep="first"
    )
    new = new.set_index("tmp_name_for_this_function_never_same_to_others")
    if pri_name == "tmp_name_for_this_function_never_same_to_others":
        new.index.name = "date"
    else:
        new.index.name = pri_name
    return new

def select_max(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """两个columns与index完全相同的df, 每个值都挑出较大值

    Parameters
    ----------
    df1 : pd.DataFrame
        第一个df
    df2 : pd.DataFrame
        第二个df

    Returns
    -------
    `pd.DataFrame`
        两个df每个value中的较大者
    """
    return (df1 + df2 + np.abs(df1 - df2)) / 2

def select_min(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """两个columns与index完全相同的df, 每个值都挑出较小值

    Parameters
    ----------
    df1 : pd.DataFrame
        第一个df
    df2 : pd.DataFrame
        第二个df

    Returns
    -------
    `pd.DataFrame`
        两个df每个value中的较小者
    """
    return (df1 + df2 - np.abs(df1 - df2)) / 2

@do_on_dfs
def debj(df: pd.DataFrame) -> pd.DataFrame:
    """去除因子中的北交所数据

    Parameters
    ----------
    df : pd.DataFrame
        包含北交所的因子dataframe, index是时间, columns是股票代码

    Returns
    -------
    pd.DataFrame
        去除北交所股票的因子dataframe
    """
    df = df[[i for i in list(df.columns) if i[0] in ["0", "3", "6"]]]
    return df

@do_on_dfs
def detect_nan(df: pd.DataFrame) -> bool:
    """检查一个pd.DataFrame中是否存在空值

    Parameters
    ----------
    df : pd.DataFrame
        待检查的pd.DataFrame

    Returns
    -------
    `bool`
        检查结果, 有空值为True, 否则为False
    """
    x = df.isna() + 0
    if x.sum().sum():
        print("存在空值")
        return True
    else:
        print("不存在空值")
        return False


@do_on_dfs
def get_abs(df: pd.DataFrame, quantile: float = None, square: bool = 0) -> pd.DataFrame:
    """均值距离化：计算因子与截面均值的距离

    Parameters
    ----------
    df : pd.DataFrame
        未均值距离化的因子, index为时间, columns为股票代码
    quantile : bool, optional
        为1则计算到某个分位点的距离, by default None
    square : bool, optional
        为1则计算距离的平方, by default 0

    Returns
    -------
    `pd.DataFrame`
        均值距离化之后的因子值
    """
    if not square:
        if quantile is not None:
            return np.abs((df.T - df.T.quantile(quantile)).T)
        else:
            return np.abs((df.T - df.T.mean()).T)
    else:
        if quantile is not None:
            return ((df.T - df.T.quantile(quantile)).T) ** 2
        else:
            return ((df.T - df.T.mean()).T) ** 2

@do_on_dfs
def get_normal(df: pd.DataFrame) -> pd.DataFrame:
    """将因子横截面正态化

    Parameters
    ----------
    df : pd.DataFrame
        原始因子, index是时间, columns是股票代码

    Returns
    -------
    `pd.DataFrame`
        每个横截面都呈现正态分布的因子
    """
    df = df.replace(0, np.nan)
    df = df.T.apply(lambda x: ss.boxcox(x)[0]).T
    return df


@do_on_dfs
def coin_reverse(
    ret20: pd.DataFrame, vol20: pd.DataFrame, mean: bool = 1, positive_negtive: bool = 0
) -> pd.DataFrame:
    """球队硬币法: 根据vol20的大小, 翻转一半ret20, 把vol20较大的部分, 给ret20添加负号

    Parameters
    ----------
    ret20 : pd.DataFrame
        要被翻转的因子, index是时间, columns是股票代码
    vol20 : pd.DataFrame
        翻转的依据, index是时间, columns是股票代码
    mean : bool, optional
        为1则以是否大于截面均值为标准翻转, 否则以是否大于截面中位数为标准, by default 1
    positive_negtive : bool, optional
        是否截面上正负值的两部分，各翻转一半, by default 0

    Returns
    -------
    `pd.DataFrame`
        翻转后的因子值
    """
    if positive_negtive:
        if not mean:
            down20 = np.sign(ret20)
            down20 = down20.replace(1, np.nan)
            down20 = down20.replace(-1, 1)

            vol20_down = down20 * vol20
            vol20_down = (vol20_down.T - vol20_down.T.median()).T
            vol20_down = np.sign(vol20_down)
            ret20_down = ret20[ret20 < 0]
            ret20_down = vol20_down * ret20_down

            up20 = np.sign(ret20)
            up20 = up20.replace(-1, np.nan)

            vol20_up = up20 * vol20
            vol20_up = (vol20_up.T - vol20_up.T.median()).T
            vol20_up = np.sign(vol20_up)
            ret20_up = ret20[ret20 > 0]
            ret20_up = vol20_up * ret20_up

            ret20_up = ret20_up.replace(np.nan, 0)
            ret20_down = ret20_down.replace(np.nan, 0)
            new_ret20 = ret20_up + ret20_down
            new_ret20_tr = new_ret20.replace(0, np.nan)
            return new_ret20_tr
        else:
            down20 = np.sign(ret20)
            down20 = down20.replace(1, np.nan)
            down20 = down20.replace(-1, 1)

            vol20_down = down20 * vol20
            vol20_down = (vol20_down.T - vol20_down.T.mean()).T
            vol20_down = np.sign(vol20_down)
            ret20_down = ret20[ret20 < 0]
            ret20_down = vol20_down * ret20_down

            up20 = np.sign(ret20)
            up20 = up20.replace(-1, np.nan)

            vol20_up = up20 * vol20
            vol20_up = (vol20_up.T - vol20_up.T.mean()).T
            vol20_up = np.sign(vol20_up)
            ret20_up = ret20[ret20 > 0]
            ret20_up = vol20_up * ret20_up

            ret20_up = ret20_up.replace(np.nan, 0)
            ret20_down = ret20_down.replace(np.nan, 0)
            new_ret20 = ret20_up + ret20_down
            new_ret20_tr = new_ret20.replace(0, np.nan)
            return new_ret20_tr
    else:
        if not mean:
            vol20_dummy = np.sign((vol20.T - vol20.T.median()).T)
            ret20 = ret20 * vol20_dummy
            return ret20
        else:
            vol20_dummy = np.sign((vol20.T - vol20.T.mean()).T)
            ret20 = ret20 * vol20_dummy
            return ret20
        

@do_on_dfs
def multidfs_to_one(*args: list) -> pd.DataFrame:
    """很多个df, 各有一部分, 其余位置都是空，
    想把各自df有值的部分保留, 都没有值的部分继续设为空

    Returns
    -------
    `pd.DataFrame`
        合并后的df
    """
    dfs = [i.fillna(0) for i in args]
    background = np.sign(np.abs(np.sign(sum(dfs))) + 1).replace(1, 0)
    dfs = [(i + background).fillna(0) for i in dfs]
    df_nans = [i.isna() for i in dfs]
    nan = reduce(lambda x, y: x * y, df_nans)
    nan = nan.replace(1, np.nan)
    nan = nan.replace(0, 1)
    df_final = sum(dfs) * nan
    return df_final


def calc_exp_list(window: int, half_life: int) -> np.ndarray:
    """生成半衰序列

    Parameters
    ----------
    window : int
        窗口期
    half_life : int
        半衰期

    Returns
    -------
    `np.ndarray`
        半衰序列
    """
    exp_wt = np.asarray([0.5 ** (1 / half_life)] * window) ** np.arange(window)
    return exp_wt[::-1] / np.sum(exp_wt)

def calcWeightedStd(series: pd.Series, weights: Union[pd.Series, np.ndarray]) -> float:
    """计算半衰加权标准差

    Parameters
    ----------
    series : pd.Series
        目标序列
    weights : Union[pd.Series,np.ndarray]
        权重序列

    Returns
    -------
    `float`
        半衰加权标准差
    """
    weights /= np.sum(weights)
    return np.sqrt(np.sum((series - np.mean(series)) ** 2 * weights))


def get_list_std(delta_sts: List[pd.DataFrame]) -> pd.DataFrame:
    """同一天多个因子，计算这些因子在当天的标准差

    Parameters
    ----------
    delta_sts : List[pd.DataFrame]
        多个因子构成的list, 每个因子index为时间, columns为股票代码

    Returns
    -------
    `pd.DataFrame`
        每天每只股票多个因子的标准差
    """
    delta_sts_mean = sum(delta_sts) / len(delta_sts)
    delta_sts_std = [(i - delta_sts_mean) ** 2 for i in delta_sts]
    delta_sts_std = sum(delta_sts_std)
    delta_sts_std = delta_sts_std**0.5 / len(delta_sts) ** 0.5
    return delta_sts_std


def get_list_std_weighted(delta_sts: List[pd.DataFrame], weights: list) -> pd.DataFrame:
    """对多个df对应位置上的值求加权标准差

    Parameters
    ----------
    delta_sts : List[pd.DataFrame]
        多个dataframe
    weights : list
        权重序列

    Returns
    -------
    pd.DataFrame
        标准差序列
    """
    weights = [i / sum(weights) for i in weights]
    delta_sts_mean = sum(delta_sts) / len(delta_sts)
    delta_sts_std = [(i - delta_sts_mean) ** 2 for i in delta_sts]
    delta_sts_std = sum([i * j for i, j in zip(delta_sts_std, weights)])
    return delta_sts_std**0.5


@do_on_dfs
def to_group(df: pd.DataFrame, group: int = 10) -> pd.DataFrame:
    """把一个index为时间, columns为code的df, 每个截面上的值, 按照排序分为group组, 将值改为组号, 从0开始

    Parameters
    ----------
    df : pd.DataFrame
        要改为组号的df
    group : int, optional
        分为多少组, by default 10

    Returns
    -------
    pd.DataFrame
        组号组成的dataframe
    """
    df = df.T.apply(lambda x: pd.qcut(x, group, labels=False, duplicates="drop")).T
    return df

def zip_many_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """将多个dataframe, 拼在一起, 相同index和columns指向的那个values, 变为多个dataframe的值的列表
    通常用于存储整合分钟数据计算的因子值

    Parameters
    ----------
    dfs : List[pd.DataFrame]
        多个dataframe, 每一个的values都是float形式

    Returns
    -------
    pd.DataFrame
        整合后的dataframe, 每一个values都是list的形式
    """
    df = merge_many(dfs)
    cols = [df[f"fac{i}"] for i in range(1, len(dfs) + 1)]
    df = df.assign(fac=pd.Series(zip(*cols)))
    df = df.pivot(index="date", columns="code", values="fac")
    return df

def get_values(df: pd.DataFrame,n_jobs:int=40) -> List[pd.DataFrame]:
    """从一个values为列表的dataframe中, 一次性取出所有值, 分别设置为一个dataframe, 并依照顺序存储在列表中

    Parameters
    ----------
    df : pd.DataFrame
        一个values为list的dataframe

    Returns
    -------
    List[pd.DataFrame]
        多个dataframe, 每一个的values都是float形式
    """
    d = df.dropna(how="all", axis=1)
    d = d.iloc[:, 0].dropna()
    num = len(d.iloc[0])
    if n_jobs>1:
        facs=joblib.Parallel(n_jobs=40)(joblib.delayed(get_value)(df, x) for x in tqdm.auto.tqdm(list(range(num))))
    else:
        facs = list(map(lambda x: get_value(df, x), range(num)))

    return facs

def judge_factor_by_third(
    fac1: pd.DataFrame, fac2: pd.DataFrame, judge: Union[pd.DataFrame, pd.Series]
) -> pd.DataFrame:
    """对于fac1和fac2两个因子，依据judge这个series或dataframe进行判断，
    judge可能为全市场的某个时序指标，也可能是每个股票各一个的指标，
    如果judge这一期的值大于0，则取fac1的值，小于0则取fac2的值

    Parameters
    ----------
    fac1 : pd.DataFrame
        因子1, index为时间, columns为股票代码, values为因子值
    fac2 : pd.DataFrame
        因子2, index为时间, columns为股票代码, values为因子值
    judge : Union[pd.DataFrame,pd.Series]
        市场指标或个股指标, 为市场指标时, 则输入series形式, index为时间, values为指标值
        为个股指标时, 则输入dataframe形式, index为时间, columns为股票代码, values为因子值

    Returns
    -------
    pd.DataFrame
        合成后的因子值, index为时间, columns为股票代码, values为因子值
    """
    if isinstance(judge, pd.Series):
        judge = pd.DataFrame(
            {k: list(judge) for k in list(fac1.columns)}, index=judge.index
        )
    s1 = (judge > 0) + 0
    s2 = (judge < 0) + 0
    fac1 = fac1 * s1
    fac2 = fac2 * s2
    fac = fac1 + fac2
    have = np.sign(fac1.abs() + 1)
    return fac * have
