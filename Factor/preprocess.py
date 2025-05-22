import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from utils import *
from typing import Union, List, Tuple
from Data.get_data import read_daily
from sklearn.linear_model import LinearRegression

def neutralize(factor, factor_name, mktmv=None, industry=None):
    neu_factor = factor.copy()
    if mktmv is not None:
        neu_factor = mktmv_neutralize(neu_factor, factor_name, mktmv)
    if industry is not None:
        neu_factor = ind_neutralize(neu_factor, factor_name, industry)
    return neu_factor


def mktmv_neutralize(factor, factor_name, mktmv):

    def single_mkt_reg(group, factor_name):
        x = group["circ_mv"].values.reshape(-1, 1)
        y = group[factor_name]
        lr = LinearRegression()
        lr.fit(x, y)  
        y_predict = lr.predict(x)  # 预测
        group[factor_name] = y - y_predict
        return group

    df = pd.merge(factor, mktmv, on=["date", "code"])
    df = df.groupby("date", group_keys=False).apply(single_mkt_reg, factor_name)
    df = df.drop(columns=["circ_mv"])
    return df


def ind_neutralize(factor, factor_name, industry):

    def single_ind_neu(group, factor_name):
        x = group.iloc[:, 3:]
        y = group[factor_name]
        lr = LinearRegression()
        lr.fit(x, y)
        y_predict = lr.predict(x)
        group[factor_name] = y - y_predict
        return group


    ind_dummies = pd.get_dummies(industry["industry"], drop_first=True, prefix="ind")
    ind_new = pd.concat([industry.drop(columns=["industry"]), ind_dummies], axis=1)
    df = pd.merge(factor, ind_new, on=["date", "code"])
    df = df.groupby("date", group_keys=False).apply(single_ind_neu, factor_name)
    df = df[["date", "code", factor_name]].copy()
    return df

def run(factor, factor_name, mkt, industry):
    factor = factor.dropna(how='all')
    factor = clip(factor, mad=1, parameter=3.0)
    factor = factor.stack().reset_index()
    factor.columns = ["date", "code", factor_name]
    factor = neutralize(factor, factor_name, mkt, industry)
    factor = factor.pivot(index="date", columns="code", values=factor_name)
    factor = standardlize(factor)
    return factor
