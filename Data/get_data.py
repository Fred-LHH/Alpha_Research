import numpy as np
import pandas as pd
import os
os.chdir('/Users/lihaohan/Alpha_Research')
from Data.utils import DButils
from config import *
from cachier import cachier
from typing import Union
import tushare as ts
ts.set_token(TOKEN)
pro = ts.pro_api()

db = DButils()

def read_trade_dates(start_date: str = DATE1,
                     end_date: str = DATE2):
    start_date = start_date.replace('-', '')
    end_date = end_date.replace('-', '')
    trade_dates = pro.trade_cal(start_date=start_date, end_date=end_date)
    trade_dates = trade_dates[trade_dates['is_open'] == 1]
    trade_dates = trade_dates['cal_date'].values
    return sorted(trade_dates)

@cachier()
def read_daily(
    open: bool = 0,
    close: bool = 0,
    high: bool = 0,
    low: bool = 0,
    volume: bool = 0,
    amount: bool = 0,
    circ_mv: bool = 0,
    total_mv: bool = 0,
    ret: bool = 0,
    limit_up: bool = 0,
    limit_down: bool = 0,
    adjust: bool = 1,
    turnover: bool = 0,
    start_date: str = '2018-01-01',
    end_date: str = '2025-02-01',
    freq: str = 'W',
) -> pd.DataFrame:
    '''
    默认读取复权数据, 在open, close....中选择一个参数指定为1
    '''

    if adjust:
        if any(x == 1 for x in [limit_up, limit_down]):
            raise IOError('涨跌停价格数据不在后复权数据表中, 请读取不复权数据表')
        table_name = DB_NAME['hfq']
        if open:
            sql = 'select date, code, open from {} where date >= %s and date <= %s'.format(table_name)
        elif close:
            sql = 'select date, code, close from {} where date >= %s and date <= %s'.format(table_name)
        elif high:
            sql = 'select date, code, high from {} where date >= %s and date <= %s'.format(table_name)
        elif low:
            sql = 'select date, code, low from {} where date >= %s and date <= %s'.format(table_name)
        elif volume:
            sql = 'select date, code, vol from {} where date >= %s and date <= %s'.format(table_name)
        elif amount:
            sql = 'select date, code, amount from {} where date >= %s and date <= %s'.format(table_name)
        elif circ_mv:
            sql = 'select date, code, circ_mv from {} where date >= %s and date <= %s'.format(table_name)
        elif total_mv:
            sql = 'select date, code, total_mv from {} where date >= %s and date <= %s'.format(table_name)
        elif ret:
            sql = 'select date, code, pct_chg from {} where date >= %s and date <= %s'.format(table_name)
        elif turnover:
            sql = 'select date, code, turnover from {} where date >= %s and date <= %s'.format(table_name)
        else:
            raise IOError('总得读点什么吧')
        
    else:
        if turnover:
            raise IOError('换手率数据不在不复权数据表中, 请读取后复权数据表')
        table_name = DB_NAME['bfq']
        if open:
            sql = 'select date, code, open from {} where date >= %s and date <= %s'.format(table_name)
        elif close:
            sql = 'select date, code, close from {} where date >= %s and date <= %s'.format(table_name)
        elif high:
            sql = 'select date, code, high from {} where date >= %s and date <= %s'.format(table_name)
        elif low:
            sql = 'select date, code, low from {} where date >= %s and date <= %s'.format(table_name)
        elif volume:
            sql = 'select date, code, vol from {} where date >= %s and date <= %s'.format(table_name)
        elif amount:
            sql = 'select date, code, amount from {} where date >= %s and date <= %s'.format(table_name)
        elif circ_mv:
            sql = 'select date, code, circ_mv from {} where date >= %s and date <= %s'.format(table_name)
        elif total_mv:
            sql = 'select date, code, total_mv from {} where date >= %s and date <= %s'.format(table_name)
        elif ret:
            sql = 'select date, code, pct_chg from {} where date >= %s and date <= %s'.format(table_name)
        elif limit_up:
           sql = 'select date, code, limit_up from {} where date >= %s and date <= %s'.format(table_name)
        elif limit_down:
            sql = 'select date, code, limit_down from {} where date >= %s and date <= %s'.format(table_name)
        else:
            raise IOError('总得读点什么吧')
        
    params = (start_date, end_date)
    db.cursor.execute(sql, params)

    df = pd.DataFrame(db.cursor.fetchall(), columns=[i[0] for i in db.cursor.description])
    df['date'] = df['date'].apply(pd.Timestamp)
    df = df.pivot(index='date', columns='code', values=df.columns[-1])

    trade_dates = read_trade_dates(start_date, end_date)
    trade_dates = pd.to_datetime(trade_dates)
    df = df[df.index.isin(trade_dates)]
    if freq != 'D':
        if open == 1:
            df = df.resample(freq).first()
        else:
            df = df.resample(freq).last()

    return df.dropna(how='all')

@cachier()
def read_market(
    open: bool = 0,
    close: bool = 0,
    high: bool = 0,
    low: bool = 0,
    volume: bool = 0,
    amount: bool = 0,
    circ_mv: bool = 0,
    total_mv: bool = 0,
    ret: bool = 0,
    start_date: str = DATE1,
    end_date: str = DATE2,
    index: str = '000010',
    freq: str = 'W',
) -> Union[pd.DataFrame, pd.Series]:
    
    table_name = DB_NAME['index_pv']
    if open:
        sql = 'select date, code, open from {} where date >= %s and date <= %s and code = %s'.format(table_name)
    elif close:
        sql = 'select date, code, close from {} where date >= %s and date <= %s and code = %s'.format(table_name)
    elif high:
        sql = 'select date, code, high from {} where date >= %s and date <= %s and code = %s'.format(table_name)
    elif low:
        sql = 'select date, code, low from {} where date >= %s and date <= %s and code = %s'.format(table_name)
    elif volume:
        sql = 'select date, code, vol from {} where date >= %s and date <= %s and code = %s'.format(table_name)
    elif amount:
        sql = 'select date, code, amount from {} where date >= %s and date <= %s and code = %s'.format(table_name)
    elif circ_mv:
        sql = 'select date, code, circ_mv from {} where date >= %s and date <= %s and code = %s'.format(table_name)
    elif total_mv:
        sql = 'select date, code, total_mv from {} where date >= %s and date <= %s and code = %s'.format(table_name)
    elif ret:
        sql = 'select date, code, pct_chg from {} where date >= %s and date <= %s and code = %s'.format(table_name)
    else:
        raise IOError('总得读点什么吧')
    
    params = (start_date, end_date, index)
    db.cursor.execute(sql, params)
    df = pd.DataFrame(db.cursor.fetchall(), columns=[i[0] for i in db.cursor.description])
    df['date'] = df['date'].apply(pd.Timestamp)
    df = df.pivot(index='date', columns='code', values=df.columns[-1])
    trade_dates = read_trade_dates(start_date, end_date)
    trade_dates = pd.to_datetime(trade_dates)
    df = df[df.index.isin(trade_dates)]
    if freq != 'D':
        df = df.resample(freq).last()
    return df.dropna(how='all')