import numpy as np
import pandas as pd
import os
os.chdir('/Users/lihaohan/Alpha_Research')
from Data.utils import DButils
from config import *
from cachier import cachier
from typing import Union, List
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

@cachier()
def read_sw_industry():
    table_name = DB_NAME['sw']
    sql = 'select l1code, ts_code, in_date, out_date from {}'.format(table_name)
    db.cursor.execute(sql)
    sw_industry = pd.DataFrame(db.cursor.fetchall())
    sw_industry.columns = ['sw_ind', 'code', 'in_date', 'out_date']
    return sw_industry



def read_Barra_factor(
        dividend: bool = 0,
        growth: bool = 0,
        liquidity: bool = 0,
        momentum: bool = 0,
        quality: bool = 0,
        size: bool = 0,
        value: bool = 0,
        volatility: bool = 0,
):
    if dividend:
        factor = pd.read_pickle(os.path.join(BARRA_FACTOR_PATH, 'Dividend.pkl'))
    elif growth:
        factor = pd.read_pickle(os.path.join(BARRA_FACTOR_PATH, 'Growth.pkl'))
    elif liquidity:
        factor = pd.read_pickle(os.path.join(BARRA_FACTOR_PATH, 'Liquidity.pkl'))
    elif momentum:
        factor = pd.read_pickle(os.path.join(BARRA_FACTOR_PATH, 'Momentum.pkl'))
    elif quality:
        factor = pd.read_pickle(os.path.join(BARRA_FACTOR_PATH, 'Quality.pkl'))
    elif size:
        factor = pd.read_pickle(os.path.join(BARRA_FACTOR_PATH, 'Size.pkl'))
    elif value:
        factor = pd.read_pickle(os.path.join(BARRA_FACTOR_PATH, 'Value.pkl'))
    elif volatility:
        factor = pd.read_pickle(os.path.join(BARRA_FACTOR_PATH, 'Volatility.pkl'))
    else:
        raise IOError('总得读点什么吧')

@cachier()
def read_filter_con(
    st: bool = 0,
    suspend: bool = 0,
    limit: bool = 0,
    start_date = DATE1,
    end_date = DATE2,
):
    if st:
        table_name = DB_NAME['st']
        sql = 'select * from {} where date >= %s and date <= %s'.format(table_name)
    elif suspend:
        table_name = DB_NAME['suspend']
        sql = 'select * from {} where date >= %s and date <= %s'.format(table_name)
    elif limit:
        table_name = DB_NAME['up_down_limit']
        sql = 'select * from {} where date >= %s and date <= %s'.format(table_name)
    else:
        raise IOError('总得读点什么吧')
    
    params = (start_date, end_date)
    db.cursor.execute(sql, params)
    df = pd.DataFrame(db.cursor.fetchall(), columns=[i[0] for i in db.cursor.description])
    df['date'] = df['date'].apply(pd.Timestamp)
    df.sort_values(by=['code', 'date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
    
    

def get_index_data(index: str, 
                   start_date: str='2020-05-23', 
                   end_date: str='2025-05-23',
                   fields: List[str]=None):
    """获取宽基指数在日期范围内的行情数据

    Args:
    index: str
        中证500: '000905'
        中证1000: '000852'
        沪深300: '000300'
    start_date: '2020-05-23'
    end_date: '2025-05-23'
    fields: List[str]
    """
    try:
        fields = "*" if not fields else f", ".join(fields)
        sql = f"""
            SELECT {fields}
            FROM index_daily_pv
            WHERE code = %s
            AND date >= %s 
            AND date <= %s
            ORDER BY date ASC
            """
        with db.cursor() as cursor:  
            cursor.execute(sql, (index, start_date, end_date))
            cols_name = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            df = pd.DataFrame(data, columns=cols_name)
            return df
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

    

