from datetime import *
import os
os.chdir('/Users/lihaohan/Alpha_Research')
from Data.utils import DButils
from Data.tushare.api import *
import pandas as pd
import numpy as np
from Data.logger import *
from config import *

api = TSDataFetcher()
db = DButils()


class Engine():

    def __init__(self):
        pass

    def create_table(self):
        pass

    def check_update_date(self):
        '''
        确认存量数据情况
        '''
        current_date = datetime.now().strftime('%Y%m%d')
        trade_dates = api.trade_dates(start_date=START_DATE, end_date=current_date)
        db.cursor.execute('select max(date) from %s ' % self.table_name)
        last_sql_date = db.cursor.fetchone()[0]
        if last_sql_date is not None:
            last_update_date = last_sql_date
        else:
            return START_DATE
        
        for i in range(0, len(trade_dates['date'])):
            # trade_dates是倒序的
            date = trade_dates.iloc[i]['date']
            if date > last_update_date:
                continue
            api_df = api.query(api_name='daily_basic', start_date=date, end_date=date)
            db.cursor.execute(
                "select count(distinct code) from %s where date = '%s' " % (self.table_name, date))
            db_counts = db.cursor.fetchone()[0]
            api_counts = len(api_df['ts_code'])
            if db_counts >= api_counts:
                self.logger.info(
                    'record in date ' + date + ' is the last proper update time in table ' + self.table_name)
                return date
            else:
                self.logger.info('record in date ' + date + ' is not proper in table ' + self.table_name)
        return START_DATE

    def get_stock_list(self):
        '''
        #获取上市的的股票表
        '''
        stocks = api.all_stocks()
        stocks = stocks[stocks['list_status'] == 'L']['symbol'].tolist()
        return stocks

    def check_open_day(self,date) -> bool:
        '''
        #判断当日是否是交易日
        '''
        df = api.query('trade_cal', start_date=date, end_date=date)
        if df is None:
            return False
        if df.empty:
            return False
        elif df.iloc[0]['is_open'] == 0:
            return False
        else:
            return True

    def clear_data(self, start_dt=START_DATE, stocks=None):
        '''
        #清除某日期前的数据,如果股票列表为None,则全部股票清除
        '''
        sql = ""
        if start_dt is not None:
            dt_suffix = " and trade_date < '%s'" % start_dt
        else:
            dt_suffix = ''
        if stocks is not None:
            stocks_list = "', '".join(stocks)
            sql = "delete from %s where code in ('%s') %s" % (self.table_name, stocks_list, dt_suffix)
        elif start_dt is not None:
            sql = "delete from %s where trade_date < '%s'" % (self.table_name, start_dt)
        else:
            sql = "drop table %s" % self.table_name
        db.cursor.execute(sql)
        self.commit = db.db.commit()
        self.logger.info('clear the table %s where sql is : %s' % (self.table_name, sql))
        self.create_table()
        return 1
    
    def update_data(self, start_dt, stocks=None):
        '''
        #更新数据 从start_dt 开始
        '''
        pass

    def update(self, is_inital=False):
        self.create_table()
        if not is_inital:
            last_trade_date = self.check_update_date()
            self.clear_data(last_trade_date)
            self.update_data(last_trade_date=last_trade_date)
        else:
            self.clear_data(start_dt=None)
            self.update_data(last_trade_date=START_DATE)

class TushareEngineBFQ(Engine):
    '''
    不复权股票日行情数据
    '''

    def __init__(self):
        self.table_name = DB_NAME.get('daily_pv')
        self.logger = logger.bfqlogger
        super().__init__()

    def create_table(self):
        sql_comm = "create table if not exists %s " \
                   "( id int not null auto_increment primary key," % (self.table_name)
        # 获取列名
        df = api.pro_bar(ts_code='000001.SZ', asset='E',
                         adj=None, freq='D', start_date=DATE1, end_date=DATE2,
                         factors=['vr', 'tor'], adjfactor=True)
        # 改变列名
        df.rename(columns={'change': 'close_chg'}, inplace=True)
        cols = df.columns.tolist()
        for ctx in range(0, len(cols)):
            col = cols[ctx]
            if isinstance(df[col].iloc[0], str):
                sql_comm += col + " varchar(40), "
            elif isinstance(df[col].iloc[0], float):
                sql_comm += col + " decimal(20, 3), "
        sql_comm += 'INDEX trade_date_index(trade_date), '
        sql_comm += 'INDEX ts_stock_index(ts_code), '
        sql_comm = sql_comm[0: len(sql_comm) - 2]
        sql_comm += ") engine=innodb default charset=utf8mb4;"
        db.cursor.execute(sql_comm)
        self.logger.info('create the table %s if not exist ' % self.table_name)
        return 1

    def update_data(self, last_trade_date=None, stocks=None):
        if last_trade_date is None:
            start_dt = START_DATE
        else:
            start_dt = (datetime.strptime(last_trade_date, '%Y%m%d') + timedelta(days=1)).strftime("%Y%m%d")
        # ---获取列名
        col_sql = 'describe %s ' % self.table_name
        db.cursor.execute(col_sql)
        cols = db.cursor.fetchall()
        if len(cols) == 0:
            return 0
        # ---构建插入sql
        sql_insert = "INSERT INTO %s ( " % self.table_name
        sql_value = "VALUES ( "
        for c in cols:
            if c[0] == 'id':
                continue
            sql_insert += c[0] + ", "
            if c[1] == 'int':
                sql_value += "'%d', "
            elif c[1] == 'decimal(20,3)':
                sql_value += "'%.3f', "
            elif c[1] == 'varchar(40)':
                sql_value += "'%s', "
        sql_insert = sql_insert[0: len(sql_insert) - 2]
        sql_insert += " )"
        sql_value = sql_value[0: len(sql_value) - 2]
        sql_value += " )"
        end_dt = datetime.now().strftime('%Y%m%d')
        if start_dt == end_dt:
            if not self.check_open_day(start_dt):
                self.logger.info("date %s is closed" % start_dt)
                return
        # ---获取数据
        if stocks is None:
            stocks = self.get_stock_list()
        for s in stocks:
            df = api.pro_bar(ts_code=s, asset='E',
                             adj=None, freq='D', factors=['vr', 'tor'], adjfactor=True,
                             start_date=start_dt, end_date=end_dt)
            if df is None:
                self.logger.info('updating stock: %s from %s to %s, the data is None!'% (s,start_dt,end_dt))
            else:
                # ---改变列名
                self.logger.info('updating stock: %s from %s to %s' % (s,start_dt,end_dt))
                df.rename(columns={'change': 'close_chg'}, inplace=True)
                df.drop_duplicates(inplace=True)
                df = df.sort_values(by=['trade_date'], ascending=False)
                df.reset_index(inplace=True, drop=True)
                c_len = df.shape[0]
                for jtx in range(0, c_len):
                    resu0 = list(df.iloc[c_len - 1 - jtx])
                    resu = []
                    for k in range(len(resu0)):
                        if isinstance(resu0[k], str):
                            resu.append(resu0[k])
                        elif isinstance(resu0[k], float):
                            if np.isnan(resu0[k]):
                                resu.append(-1)
                            else:
                                resu.append(resu0[k])
                        elif resu0[k] == None:
                            resu.append(-1)
                    try:
                        sql_impl = sql_insert + sql_value
                        sql_impl = sql_impl % tuple(resu)
                        db.cursor.execute(sql_impl)
                        db.db.commit()
                    except Exception as err:
                        self.logger.error(err)
                        continue
        self.logger.info('bfq data is fully updated')



