from datetime import *
import os
os.chdir('/Users/lihaohan/Alpha_Research')
from Data.utils import DButils
from Data.tushare.api import *
import pandas as pd
import numpy as np
from Data.logger import *
from config import *
from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://root:walhh123@localhost/Stock?charset=utf8')

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
            last_update_date = last_update_date.replace('-', '')
        else:
            return START_DATE
        
        for i in range(0, len(trade_dates['date'])):
            # trade_dates是倒序的
            date = trade_dates.iloc[i]['date']
            if date > last_update_date:
                continue
            sql_date = date[:4] + '-' + date[4:6] + '-' + date[6:]
            api_df = api.query(api_name='daily_basic', start_date=date, end_date=date)
            db.cursor.execute(
                "select count(distinct code) from %s where date = '%s' " % (self.table_name, sql_date))
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
        date = date.replace('-', '')
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
            #self.clear_data(last_trade_date)
            self.update_data(last_trade_date=last_trade_date)
        else:
            #self.clear_data(start_dt=None)
            self.update_data(last_trade_date=START_DATE)


class BFQPVEngine(Engine):

    def __init__(self):
        self.table_name = DB_NAME['bfq']
        self.logger = logger.bfqlogger
        self.path = os.path.join(LOCAL_DATA_PATH, 'bfq')
        super().__init__()

    def create_table(self):
        sql_comm = "create table if not exists %s " \
                   "( id int not null auto_increment primary key," % (self.table_name)
        for col, type in bfq_table_cols.items():
            sql_comm += col + ' ' + type + ','
        sql_comm = sql_comm[0:-1] + ")"
        db.cursor.execute(sql_comm)
        self.logger.info('create the table %s if not exist ' % self.table_name)
        return 1
        
    
    def update_data(self, last_trade_date=None, stocks=None):
        if last_trade_date is None:
            start_dt = START_DATE
        else:
            start_dt = (datetime.strptime(last_trade_date, '%Y%m%d') + timedelta(days=1)).strftime("%Y-%m-%d")
        # ---获取列名
        col_sql = 'describe %s ' % self.table_name
        db.cursor.execute(col_sql)
        cols = db.cursor.fetchall()
        if len(cols) == 0:
            return 0
        
        end_dt = datetime.now().strftime('%Y-%m-%d')
        if start_dt == end_dt:
            if not self.check_open_day(start_dt):
                self.logger.info("date %s is closed" % start_dt)
                return 
        
        files = os.listdir(self.path)
        files = [f for f in files if f.endswith('.xlsx')]
        df = pd.DataFrame()
        for file in files:
            data = pd.read_excel(os.path.join(self.path, file))
            data = data.drop(columns=['Dretwd', 'Dretnd', 'Adjprcwd', 'Adjprcnd', 'Capchgdt', 'Ahshrtrd_D', 'Ahvaltrd_D'])
            data.columns = bfq_columns
            data = data.iloc[2:, :]
            data.reset_index(drop=True, inplace=True)
            df = pd.concat([df, data])
        df.drop_duplicates(subset=['date', 'code'], keep='last', inplace=True)
        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
        try:
            with engine.begin() as conn:
                data.to_sql(
                    name=self.table_name,
                    con=conn,
                    if_exists='append',
                    index=False
                    )
            self.logger.info('update the table %s from %s to %s' % (self.table_name, start_dt, end_dt))
        except Exception as err:
            self.logger.error(err)

        
class HFQPVEngine(Engine):

    def __init__(self):
        self.table_name = DB_NAME['hfq']
        self.logger = logger.hfqlogger
        self.path = os.path.join(LOCAL_DATA_PATH, 'hfq')
        super().__init__()

    def create_table(self):
        sql_comm = "create table if not exists %s " \
                   "( id int not null auto_increment primary key," % (self.table_name)
        for col, type in bfq_table_cols.items():
            sql_comm += col + ' ' + type + ','
        sql_comm = sql_comm[0:-1] + ")"
        db.cursor.execute(sql_comm)
        self.logger.info('create the table %s if not exist ' % self.table_name)
        return 1
        
    
    def update_data(self, last_trade_date=None, stocks=None):
        if last_trade_date is None:
            start_dt = START_DATE
        else:
            start_dt = (datetime.strptime(last_trade_date, '%Y%m%d') + timedelta(days=1)).strftime("%Y-%m-%d")
        # ---获取列名
        col_sql = 'describe %s ' % self.table_name
        db.cursor.execute(col_sql)
        cols = db.cursor.fetchall()
        if len(cols) == 0:
            return 0
        
        end_dt = datetime.now().strftime('%Y-%m-%d')
        if start_dt == end_dt:
            if not self.check_open_day(start_dt):
                self.logger.info("date %s is closed" % start_dt)
                return 
        
        files = os.listdir(self.path)
        files = [f for f in files if f.endswith('.xlsx')]
        df = pd.DataFrame()
        for file in files:
            data = pd.read_excel(os.path.join(self.path, file))
            data = data.drop(columns=['ACirculatedShare', 'BCirculatedShare', 'AValue', 'BValue', 'Capchgdt'])
            data.columns = hfq_columns
            data = data.iloc[2:, :]
            data.reset_index(drop=True, inplace=True)
            df = pd.concat([df, data])
        df.drop_duplicates(subset=['date', 'code'], keep='last', inplace=True)
        df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
        try:
            with engine.begin() as conn:
                data.to_sql(
                    name=self.table_name,
                    con=conn,
                    if_exists='append',
                    index=False
                    )
            self.logger.info('update the table %s from %s to %s' % (self.table_name, start_dt, end_dt))
        except Exception as err:
            self.logger.error(err)
            
        
