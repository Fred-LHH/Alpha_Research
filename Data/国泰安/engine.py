from datetime import *
import os
os.chdir('/Users/lihaohan/Alpha_Research')
from Data.utils import DButils
from Data.tushare.api import *
import pandas as pd
import numpy as np
from Data.logger import *
from config import *
from pathlib import Path


api = TSDataFetcher()
db = DButils()

class LocalEngine():

    def __init__(self,
                 name,
                 data_type):
        self.table_name = DB_NAME.get(name)
        self.data_path = os.path.join(LOCAL_DATA_PATH, data_type)

        if not os.path.exists(self.data_path):
            raise NotImplementedError(f'Data dir not found {self.data_path}')
    def create_table(self):
        pass

    def get_data_files(self):
        data_files = Path(self.data_path).rglob('*.xlsx')
        return [str(file.resolve()) for file in data_files]

    def get_stocks(self):
        '''
        获取单日可交易股票
        '''
        db.cursor.execute("SELECT code FROM stocks WHERE list_status = 'L'")
        list_codes = pd.DataFrame(db.cursor.fetchall(), columns=['code'])
        return list_codes

    def check_update_date(self):
        try:
            current_date = datetime.now().strftime('%Y%m%d')
            db.cursor.execute('SELECT date FROM trade_dates')
            trade_dates = db.cursor.fetchall()
            trade_dates = pd.DataFrame(trade_dates, columns=['date'])
            trade_dates.sort_values('date', ascending=False, inplace=True)
            trade_dates.reset_index(drop=True, inplace=True)

            db.cursor.execute(f'SELECT MAX(date) FROM {self.table_name}')
            last_db_date = db.cursor.fetchone()[0]
            if last_db_date is not None:
                last_update_date = last_db_date
            else:
                last_update_date = START_DATE

            for i in range(0, len(trade_dates)):
                if trade_dates['date'][i] > last_update_date:
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
        
        except Exception as e:
            self.logger.error(e)
            raise e
        
    def clear_data(self, start_dt=START_DATE, stocks=None):
        '''
        清除某日期前的数据,如果股票列表为None,则全部股票清除
        '''
        sql = ""
        if start_dt is not None:
            dt_suffix = " and date >'%s'" % start_dt
        else:
            dt_suffix = ''
        if stocks is not None:
            for s in stocks:
                sql = "delete from %s where code = '%s' " % (self.table_name, s)
                sql = sql + dt_suffix
        elif start_dt is not None:
            sql = "delete from %s where date > '%s'" % (self.table_name, start_dt)
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
            last_update_date = self.check_update_date()
            self.clear_data(last_update_date)
            self.update_data(start_dt=last_update_date)
        else:
            self.clear_data(start_dt=None)
            self.update_data(start_dt=START_DATE)

                    
        





