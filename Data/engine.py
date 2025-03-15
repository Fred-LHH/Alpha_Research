import decimal
from datetime import *
import os
os.chdir('/Users/lihaohan/Alpha_Research')
from Data.utils import DButils
from api import *
import pandas as pd
import numpy as np
from logger import *
from decimal import Decimal
from config import *

api = TSDataFetcher()
db = DButils()

class TradeDates():

    def __init__(self):
        self.table_map = {
            'trade_dates': 'trade_dates'
        }
        self.logger = logger.dateslogger

    def create_table(self):
        dates_sql = """CREATE TABLE IF NOT EXISTS {} (
            id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            date VARCHAR(10)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;""".format(self.table_map['trade_dates'])
        
        db.cursor.execute(dates_sql)
        self.logger.info('create the table %s if not exist ' % self.table_map['trade_dates'])

    def check_update_date(self):
        '''
        确定存量数据情况
        '''
        current_date = datetime.now().strftime('%Y%m%d')
        trade_dates = api.query('trade_cal', start_date=START_DATE, end_date=current_date)
        trade_dates.sort_values('cal_date', ascending=False)
        trade_dates = trade_dates.loc[trade_dates['is_open'] == 1]['cal_date']
        sql = "SELECT date FROM {} ORDER BY date DESC LIMIT 1".format(self.table_map['trade_dates'])
        try:
            db.cursor.execute(sql)
            last_sql_date = db.cursor.fetchone()[0]
        except:
            last_sql_date = None
        if last_sql_date is None:
            return START_DATE
        return last_sql_date
    
    def check_open_day(self, date) -> bool:
        '''
        判断当日是否是交易日
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

    def update_data(self, last_trade_date=None):
        if last_trade_date is None:
            start_dt = START_DATE
        else:
            start_dt = (datetime.strptime(last_trade_date, '%Y%m%d') + timedelta(days=1)).strftime("%Y%m%d")

        for api_name, table_name in self.table_map.items():
            col_sql = 'describe %s ' % table_name
            db.cursor.execute(col_sql)
            cols = db.cursor.fetchall()
            if len(cols) == 0:
                return 0
            sql_insert = "INSERT INTO %s ( " % table_name
            sql_value = "VALUES ( "
            for c in cols:
                if c[0] == 'id':
                    continue
                sql_insert += c[0] + ", "
                if c[1] == 'int':
                    sql_value += "'%d', "
                elif c[1] == 'decimal(20,3)':
                    sql_value += "'%.3f', "
                elif c[1] == 'varchar(10)':
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
            get_data = getattr(api, api_name)
            res = get_data(start_date=start_dt, end_date=end_dt)
            if res is None and not res.empty:
                res.columns = 'date'
                self.logger.info('updating %s from %s to %s, the data is None!' % (table_name, start_dt, end_dt))
            else:
                self.logger.info('updating %s from %s to %s' % (table_name, start_dt, end_dt))
                res.drop_duplicates(inplace=True)
                try:
                    res = res.sort_values(by=['date'], ascending=False)
                except Exception as e:
                    pass
                res.reset_index(inplace=True, drop=True)
                c_len = res.shape[0]
                for jtx in range(0, c_len):
                    resu0 = list(res.iloc[c_len - 1 - jtx])
                    resu = []
                    for k in range(len(resu0)):
                        if isinstance(resu0[k], str):
                            resu.append(resu0[k])
                        elif isinstance(resu0[k], float):
                            if np.isnan(resu0[k]):
                                resu.append(-1)
                            else:
                                resu.append(resu0[k])
                        elif resu0[k] is None:
                            resu.append(-1)
                    try:
                        sql_impl = sql_insert + sql_value
                        sql_impl = sql_impl % tuple(resu)
                        db.cursor.execute(sql_impl)
                        db.db.commit()
                    except Exception as err:
                        self.logger.error(err)
                        continue

    def update(self, is_inital=False):
        self.create_table()
        if not is_inital:
            last_trade_date = self.check_update_date()
            self.update_data(last_trade_date=last_trade_date)
        else:
            self.update_data(last_trade_date=START_DATE)

class All_Stocks():
    def __init__(self):
        self.table_name = DB_NAME.get('code')
        self.logger = logger.stockslogger

    def create_table(self):
        stocks_sql = """CREATE TABLE IF NOT EXISTS {} (
            id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            ts_code VARCHAR(10),
            code VARCHAR(10),
            list_date VARCHAR(10),
            delist_date VARCHAR(10),
            list_status VARCHAR(10)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;""".format(self.table_name)
        
        db.cursor.execute(stocks_sql)
        self.logger.info('create the table %s if not exist ' % self.table_name)

    def update_daily(self, trade_date):
        self.logger.info(f'开始更新股票状态, 日期: {trade_date}')

        try:
            df = api.all_stocks()
            if df.empty:
                self.logger.info('api返回空数据, 跳过更新')
                return
            df.rename(columns={
                'symbol': 'code'
            }, inplace=True)
            # 全量替换
            sql = f"TRUNCATE TABLE {self.table_name};"
            db.cursor.execute(sql)
            
            holders = ', '.join(['%s'] * len(df.columns))
            columns = ', '.join(df.columns)
            sql = f"INSERT INTO {self.table_name} ({columns}) VALUES ({holders});"
            data = [tuple(row) for row in df.itertuples(index=False)]
            db.cursor.executemany(sql, data)
            db.db.commit()

            self.logger.info(f'股票状态更新完成, 日期: {trade_date}')
        except Exception as e:
            db.db.rollback()
            self.logger.error(f'股票状态更新失败, 错误: {e}', exc_info=True)
            raise e
        
    def update(self, is_inital=False):
        self.create_table()
        current_date = datetime.now().strftime('%Y%m%d')
        self.update_daily(current_date)


class Engine():

    def __init__(self, name):
        self.table_name = DB_NAME.get(name)

    def create_table(self):
        pass

    def check_update_date(self):
        '''
        确认存量数据情况
        '''
        current_date = datetime.now().strftime('%Y%m%d')
        db.cursor.execute('SELECT date FROM trade_dates')
        trade_dates = db.cursor.fetchall()
        trade_dates = [date[0] for date in trade_dates]
        trade_dates = pd.DataFrame(trade_dates, columns=['date'])
        db.cursor.execute('select max(date) from %s ' % self.table_name)
        last_sql_date = db.cursor.fetchone()[0]
        if last_sql_date is not None:
            last_update_date = last_sql_date
        else:
            return START_DATE
        
        for i in range(0, len(trade_dates['date'])):
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
        #获取全量可以交易的股票表
        '''
        db.cursor.execute('SELECT date FROM trade_dates')
        trade_dates = db.cursor.fetchall()

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
            dt_suffix = " and trade_date >'%s'" % start_dt
        else:
            dt_suffix = ''
        if stocks is not None:
            for s in stocks:
                sql = "delete from %s where ts_code = '%s' " % (self.table_name, s)
                sql = sql + dt_suffix
        elif start_dt is not None:
            sql = "delete from %s where trade_date>'%s'" % (self.table_name, start_dt)
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
    #不复权股票日行情数据
    '''

    def __init__(self):
        self.table_name = DB_NAME.get('daily_pv')
        super().__init__()

    def create_table(self):
        sql_comm = "create table if not exists %s " \
                   "( id int not null auto_increment primary key," % (self.table_name)
        # ---获取列名
        df = api.pro_bar(ts_code='000001.SZ', asset='E',
                         adj=None, freq='D', start_date=DATE1, end_date=DATE2,
                         factors=['vr', 'tor'], adjfactor=True)
        # ---改变列名
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



class TushareMysqlEngineBASIC(Engine):

    def __init__(self):
        self.table_name = DB_NAME.get('basic')
        super().__init__()

    def create_table(self):
        sql_comm = "create table if not exists %s " \
                   "( id int not null auto_increment primary key," % self.table_name
        # ---获取列名
        df = api.query(api_name='daily_basic', ts_code='000001.SZ')
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
        # ---获取数据
        if start_dt == end_dt:
            if not self.check_open_day(start_dt):
                self.logger.info("date %s is closed" % start_dt)
                return
        if stocks is None:
            stocks = self.get_stock_list()
        for s in stocks:
            df = api.query(api_name='daily_basic', ts_code=s, start_date=start_dt, end_date=end_dt)
            if len(df) == 6000:  # 最多下载6000条记录
                last_download_date = df['trade_date'].iloc[-1]
                last_download_date = (datetime.datetime.strptime(last_download_date, '%Y%m%d')
                                      - timedelta(days=1)).strftime("%Y%m%d")
                df2 = api.query(api_name='daily_basic', ts_code=s, start_date=last_download_date, end_date=end_dt)
                if len(df2) > 0:
                    df = pd.concat([df, df2], axis=0)
            if df is None:
                self.logger.info('updating stock: %s from %s to %s, the data is None!'% (s,start_dt,end_dt))
            else:
                self.logger.info('updating stock: %s from %s to %s' % (s,start_dt,end_dt))
                df.drop_duplicates(inplace=True)
                try:
                    df = df.sort_values(by=['trade_date'], ascending=False)
                except Exception as e:
                    print(df)
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
        self.logger.info('dailybaisic data is fully updated')


class TushareMysqlEngineIndex(Engine):

    def __init__(self):
        self.index = {'000001.SH',  # 上证综指
                      '399001.SZ',  # 深证成指
                      '000300.SH',  # 沪深300
                      '399006.SZ',  # 创业板指
                      '000016.SH',  # 上证50
                      '000905.SH',  # 中证500
                      '399005.SZ',  # 中小板指
                      '000010.SH'   # 上证180
                      }
        self.table_name = DB_NAME.get('index')
        super().__init__()

    def create_table(self):
        sql_comm = "create table if not exists %s " \
                   "( id int not null auto_increment primary key," % self.table_name
        # ---获取列名
        df = api.pro_bar(ts_code='000001.SH', asset='I',
                         adj='qfq', freq='D', start_date=DATE1, end_date=DATE2,
                         factors=['vr', 'tor'], adjfactor=True, ma=(5, 10, 20, 30, 60))
        # ---改变列名
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




    def check_update_date(self):
        '''
        #确认存量数据情况
        #明确历史数据到哪天是可靠的
        '''
        db.cursor.execute('select max(trade_date) from %s ' % self.table_name)
        return db.cursor.fetchone()[0]

    def update_data(self, last_trade_date=None, stocks=None):
        '''
        #指数数据全量更新
        '''
        end_dt = datetime.now().strftime('%Y%m%d')
        start_dt = START_DATE
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
        # ---获取数据
        stocks = self.index
        self.clear_data(start_dt=None)
        for s in stocks:
            df = api.pro_bar(ts_code=s, asset='I',
                             adj='qfq', freq='D', factors=['vr', 'tor'], adjfactor=True,
                             start_date=start_dt, end_date=end_dt, ma=(5, 10, 20, 30, 60))
            if df is None:
                self.logger.info('stock: ' + s + ' is Empty')
                continue          # ---改变列名
            self.logger.info('stock ' + s + ' is updating')
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
                    elif resu0[k] is None:
                        resu.append(-1)
                try:
                    sql_impl = sql_insert + sql_value
                    sql_impl = sql_impl % tuple(resu)
                    db.cursor.execute(sql_impl)
                    db.db.commit()
                except Exception as err:
                    self.logger.error(err)
                    continue
        self.logger.info('index data is fully updated')
