from datetime import *
import tushare as ts
import time
import Data.logger as logger
from config import *
import pandas as pd
import numpy as np

def singleton(cls):
    '''
    单例方法
    '''
    instances = {}
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return wrapper


@singleton
class TSDataFetcher():

    def __init__(self):
        ts.set_token(TOKEN)
        self.counts = 0
        self.last_tik = time.time()
        self.qpm = QPM / 2
        self.pro = ts.pro_api()
        self.counts = 0
        self.freq = 0
        self.ft = 5

    def pro_bar(self, ts_code='', api=None, start_date='', end_date='', freq='D',asset='E', exchange='', adj=None, ma=[], factors=None, adjfactor=False, offset=None, limit=None, contract_type='', retry_count=3):
        '''
        在tushare的pro_bar的基础上加伤超频阻塞逻辑
        '''
        try:
            df = ts.pro_bar(ts_code, api, start_date, end_date, freq, asset, exchange, adj, ma, factors, adjfactor, offset, limit, contract_type, retry_count)
        except Exception as e:
            logger.apilogger.error(e)
            logger.apilogger.info('sleeping 30s ... and retry')
            self.ft += 5
            self.freq = self.counts / (time.time() - self.last_tik) * 60
            self.counts = 0
            self.last_tik = time.time()
            logger.apilogger.info('the freq is' + str(self.freq) + 'times per minute')
            logger.apilogger.info('ft is update to %d' % self.ft)
            time.sleep(30)
            df = ts.pro_bar(ts_code, api, start_date, end_date, freq, asset, exchange, adj, ma, factors, adjfactor, offset, limit, contract_type, retry_count)
        
        self.counts += 1
        if df is None:
            return None
        
        if self.counts >= (self.qpm - self.ft):
            # 安全起见，降低ft qpm
            current_time = time.time()
            logger.apilogger.info('TIME GAP = ' + str(current_time - self.last_tik) + ", run " + str(self.qpm-self.ft) + " stocks ")
            self.freq = self.counts/(current_time - self.last_tik)*60
            logger.apilogger.info('the freqency is ' + str(self.freq) +'times per minute')
            if (current_time - self.last_tik) > 30:
                # 重置计数器
                self.counts = 0
                self.last_tik = current_time
                if self.ft > 5:
                    self.ft -= 1
            else:
                # 等到35秒
                self.counts = 0
                self.last_tik = time.time()
                logger.apilogger.info('pro_bar wait for '+str(35-(current_time - self.last_tik))+' seconds')
                time.sleep(35 - (current_time - self.last_tik))
        return df

    def query(self, api_name, fields='', **kwargs):
        '''
        重写tushare query方法
        '''
        try:
            df = self.pro.query(api_name, fields, **kwargs)
        except Exception as e:
            logger.apilogger.error(e)
            logger.apilogger.info('sleeping 30s... and retry')
            time.sleep(30)
            self.ft += 5
            self.freq = self.counts / (time.time() - self.last_tik) * 60
            self.counts = 0
            self.last_tik = time.time()
            logger.apilogger.info('the frequency is ' + str(self.freq) + 'times per minute')
            logger.apilogger.info("ft is update to %d" % self.ft)
            df = self.pro.query(api_name, fields, **kwargs)
        self.counts += 1
        if self.counts >= (self.qpm - self.ft):
            # 安全起见，降低c qpm
            current_time = time.time()
            logger.apilogger.info('TIME GAP = ' + str(current_time - self.last_tik) + ", run " + str(self.qpm-self.c) + " stocks ")
            self.freq = self.counts/(current_time - self.last_tik)*60
            logger.apilogger.info('the frequency is ' + str(self.freq) +'times per minute')
            if (current_time - self.last_tik) > 30:
                # 重置计数器
                if self.ft > 5:
                    self.ft -= 1
                self.counts = 0
                self.last_tik = current_time
            else:
                # 等到30秒
                self.counts = 0
                self.last_tik = time.time()
                logger.apilogger.info('query api wait for ' + str(35-(current_time - self.last_tik)) + ' seconds')
                time.sleep(35-(current_time - self.last_tik))

        return df
    
    def all_stocks(self):
        '''
        获取所有股票代码
        '''
        list_stocks = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,list_date, delist_date, list_status')
        delist_stocks = self.pro.stock_basic(exchange='', list_status='D', fields='ts_code,symbol,list_date, delist_date, list_status')
        all_stocks = pd.concat([list_stocks, delist_stocks], axis=0)
        return all_stocks
    
    
    def trade_dates(self, start_date='', end_date=''):
        '''
        获取交易日期
        '''
        dates = self.pro.trade_cal(start_date=start_date, end_date=end_date)
        trade_date = dates[dates['is_open'] == 1]
        trade_date = pd.DataFrame(trade_date[['cal_date']])
        trade_date.columns = ['date']
        return trade_date.reset_index(drop=True)
    
    def up_down_price(self, trade_date=''):
        '''
        获取涨跌停价格
        '''
        up_down = self.pro.stk_limit(trade_date=trade_date)
        return up_down

    
    ### 下面为财务指标api, tushare仅支持单只股票查询
    def income(self, ts_code='', start_date='', end_date='', fields=''):
        '''
        获取上市公司利润表
        '''
        return self.pro.income(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=fields)
    
    def balance(self, ts_code='', start_date='', end_date='', fields=''):
        '''
        获取上市公司资产负债表
        '''
        return self.pro.balancesheet(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=fields)
    
    def cashflow(self, ts_code='', start_date='', end_date='', fields=''):
        '''
        获取上市公司现金流量表
        '''
        return self.pro.cashflow(ts_code=ts_code, start_date=start_date, end_date=end_date, fields=fields)
    
    def fina_indicator(self, ts_code=''):
        '''
        获取上市公司财务指标数据
        '''
        return self.pro.fina_indicator(ts_code=ts_code)
    


