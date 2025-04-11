START_DATE = '2009-01-01'

DATE1 = '2018-01-01'
DATE2 = '2025-02-01'

TOKEN = '18427aa0a10e23a2bf2bf2de0b240aa0005db0629feea9fa2a3bd6a8'
# tushare积分对应的频次
QPM = 500

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'walhh123',
    'database': 'Stock',
    'charset': 'utf8'
}

DB_NAME = {
    'dt' : 'trade_dates',
    'code': 'stocks',
    'bfq': 'bfq_daily_pv',
    'hfq': 'hfq_daily_pv',
    'st': 'st',
    'up_down_limit': 'limit_status',
    'suspend': 'suspend',
    'sw': 'sw_industry',
    'index_pv': 'index_daily_pv',
    '000010': 'index_000010',
    '000016': 'index_000016',
    '000300': 'index_000300',
    '000852': 'index_000852',
    '000905': 'index_000905',
    '399300': 'index_399300',
    '399329': 'index_399329',
    '399852': 'index_399852',
    '399905': 'index_399905',
}

LOCAL_DATA_PATH = '/Users/lihaohan/Desktop/国泰安'


STATES = {
    'log': False,
    'comment': False,
    'save': False,
    'plot': False
}