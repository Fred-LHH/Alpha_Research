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
BARRA_FACTOR_PATH = '/Volumes/T7Shield/资源/Alpha_research/Factor/Barra因子'


STATES = {
    'log': False,
    'comment': False,
    'save': False,
    'plot': False
}

hfq_columns = ['date', 'code', 'name', 'filling', 'open', 'close', 'high', 'low', 'vol', 'amount', 'distance', 'status', 'avg_price', 'pct_chg', 'total_share', 'circ_share', 'turnover', 'total_mv', 'circ_mv']

bfq_columns = ['code', 'date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'circ_mv', 'total_mv', 
           'market_type', 'status', 'pre_close', 'pct_chg', 'limit_down', 'limit_up', 'limit_status']


bfq_table_cols = {
    'code': 'VARCHAR(40)',
    'date': 'VARCHAR(40)',
    'open': 'DECIMAL(20, 3)',
    'high': 'DECIMAL(20, 3)',
    'low': 'DECIMAL(20, 3)',
    'close': 'DECIMAL(20, 3)',
    'vol': 'DECIMAL(20, 3)',
    'amount': 'DECIMAL(20, 3)',
    'circ_mv': 'DECIMAL(20, 3)',
    'total_mv': 'DECIMAL(20, 3)',
    'market_type': 'DECIMAL(20, 3)',
    'status': 'DECIMAL(20, 3)',
    'pre_close': 'DECIMAL(20, 3)',
    'pct_chg': 'DECIMAL(20, 3)',
    'limit_down': 'DECIMAL(20, 3)',
    'limit_up': 'DECIMAL(20, 3)',
    'limit_status': 'DECIMAL(20, 3)',
}

hfq_table_cols = {
    'code': 'VARCHAR(40)',
    'date': 'VARCHAR(40)',
    'name': 'VARCHAR(40)',
    'filling': 'VARCHAR(40)',
    'open': 'DECIMAL(20, 3)',
    'high': 'DECIMAL(20, 3)',
    'low': 'DECIMAL(20, 3)',
    'close': 'DECIMAL(20, 3)',
    'vol': 'DECIMAL(20, 3)',
    'amount': 'DECIMAL(20, 3)',
    'distance': 'DECIMAL(20, 3)',
    'circ_mv': 'DECIMAL(20, 3)',
    'total_mv': 'DECIMAL(20, 3)',
    'status': 'DECIMAL(20, 3)',
    'avg_price': 'DECIMAL(20, 3)',
    'pct_chg': 'DECIMAL(20, 3)',
    'total_share': 'DECIMAL(20, 3)',
    'circ_share': 'DECIMAL(20, 3)',
    'turnover': 'DECIMAL(20, 3)',
}



