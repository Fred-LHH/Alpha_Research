import pymysql
from config import *


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
class DButils():
    
    def __init__(self):
        self.db = pymysql.connect(**DB_CONFIG)
        self.cursor = self.db.cursor()
        self.cursor_d = self.db.cursor(cursor=pymysql.cursors.DictCursor)
        sql_database = 'use Stock'
        self.cursor.execute(sql_database)

    def refresh(self):
        try:
            sql_database = 'use Stock'
            self.cursor.execute(sql_database)
            print('database connection successful')
        except Exception as e:
            self.db = pymysql.connect(**DB_CONFIG)
            self.cursor = self.db.cursor()
            self.cursor_d = self.db.cursor(cursor=pymysql.cursors.DictCursor)
            print('reboot database connection')

            




