import time
import os
os.chdir('/Users/lihaohan/Alpha_Research')
import schedule
from Data.CSMAR.engine import *
from Data.utils import DButils
db = DButils()


def everyday_run(CSMAR_BFQ):
    print('timer is running ' + time.strftime('%Y-%m-%d %H:%M:%S'))
    schedule.every(1).hour.do(db.refresh)
    schedule.every().day.at("14:38").do(CSMAR_BFQ.update)



    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
    CSMAR_BFQ: Engine = BFQPVEngine()
    
    everyday_run(CSMAR_BFQ)