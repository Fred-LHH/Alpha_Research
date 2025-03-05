import re
import time
import logging
import datetime
from typing import Optional, Union, Iterable, Generator
from mysql.connector import pooling, Error
from mysql.connector.abstracts import MySQLConnectionAbstract
import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt

# 配置日志
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SQLError(Exception):
    """自定义数据库异常"""
    pass

class SQLDriver:
    def __init__(
        self,
        user: str,
        password: str,
        host: str = "localhost",
        port: int = 3306,
        database: Optional[str] = None,
        pool_size: int = 5,
        ssl_ca: Optional[str] = None,
        ssl_cert: Optional[str] = None,
        ssl_key: Optional[str] = None
    ):
        """
        初始化数据库驱动
        
        参数:
        ssl_ca: CA证书路径
        ssl_cert: 客户端证书路径
        ssl_key: 客户端密钥路径
        """
        self._validate_credentials(user, password)
        
        self.config = {
            "user": user,
            "password": password,
            "host": host,
            "port": port,
            "database": database,
            "pool_name": f"pool_{id(self)}",
            "pool_size": pool_size
        }

        if all([ssl_ca, ssl_cert, ssl_key]):
            self.config.update({
                "ssl_ca": ssl_ca,
                "ssl_cert": ssl_cert,
                "ssl_key": ssl_key
            })

        try:
            self.pool = pooling.MySQLConnectionPool(**self.config)
            logger.info("数据库连接池初始化成功")
        except Error as e:
            logger.critical("连接池创建失败", exc_info=True)
            raise SQLError(f"数据库连接失败: {e}") from e

    @staticmethod
    def _validate_credentials(user: str, password: str):
        """基础凭证验证"""
        if len(password) < 8:
            logger.warning("使用弱密码可能存在安全风险")
        if not re.match(r"^[a-zA-Z0-9_]+$", user):
            raise ValueError("非法用户名格式")

    @classmethod
    def from_config(cls, config_path: str = "db_config.yaml"):
        """从配置文件初始化"""
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def get_connection(self) -> MySQLConnectionAbstract:
        """获取连接（自动重连）"""
        try:
            conn = self.pool.get_connection()
            if not conn.is_connected():
                conn.ping(reconnect=True, attempts=3, delay=5)
            return conn
        except Error as e:
            logger.error("获取数据库连接失败")
            raise SQLError(f"连接获取失败: {e}") from e

    @retry(stop=stop_after_attempt(3))
    def execute(
        self,
        query: str,
        params: Optional[Union[tuple, dict]] = None,
        commit: bool = False
    ):
        """
        通用执行方法
        
        参数:
        commit: 是否自动提交（用于INSERT/UPDATE）
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params or ())
                    if commit:
                        conn.commit()
                    return cur
        except Error as e:
            logger.error(f"执行失败: {query[:50]}...")
            raise SQLError(f"SQL执行错误: {e}") from e

    def execute_many(self, query: str, data: Iterable):
        """批量执行"""
        with self.get_connection() as conn:
            try:
                conn.cursor().executemany(query, data)
                conn.commit()
            except Error as e:
                conn.rollback()
                raise SQLError(f"批量执行失败: {e}") from e

    def transaction(self, queries: list[tuple[str, tuple]]):
        """
        事务执行多个查询
        示例: driver.transaction([("INSERT ...", (params)), ("UPDATE ...", (params))])
        """
        with self.get_connection() as conn:
            try:
                conn.start_transaction()
                for query, params in queries:
                    conn.cursor().execute(query, params)
                conn.commit()
            except Error as e:
                conn.rollback()
                raise SQLError(f"事务执行失败: {e}") from e

    def _validate_identifier(self, name: str):
        """防止SQL注入"""
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(f"非法标识符: {name}")

    def add_database(self, db_name: str):
        """安全创建数据库"""
        self._validate_identifier(db_name)
        self.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`", commit=True)
        logger.info(f"数据库 {db_name} 创建/验证成功")

    def get_data(
        self,
        query: str,
        params: Optional[Union[tuple, dict]] = None,
        return_type: str = 'dataframe'
    ) -> Union[pd.DataFrame, np.ndarray, list]:
        """
        获取查询数据
        
        参数:
        return_type: dataframe / array / dict
        """
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                df = pd.read_sql(query, conn, params=params)
                
                logger.info(
                    f"查询返回 {len(df)} 行, 耗时 {time.time()-start_time:.2f}s"
                )
                
                if return_type == 'array':
                    return df.values
                elif return_type == 'dict':
                    return df.to_dict('records')
                return df
        except Error as e:
            raise SQLError(f"数据获取失败: {e}") from e

    def stream_data(self, query: str, chunk_size: int = 1000) -> Generator[pd.DataFrame, None, None]:
        """流式读取大数据"""
        with self.get_connection() as conn:
            with conn.cursor(named_tuple=True) as cur:
                cur.execute(query)
                while True:
                    rows = cur.fetchmany(chunk_size)
                    if not rows:
                        break
                    yield pd.DataFrame(rows)

    # 元数据操作
    def list_tables(self) -> list[str]:
        """列出所有数据表"""
        result = self.execute("SHOW TABLES")
        return [row[0] for row in result]

    def table_schema(self, table_name: str) -> pd.DataFrame:
        """获取表结构"""
        self._validate_identifier(table_name)
        return self.get_data(f"DESCRIBE {table_name}")

    # 性能监控
    def query_stats(self) -> dict:
        """返回查询统计信息"""
        return {
            'active_connections': self.pool.pool_size - self.pool._cnx_queue.qsize(),
            'total_connections': self.pool.pool_size
        }