# app/database/__init__.py
import sqlite3
import click
from flask import current_app, g
from flask.cli import with_appcontext
from app.utils.logger import setup_logger
from app.config.environment import SQLITE_DB_PATH

logger = setup_logger(__name__)

def get_db():
    """
    获取数据库连接。
    """
    if 'db' not in g:
        try:
            g.db = sqlite3.connect(
                SQLITE_DB_PATH,
                detect_types=sqlite3.PARSE_DECLTYPES
            )
            g.db.row_factory = sqlite3.Row
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    return g.db

def close_db(e=None):
    """
    关闭数据库连接。
    """
    db = g.pop('db', None)
    if db is not None:
        db.close()
        logger.info("Database connection closed")

def init_db():
    """
    初始化数据库。
    """
    try:
        db = get_db()
        
        # 创建必要的表
        with current_app.open_resource('database/schema.sql') as f:
            db.executescript(f.read().decode('utf8'))
            
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

def init_app(app):
    """
    初始化应用的数据库配置。
    """
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)

@click.command('init-db')
@with_appcontext
def init_db_command():
    """
    清除现有数据并创建新表。
    """
    init_db()
    click.echo('Initialized the database.')