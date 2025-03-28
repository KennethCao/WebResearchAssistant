from celery import Celery
from app.config.environment import (
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def create_celery_app(app=None):
    """创建Celery应用"""
    celery = Celery(
        'blockchain_research',
        broker=CELERY_BROKER_URL,
        backend=CELERY_RESULT_BACKEND,
        include=[
            'app.tasks.analysis',
            'app.tasks.embedding'
        ]
    )
    
    # 更新Celery配置
    celery.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,
        task_time_limit=3600,  # 1小时超时
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=200
    )
    
    if app:
        class ContextTask(celery.Task):
            def __call__(self, *args, **kwargs):
                with app.app_context():
                    return self.run(*args, **kwargs)
        
        celery.Task = ContextTask
    
    return celery

def handle_task_failure(self, exc, task_id, args, kwargs, einfo):
    """处理任务失败"""
    logger.error(f"Task {task_id} failed: {exc}")
    # 这里可以添加失败处理逻辑，如重试或通知

def handle_task_success(self, retval, task_id, args, kwargs):
    """处理任务成功"""
    logger.info(f"Task {task_id} completed successfully")
    # 这里可以添加成功处理逻辑

# 注册任务处理器
celery = create_celery_app()
celery.Task.on_failure = handle_task_failure
celery.Task.on_success = handle_task_success 