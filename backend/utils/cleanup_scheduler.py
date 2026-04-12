"""清理调度器

提供定时清理任务的调度和管理功能。
"""

import os
import time
import threading
from typing import Callable, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import shutil

logger = logging.getLogger(__name__)


class CleanupScheduler:
    """清理调度器"""
    
    def __init__(self):
        self.tasks = {}
        self.running = False
        self.thread = None
    
    def add_cleanup_task(
        self,
        task_name: str,
        cleanup_func: Callable,
        interval_hours: float = 24.0,
        max_age_days: int = 7,
        enabled: bool = True
    ) -> None:
        """添加清理任务
        
        Args:
            task_name: 任务名称
            cleanup_func: 清理函数
            interval_hours: 执行间隔（小时）
            max_age_days: 最大保留天数
            enabled: 是否启用
        """
        self.tasks[task_name] = {
            'func': cleanup_func,
            'interval': interval_hours * 3600,  # 转换为秒
            'max_age': max_age_days * 24 * 3600,  # 转换为秒
            'enabled': enabled,
            'last_run': None,
            'next_run': time.time() + interval_hours * 3600
        }
        
        logger.info(f"Cleanup task added: {task_name}")
    
    def remove_cleanup_task(self, task_name: str) -> bool:
        """移除清理任务
        
        Args:
            task_name: 任务名称
            
        Returns:
            是否移除成功
        """
        if task_name in self.tasks:
            del self.tasks[task_name]
            logger.info(f"Cleanup task removed: {task_name}")
            return True
        return False
    
    def enable_task(self, task_name: str) -> bool:
        """启用任务
        
        Args:
            task_name: 任务名称
            
        Returns:
            是否启用成功
        """
        if task_name in self.tasks:
            self.tasks[task_name]['enabled'] = True
            logger.info(f"Cleanup task enabled: {task_name}")
            return True
        return False
    
    def disable_task(self, task_name: str) -> bool:
        """禁用任务
        
        Args:
            task_name: 任务名称
            
        Returns:
            是否禁用成功
        """
        if task_name in self.tasks:
            self.tasks[task_name]['enabled'] = False
            logger.info(f"Cleanup task disabled: {task_name}")
            return True
        return False
    
    def start(self) -> None:
        """启动调度器"""
        if self.running:
            logger.warning("Cleanup scheduler is already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()
        logger.info("Cleanup scheduler started")
    
    def stop(self) -> None:
        """停止调度器"""
        if not self.running:
            logger.warning("Cleanup scheduler is not running")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("Cleanup scheduler stopped")
    
    def _run_scheduler(self) -> None:
        """运行调度器"""
        while self.running:
            try:
                current_time = time.time()
                
                for task_name, task_info in self.tasks.items():
                    if not task_info['enabled']:
                        continue
                    
                    if current_time >= task_info['next_run']:
                        self._execute_task(task_name, task_info)
                        task_info['last_run'] = current_time
                        task_info['next_run'] = current_time + task_info['interval']
                
                # 休眠一段时间避免过度占用CPU
                time.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                logger.error(f"Error in cleanup scheduler: {e}")
                time.sleep(60)
    
    def _execute_task(self, task_name: str, task_info: Dict[str, Any]) -> None:
        """执行清理任务"""
        try:
            logger.info(f"Executing cleanup task: {task_name}")
            
            # 调用清理函数
            task_info['func'](
                max_age=task_info['max_age']
            )
            
            logger.info(f"Cleanup task completed: {task_name}")
            
        except Exception as e:
            logger.error(f"Error executing cleanup task {task_name}: {e}")
    
    def get_task_status(self) -> Dict[str, Any]:
        """获取任务状态"""
        status = {}
        current_time = time.time()
        
        for task_name, task_info in self.tasks.items():
            status[task_name] = {
                'enabled': task_info['enabled'],
                'interval_hours': task_info['interval'] / 3600,
                'max_age_days': task_info['max_age'] / (24 * 3600),
                'last_run': datetime.fromtimestamp(task_info['last_run']).isoformat() if task_info['last_run'] else None,
                'next_run': datetime.fromtimestamp(task_info['next_run']).isoformat() if task_info['next_run'] else None,
                'due_soon': task_info['next_run'] - current_time < 3600 if task_info['next_run'] else False
            }
        
        return status


# 通用清理函数
def cleanup_temp_files(temp_dir: str = "/tmp", max_age: int = 7 * 24 * 3600) -> None:
    """清理临时文件
    
    Args:
        temp_dir: 临时目录路径
        max_age: 最大文件年龄（秒）
    """
    if not os.path.exists(temp_dir):
        return
    
    cutoff_time = time.time() - max_age
    
    for filename in os.listdir(temp_dir):
        filepath = os.path.join(temp_dir, filename)
        
        try:
            # 检查文件修改时间
            mod_time = os.path.getmtime(filepath)
            if mod_time < cutoff_time:
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    logger.debug(f"Removed temp file: {filepath}")
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
                    logger.debug(f"Removed temp directory: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to clean temp file {filepath}: {e}")


def cleanup_log_files(log_dir: str = "./logs", max_age: int = 30 * 24 * 3600) -> None:
    """清理日志文件
    
    Args:
        log_dir: 日志目录路径
        max_age: 最大文件年龄（秒）
    """
    if not os.path.exists(log_dir):
        return
    
    cutoff_time = time.time() - max_age
    
    for filename in os.listdir(log_dir):
        if filename.endswith('.log'):
            filepath = os.path.join(log_dir, filename)
            
            try:
                # 检查文件修改时间
                mod_time = os.path.getmtime(filepath)
                if mod_time < cutoff_time:
                    os.remove(filepath)
                    logger.debug(f"Removed log file: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to clean log file {filepath}: {e}")


def cleanup_checkpoint_files(checkpoint_dir: str = "./checkpoints", max_age: int = 7 * 24 * 3600) -> None:
    """清理检查点文件
    
    Args:
        checkpoint_dir: 检查点目录路径
        max_age: 最大文件年龄（秒）
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    cutoff_time = time.time() - max_age
    
    # 获取所有检查点文件
    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith(('.pt', '.pth')):
            filepath = os.path.join(checkpoint_dir, filename)
            checkpoint_files.append((filepath, os.path.getmtime(filepath)))
    
    # 按修改时间排序，保留最新的几个
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    
    # 删除超过保留数量的旧文件
    max_checkpoints = 5  # 保留最新的5个检查点
    for i, (filepath, mod_time) in enumerate(checkpoint_files):
        try:
            if i >= max_checkpoints or mod_time < cutoff_time:
                os.remove(filepath)
                logger.debug(f"Removed checkpoint file: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to clean checkpoint file {filepath}: {e}")


def cleanup_cache_files(cache_dir: str = "./cache", max_age: int = 24 * 3600) -> None:
    """清理缓存文件
    
    Args:
        cache_dir: 缓存目录路径
        max_age: 最大文件年龄（秒）
    """
    if not os.path.exists(cache_dir):
        return
    
    cutoff_time = time.time() - max_age
    
    for filename in os.listdir(cache_dir):
        filepath = os.path.join(cache_dir, filename)
        
        try:
            # 检查文件修改时间
            mod_time = os.path.getmtime(filepath)
            if mod_time < cutoff_time:
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    logger.debug(f"Removed cache file: {filepath}")
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
                    logger.debug(f"Removed cache directory: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to clean cache file {filepath}: {e}")


# 全局清理调度器实例
_global_cleanup_scheduler = CleanupScheduler()


def get_cleanup_scheduler() -> CleanupScheduler:
    """获取全局清理调度器实例
    
    Returns:
        CleanupScheduler实例
    """
    return _global_cleanup_scheduler


def setup_default_cleanup_tasks() -> None:
    """设置默认清理任务"""
    scheduler = get_cleanup_scheduler()
    
    # 添加默认清理任务
    scheduler.add_cleanup_task(
        'temp_files',
        cleanup_temp_files,
        interval_hours=6.0,
        max_age_days=1
    )
    
    scheduler.add_cleanup_task(
        'log_files',
        cleanup_log_files,
        interval_hours=24.0,
        max_age_days=30
    )
    
    scheduler.add_cleanup_task(
        'checkpoint_files',
        cleanup_checkpoint_files,
        interval_hours=24.0,
        max_age_days=7
    )
    
    scheduler.add_cleanup_task(
        'cache_files',
        cleanup_cache_files,
        interval_hours=1.0,
        max_age_days=1
    )
    
    logger.info("Default cleanup tasks configured")


def start_cleanup_scheduler() -> None:
    """启动清理调度器"""
    scheduler = get_cleanup_scheduler()
    scheduler.start()


def stop_cleanup_scheduler() -> None:
    """停止清理调度器"""
    scheduler = get_cleanup_scheduler()
    scheduler.stop()