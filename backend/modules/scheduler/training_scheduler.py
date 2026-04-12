"""训练调度器"""

import logging
import threading
import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# 修复导入路径
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 修复导入错误，使用正确的模块路径
from backend.core.exceptions import BusinessLogicError
from backend.modules.training.scenarios import get_scenario_manager, TrainingScenario
from backend.modules.training.scenarios.scenario_manager import ScenarioConfig
from backend.services.training_service import get_training_service

logger = logging.getLogger(__name__)


class TrainingScheduler:
    """训练任务调度器"""
    
    def __init__(self):
        self.scheduled_tasks = {}
        self.scheduler_lock = threading.Lock()
        self.running = False
        self.scheduler_thread = None
        self.scenario_manager = get_scenario_manager()
        self.training_service = get_training_service()
        
    def start(self):
        """启动调度器"""
        if not self.running:
            self.running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
            logger.info("训练调度器已启动")
    
    def stop(self):
        """停止调度器"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        logger.info("训练调度器已停止")
    
    def schedule_task(self, task_config: Dict[str, Any], 
                     schedule_time: datetime) -> str:
        """调度任务"""
        try:
            task_id = str(uuid.uuid4())
            
            with self.scheduler_lock:
                self.scheduled_tasks[task_id] = {
                    'id': task_id,
                    'config': task_config,
                    'schedule_time': schedule_time,
                    'status': 'scheduled',
                    'created_at': datetime.now()
                }
            
            logger.info(f"任务 {task_id} 已调度于 {schedule_time}")
            return task_id
        except Exception as e:
            raise BusinessLogicError(f"调度任务失败: {e}")
    
    def cancel_task(self, task_id: str) -> bool:
        """取消调度任务"""
        try:
            with self.scheduler_lock:
                if task_id in self.scheduled_tasks:
                    del self.scheduled_tasks[task_id]
                    logger.info(f"任务 {task_id} 已取消")
                    return True
            return False
        except Exception as e:
            raise BusinessLogicError(f"取消任务失败: {e}")
    
    def get_scheduled_tasks(self) -> List[Dict[str, Any]]:
        """获取所有调度任务"""
        try:
            with self.scheduler_lock:
                return list(self.scheduled_tasks.values())
        except Exception as e:
            raise BusinessLogicError(f"获取调度任务失败: {e}")
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取指定任务"""
        try:
            with self.scheduler_lock:
                return self.scheduled_tasks.get(task_id)
        except Exception as e:
            raise BusinessLogicError(f"获取任务失败: {e}")
    
    def _scheduler_loop(self):
        """调度器主循环"""
        while self.running:
            try:
                current_time = datetime.now()
                tasks_to_execute = []
                
                with self.scheduler_lock:
                    for task_id, task in list(self.scheduled_tasks.items()):
                        if task['schedule_time'] <= current_time and task['status'] == 'scheduled':
                            tasks_to_execute.append(task)
                            task['status'] = 'executing'
                
                # 执行到期的任务
                for task in tasks_to_execute:
                    self._execute_scheduled_task(task)
                
                time.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                logger.error(f"调度器循环错误: {e}")
                time.sleep(5)
    
    def _execute_scheduled_task(self, task: Dict[str, Any]):
        """执行调度任务 - 使用真实的调度逻辑"""
        try:
            logger.info(f"开始执行调度任务: {task['id']}")
            
            # 获取任务配置
            config = task['config']
            
            # 创建场景配置 - 使用真实的配置
            scenario_type_value = config.get('scenario_type', 'basic_model')
            try:
                scenario_enum = TrainingScenario(scenario_type_value)
                scenario_type = scenario_enum.value
            except ValueError:
                scenario_enum = TrainingScenario.BASIC_MODEL
                scenario_type = scenario_enum.value
            
            scenario_name = config.get('name') or f"{scenario_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            scenario_config = ScenarioConfig(
                scenario=scenario_enum,
                name=scenario_name,
                output_dir=config.get('output_dir', './outputs'),
                custom_config={
                    'model_name': config.get('model_name', 'default_model'),
                    'train_data_path': config.get('train_data_path', './data/train'),
                    'val_data_path': config.get('val_data_path'),
                    'test_data_path': config.get('test_data_path'),
                    'num_epochs': config.get('num_epochs', 10),
                    'batch_size': config.get('batch_size', 16),
                    'learning_rate': config.get('learning_rate', 2e-5),
                    'max_concurrent_jobs': config.get('max_concurrent_jobs', 2)
                }
            )
            
            # 提交任务到场景管理器 - 使用真实的调度逻辑
            job_id = self.scenario_manager.submit_job(
                user_id=config.get('user_id', 'system'),
                scenario_type=scenario_type,
                name=scenario_name,
                config=scenario_config.to_dict(),
                description=config.get('description', ''),
                scheduled_at=task.get('schedule_time')
            )
            
            # 更新任务状态
            with self.scheduler_lock:
                if task['id'] in self.scheduled_tasks:
                    self.scheduled_tasks[task['id']]['status'] = 'completed'
                    self.scheduled_tasks[task['id']]['completed_at'] = datetime.now()
                    self.scheduled_tasks[task['id']]['job_id'] = job_id  # 关联实际的训练任务ID
            
            logger.info(f"调度任务执行完成: {task['id']}, 创建训练任务: {job_id}")
            
        except Exception as e:
            logger.error(f"执行调度任务失败: {e}")
            # 更新任务状态为失败
            with self.scheduler_lock:
                if task['id'] in self.scheduled_tasks:
                    self.scheduled_tasks[task['id']]['status'] = 'failed'
                    self.scheduled_tasks[task['id']]['error'] = str(e)
                    self.scheduled_tasks[task['id']]['completed_at'] = datetime.now()


# 全局调度器实例
_scheduler = TrainingScheduler()

def get_scheduler():
    """获取调度器实例"""
    return _scheduler

def schedule_training_task(task_config: Dict[str, Any], 
                          schedule_time: datetime) -> str:
    """调度训练任务"""
    return _scheduler.schedule_task(task_config, schedule_time)

def cancel_training_task(task_id: str) -> bool:
    """取消调度训练任务"""
    return _scheduler.cancel_task(task_id)

def get_scheduled_training_tasks() -> List[Dict[str, Any]]:
    """获取所有调度训练任务"""
    return _scheduler.get_scheduled_tasks()

def get_scheduled_training_task(task_id: str) -> Optional[Dict[str, Any]]:
    """获取指定的调度训练任务"""
    return _scheduler.get_task(task_id)


# 兼容性函数 - 来自 backend.modules.training.scheduler.training_scheduler
# 这些函数提供了与原训练调度器相同的接口

def schedule_training_task_with_session(
    session_id: str,
    task_config: Dict[str, Any],
    schedule_time: Optional[datetime] = None
) -> str:
    """调度训练任务（带会话ID）
    
    Args:
        session_id: 训练会话ID
        task_config: 任务配置
        schedule_time: 调度时间，如果为None则立即执行
        
    Returns:
        str: 任务ID
    """
    try:
        # 将 session_id 添加到任务配置中
        enhanced_config = task_config.copy()
        enhanced_config['session_id'] = session_id
        
        # 使用统一调度器
        return _scheduler.schedule_task(enhanced_config, schedule_time or datetime.now())
        
    except Exception as e:
        logger.error(f"调度训练任务失败: {session_id}, 错误: {e}")
        raise


def get_scheduled_training_tasks_filtered(
    session_id: Optional[str] = None,
    status: Optional[str] = None
) -> List[Dict[str, Any]]:
    """获取调度的训练任务列表（带过滤）
    
    Args:
        session_id: 训练会话ID，如果为None则返回所有任务
        status: 任务状态，如果为None则返回所有状态的任务
        
    Returns:
        List[Dict[str, Any]]: 任务列表
    """
    try:
        all_tasks = _scheduler.get_scheduled_tasks()
        filtered_tasks = []
        
        for task in all_tasks:
            # 过滤条件
            task_session_id = task.get('config', {}).get('session_id')
            if session_id and task_session_id != session_id:
                continue
            if status and task.get('status') != status:
                continue
                
            filtered_tasks.append(task)
        
        # 按创建时间排序
        filtered_tasks.sort(key=lambda x: x.get('created_at', datetime.now()), reverse=True)
        
        logger.debug(f"获取调度任务列表: {len(filtered_tasks)} 个任务")
        return filtered_tasks
        
    except Exception as e:
        logger.error(f"获取调度任务列表失败, 错误: {e}")
        return []


def get_task_by_id(task_id: str) -> Optional[Dict[str, Any]]:
    """根据ID获取任务
    
    Args:
        task_id: 任务ID
        
    Returns:
        Optional[Dict[str, Any]]: 任务数据
    """
    try:
        return _scheduler.get_task(task_id)
    except Exception as e:
        logger.error(f"获取任务失败: {task_id}, 错误: {e}")
        return None


def update_task_status(task_id: str, status: str, **kwargs) -> bool:
    """更新任务状态
    
    Args:
        task_id: 任务ID
        status: 新状态
        **kwargs: 其他要更新的字段
        
    Returns:
        bool: 是否成功更新
    """
    try:
        with _scheduler.scheduler_lock:
            if task_id in _scheduler.scheduled_tasks:
                task_data = _scheduler.scheduled_tasks[task_id]
                task_data['status'] = status
                task_data['updated_at'] = datetime.now()
                
                # 更新其他字段
                for key, value in kwargs.items():
                    task_data[key] = value
                
                logger.info(f"成功更新任务状态: {task_id}, 状态: {status}")
                return True
            else:
                logger.warning(f"未找到要更新的任务: {task_id}")
                return False
                
    except Exception as e:
        logger.error(f"更新任务状态失败: {task_id}, 错误: {e}")
        return False
