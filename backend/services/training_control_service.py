"""训练控制服务

提供训练任务的完整生命周期管理，包括创建、启动、暂停、恢复、取消等操作。
集成训练模块、仓库层和调度系统。

架构调用关系：
API层 -> Service层 (本模块) -> Launcher层 (training_launcher.py) -> Core层 -> 下游训练模块
"""

import sys
import os
import logging
import threading
import uuid
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.exceptions import ValidationError, BusinessLogicError, ResourceNotFoundError
from backend.schemas.enums import TrainingStatus, TrainingScenario
from backend.repositories.training_job_repository import (
    get_training_job_repository, 
    get_training_job_log_repository
)
from backend.schemas.training_models import TrainingJob

logger = logging.getLogger(__name__)

# =============================================================================
# 训练启动器集成
# =============================================================================

from backend.modules.training.launcher import (
    TrainingSystemLauncher,
    ProductionTrainingLauncher,
    launch_training_system,
    get_module_availability,
    diagnose_launcher_module,
    create_scenario_training_config,
    create_pipeline_training_config,
)

from backend.modules.training.progress import (
    get_progress_manager,
    TrainingProgressManager,
)

from backend.modules.training.scenarios.scenario_manager import get_scenario_manager

class TrainingControlAction(str, Enum):
    """训练控制操作枚举"""
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    RESTART = "restart"


@dataclass
class TrainingJobConfig:
    """训练任务配置"""
    name: str
    scenario_type: str
    model_name: str = None
    model_id: str = None
    dataset_id: str = None
    training_mode: str = 'standard'
    description: str = ''
    priority: int = 5
    config: Dict[str, Any] = None
    resource_config: Dict[str, Any] = None
    tags: List[str] = None


class TrainingControlService:
    """训练控制服务
    
    提供训练任务的完整生命周期管理功能。
    所有训练执行统一通过 TrainingSystemLauncher 调度。
    """
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化训练控制服务
        
        Args:
            use_memory_storage: 是否使用内存存储
        """
        self._use_memory_storage = use_memory_storage
        self.job_repo = get_training_job_repository(use_memory_storage)
        self.log_repo = get_training_job_log_repository(use_memory_storage)
        
        # 训练执行器（延迟加载）
        self._training_executor = None
        self._scenario_manager = None
        self._task_manager = None
        
        # 训练启动器（统一调度）
        self._launcher: Optional[Any] = None
        self._progress_manager: Optional[Any] = None
        
        # 进度回调
        self._progress_callbacks: Dict[str, Callable] = {}
        
        # 运行中的任务跟踪
        self._running_tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
        # 初始化启动器
        self._init_launcher()
        
        logger.info("Training control service initialized")
    
    def _init_launcher(self):
        """初始化训练启动器"""
        try:
            availability = get_module_availability()
            logger.info(f"Launcher module availability: {availability}")
        except Exception as e:
            logger.warning(f"Failed to check launcher availability: {e}")

        try:
            self._progress_manager = get_progress_manager()
            logger.info("Progress manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize progress manager: {e}")
    
    def _get_launcher(self, config: Dict[str, Any]) -> Optional[Any]:
        """获取训练启动器实例
        
        Args:
            config: 启动器配置
            
        Returns:
            TrainingSystemLauncher 实例
        """
        
        try:
            launcher_config = self._build_launcher_config(config)
            return TrainingSystemLauncher(launcher_config)
        except Exception as e:
            logger.error(f"Failed to create launcher: {e}")
            return None
    
    def _build_launcher_config(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """构建启动器配置
        
        Args:
            job_config: 任务配置
            
        Returns:
            启动器配置字典
        """
        training_mode = job_config.get('training_mode', 'standard')
        scenario_type = job_config.get('scenario_type', 'standard')
        
        launcher_config = {
            'model': {
                'name': job_config.get('model_name', 'default_model'),
                'id': job_config.get('model_id'),
                'type': job_config.get('model_type', 'transformer')
            },
            'training': {
                'mode': training_mode,
                'scenario_type': scenario_type,
                'num_epochs': job_config.get('config', {}).get('epochs', 10),
                'batch_size': job_config.get('config', {}).get('batch_size', 16),
                'learning_rate': job_config.get('config', {}).get('learning_rate', 2e-5),
                'output_dir': job_config.get('config', {}).get('output_dir', './outputs')
            },
            'data': {
                'dataset_id': job_config.get('dataset_id'),
                'train_path': job_config.get('config', {}).get('train_data_path')
            }
        }
        
        # 根据训练模式添加特定配置
        if training_mode == 'three_stage':
            launcher_config['three_stage'] = {
                'enabled': True,
                **job_config.get('config', {}).get('three_stage', {})
            }
        elif training_mode == 'distributed':
            launcher_config['distributed'] = {
                'enabled': True,
                **job_config.get('config', {}).get('distributed', {})
            }
        elif training_mode == 'multimodal':
            launcher_config['multimodal'] = {
                'enabled': True,
                **job_config.get('config', {}).get('multimodal', {})
            }
        elif training_mode == 'distillation':
            launcher_config['distillation'] = {
                'enabled': True,
                **job_config.get('config', {}).get('distillation', {})
            }
        elif training_mode == 'industry':
            launcher_config['industry'] = {
                'enabled': True,
                **job_config.get('config', {}).get('industry', {})
            }
        
        return launcher_config
    
    # ==================== 任务创建 ====================
    
    def create_job(self, tenant_id: str, user_id: str, 
                   config: TrainingJobConfig) -> Dict[str, Any]:
        """创建训练任务
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            config: 任务配置
            
        Returns:
            创建的任务信息
        """
        try:
            # 验证配置
            self._validate_job_config(config)
            
            # 生成任务ID
            job_id = f"job_{uuid.uuid4().hex[:16]}"
            
            # 创建任务对象
            job_data = {
                'job_id': job_id,
                'tenant_id': tenant_id,
                'user_id': user_id,
                'name': config.name,
                'description': config.description,
                'scenario_type': config.scenario_type,
                'training_mode': config.training_mode,
                'model_id': config.model_id,
                'model_name': config.model_name,
                'dataset_id': config.dataset_id,
                'config': config.config or {},
                'resource_config': config.resource_config or {},
                'priority': config.priority,
                'tags': config.tags or [],
                'status': TrainingStatus.PENDING.value,
                'progress': 0.0
            }
            
            # 保存到数据库
            job = self.job_repo.create(job_data)
            
            # 记录日志
            self._log_action(job_id, 'status_change', 
                           f"Training job created: {config.name}",
                           to_status=TrainingStatus.PENDING.value,
                           operator_id=user_id)
            
            logger.info(f"Training job created: {job_id}")
            
            return {
                'success': True,
                'job_id': job_id,
                'status': job.status,
                'message': '训练任务创建成功',
                'job': job.to_dict() if hasattr(job, 'to_dict') else job_data
            }
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to create training job: {e}")
            raise BusinessLogicError(f"创建训练任务失败: {str(e)}", operation="create_job")
    
    def _validate_job_config(self, config: TrainingJobConfig):
        """验证任务配置"""
        if not config.name:
            raise ValidationError("任务名称不能为空")
        
        if not config.scenario_type:
            raise ValidationError("训练场景类型不能为空")
        
        # 验证场景类型
        try:
            TrainingScenario(config.scenario_type)
        except ValueError:
            # 允许自定义场景类型
            logger.warning(f"Using custom scenario type: {config.scenario_type}")
        
        # 验证训练模式
        valid_modes = ['standard', 'distributed', 'multimodal', 'distillation', 
                      'three_stage', 'scenario', 'industry']
        if config.training_mode and config.training_mode not in valid_modes:
            raise ValidationError(f"不支持的训练模式: {config.training_mode}")
    
    # ==================== 任务控制 ====================
    
    def start_job(self, job_id: str, tenant_id: str, user_id: str) -> Dict[str, Any]:
        """启动训练任务
        
        Args:
            job_id: 任务ID
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            操作结果
        """
        try:
            # 获取任务
            job = self.job_repo.get_by_job_id(job_id, tenant_id)
            if not job:
                raise ResourceNotFoundError(f"训练任务不存在: {job_id}")
            
            # 检查状态
            if job.status == TrainingStatus.RUNNING.value:
                return {
                    'success': True,
                    'job_id': job_id,
                    'status': job.status,
                    'message': '训练任务已在运行中'
                }
            
            if job.status not in [TrainingStatus.PENDING.value, TrainingStatus.PAUSED.value]:
                raise BusinessLogicError(
                    f"无法启动状态为 {job.status} 的任务",
                    operation="start_job"
                )
            
            # 更新状态
            old_status = job.status
            job = self.job_repo.update_status(
                job_id, TrainingStatus.RUNNING,
                tenant_id=tenant_id,
                started_at=datetime.utcnow()
            )
            
            # 记录日志
            self._log_action(job_id, 'status_change',
                           f"Training job started by user {user_id}",
                           from_status=old_status,
                           to_status=TrainingStatus.RUNNING.value,
                           operator_id=user_id)
            
            # 启动实际训练（异步）
            self._start_training_async(job)
            
            logger.info(f"Training job started: {job_id}")
            
            return {
                'success': True,
                'job_id': job_id,
                'status': TrainingStatus.RUNNING.value,
                'message': '训练任务已开始'
            }
            
        except (ResourceNotFoundError, BusinessLogicError):
            raise
        except Exception as e:
            logger.error(f"Failed to start training job: {e}")
            raise BusinessLogicError(f"启动训练任务失败: {str(e)}", operation="start_job")
    
    def pause_job(self, job_id: str, tenant_id: str, user_id: str) -> Dict[str, Any]:
        """暂停训练任务
        
        Args:
            job_id: 任务ID
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            操作结果
        """
        try:
            # 获取任务
            job = self.job_repo.get_by_job_id(job_id, tenant_id)
            if not job:
                raise ResourceNotFoundError(f"训练任务不存在: {job_id}")
            
            # 检查状态
            if job.status == TrainingStatus.PAUSED.value:
                return {
                    'success': True,
                    'job_id': job_id,
                    'status': job.status,
                    'message': '训练任务已处于暂停状态'
                }
            
            if job.status != TrainingStatus.RUNNING.value:
                raise BusinessLogicError(
                    f"只能暂停运行中的任务，当前状态: {job.status}",
                    operation="pause_job"
                )
            
            # 暂停实际训练
            self._pause_training_async(job)
            
            # 更新状态
            old_status = job.status
            job = self.job_repo.update_status(
                job_id, TrainingStatus.PAUSED,
                tenant_id=tenant_id,
                paused_at=datetime.utcnow()
            )
            
            # 记录日志
            self._log_action(job_id, 'status_change',
                           f"Training job paused by user {user_id}",
                           from_status=old_status,
                           to_status=TrainingStatus.PAUSED.value,
                           operator_id=user_id)
            
            logger.info(f"Training job paused: {job_id}")
            
            return {
                'success': True,
                'job_id': job_id,
                'status': TrainingStatus.PAUSED.value,
                'message': '训练任务已暂停'
            }
            
        except (ResourceNotFoundError, BusinessLogicError):
            raise
        except Exception as e:
            logger.error(f"Failed to pause training job: {e}")
            raise BusinessLogicError(f"暂停训练任务失败: {str(e)}", operation="pause_job")
    
    def resume_job(self, job_id: str, tenant_id: str, user_id: str) -> Dict[str, Any]:
        """恢复训练任务
        
        Args:
            job_id: 任务ID
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            操作结果
        """
        try:
            # 获取任务
            job = self.job_repo.get_by_job_id(job_id, tenant_id)
            if not job:
                raise ResourceNotFoundError(f"训练任务不存在: {job_id}")
            
            # 检查状态
            if job.status == TrainingStatus.RUNNING.value:
                return {
                    'success': True,
                    'job_id': job_id,
                    'status': job.status,
                    'message': '训练任务已在运行中'
                }
            
            if job.status != TrainingStatus.PAUSED.value:
                raise BusinessLogicError(
                    f"只能恢复暂停状态的任务，当前状态: {job.status}",
                    operation="resume_job"
                )
            
            # 更新状态
            old_status = job.status
            job = self.job_repo.update_status(
                job_id, TrainingStatus.RUNNING,
                tenant_id=tenant_id,
                resumed_at=datetime.utcnow()
            )
            
            # 恢复实际训练
            self._resume_training_async(job)
            
            # 记录日志
            self._log_action(job_id, 'status_change',
                           f"Training job resumed by user {user_id}",
                           from_status=old_status,
                           to_status=TrainingStatus.RUNNING.value,
                           operator_id=user_id)
            
            logger.info(f"Training job resumed: {job_id}")
            
            return {
                'success': True,
                'job_id': job_id,
                'status': TrainingStatus.RUNNING.value,
                'message': '训练任务已恢复'
            }
            
        except (ResourceNotFoundError, BusinessLogicError):
            raise
        except Exception as e:
            logger.error(f"Failed to resume training job: {e}")
            raise BusinessLogicError(f"恢复训练任务失败: {str(e)}", operation="resume_job")
    
    def cancel_job(self, job_id: str, tenant_id: str, user_id: str,
                  reason: str = None) -> Dict[str, Any]:
        """取消训练任务
        
        Args:
            job_id: 任务ID
            tenant_id: 租户ID
            user_id: 用户ID
            reason: 取消原因
            
        Returns:
            操作结果
        """
        try:
            # 获取任务
            job = self.job_repo.get_by_job_id(job_id, tenant_id)
            if not job:
                raise ResourceNotFoundError(f"训练任务不存在: {job_id}")
            
            # 检查状态
            final_statuses = [TrainingStatus.COMPLETED.value, 
                             TrainingStatus.CANCELLED.value,
                             TrainingStatus.FAILED.value]
            
            if job.status in final_statuses:
                raise BusinessLogicError(
                    f"无法取消已完成的任务，当前状态: {job.status}",
                    operation="cancel_job"
                )
            
            # 取消实际训练
            self._cancel_training_async(job)
            
            # 更新状态
            old_status = job.status
            job = self.job_repo.update_status(
                job_id, TrainingStatus.CANCELLED,
                tenant_id=tenant_id,
                completed_at=datetime.utcnow(),
                error_message=reason or "User cancelled"
            )
            
            # 记录日志
            self._log_action(job_id, 'status_change',
                           f"Training job cancelled by user {user_id}: {reason or 'No reason provided'}",
                           from_status=old_status,
                           to_status=TrainingStatus.CANCELLED.value,
                           operator_id=user_id)
            
            logger.info(f"Training job cancelled: {job_id}")
            
            return {
                'success': True,
                'job_id': job_id,
                'status': TrainingStatus.CANCELLED.value,
                'message': '训练任务已取消'
            }
            
        except (ResourceNotFoundError, BusinessLogicError):
            raise
        except Exception as e:
            logger.error(f"Failed to cancel training job: {e}")
            raise BusinessLogicError(f"取消训练任务失败: {str(e)}", operation="cancel_job")
    
    def restart_job(self, job_id: str, tenant_id: str, user_id: str,
                   from_checkpoint: bool = True) -> Dict[str, Any]:
        """重新启动训练任务
        
        Args:
            job_id: 任务ID
            tenant_id: 租户ID
            user_id: 用户ID
            from_checkpoint: 是否从检查点恢复
            
        Returns:
            操作结果
        """
        try:
            # 获取任务
            job = self.job_repo.get_by_job_id(job_id, tenant_id)
            if not job:
                raise ResourceNotFoundError(f"训练任务不存在: {job_id}")
            
            # 如果任务正在运行，先取消
            if job.status == TrainingStatus.RUNNING.value:
                self._cancel_training_async(job)
            
            # 重置任务状态
            old_status = job.status
            update_data = {
                'started_at': datetime.utcnow(),
                'completed_at': None,
                'error_message': None,
                'retry_count': (job.retry_count or 0) + 1
            }
            
            if not from_checkpoint:
                update_data.update({
                    'progress': 0.0,
                    'current_epoch': 0,
                    'current_step': 0,
                    'metrics': None,
                    'checkpoint_path': None,
                    'checkpoint_epoch': None
                })
            
            job = self.job_repo.update_status(
                job_id, TrainingStatus.RUNNING,
                tenant_id=tenant_id,
                **update_data
            )
            
            # 记录日志
            self._log_action(job_id, 'status_change',
                           f"Training job restarted by user {user_id} (from_checkpoint={from_checkpoint})",
                           from_status=old_status,
                           to_status=TrainingStatus.RUNNING.value,
                           operator_id=user_id)
            
            # 启动实际训练
            self._start_training_async(job, from_checkpoint=from_checkpoint)
            
            logger.info(f"Training job restarted: {job_id}")
            
            return {
                'success': True,
                'job_id': job_id,
                'status': TrainingStatus.RUNNING.value,
                'message': '训练任务已重新开始',
                'from_checkpoint': from_checkpoint
            }
            
        except (ResourceNotFoundError, BusinessLogicError):
            raise
        except Exception as e:
            logger.error(f"Failed to restart training job: {e}")
            raise BusinessLogicError(f"重新开始训练任务失败: {str(e)}", operation="restart_job")
    
    # ==================== 状态和进度查询 ====================
    
    def get_job_status(self, job_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """获取训练任务状态
        
        Args:
            job_id: 任务ID
            tenant_id: 租户ID
            
        Returns:
            任务状态信息
        """
        try:
            job = self.job_repo.get_by_job_id(job_id, tenant_id)
            if not job:
                return None
            
            # 计算运行时间
            duration = None
            if job.started_at:
                end_time = job.completed_at or datetime.utcnow()
                duration = (end_time - job.started_at).total_seconds()
            
            return {
                'job_id': job.job_id,
                'name': job.name,
                'status': job.status,
                'progress': job.progress,
                'current_epoch': job.current_epoch,
                'total_epochs': job.total_epochs,
                'current_step': job.current_step,
                'total_steps': job.total_steps,
                'metrics': job.metrics,
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'duration_seconds': duration,
                'error_message': job.error_message
            }
            
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return None
    
    def get_job_progress(self, job_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """获取训练任务进度
        
        Args:
            job_id: 任务ID
            tenant_id: 租户ID
            
        Returns:
            任务进度信息
        """
        try:
            job = self.job_repo.get_by_job_id(job_id, tenant_id)
            if not job:
                return None
            
            # 计算ETA
            eta = None
            if job.status == TrainingStatus.RUNNING.value and job.progress > 0:
                if job.started_at:
                    elapsed = (datetime.utcnow() - job.started_at).total_seconds()
                    remaining_ratio = (100 - job.progress) / job.progress
                    eta = int(elapsed * remaining_ratio)
            
            return {
                'job_id': job.job_id,
                'progress': job.progress,
                'current_epoch': job.current_epoch,
                'total_epochs': job.total_epochs,
                'current_step': job.current_step,
                'total_steps': job.total_steps,
                'eta': eta,
                'metrics': job.metrics or {},
                'best_metrics': job.best_metrics,
                'checkpoint_path': job.checkpoint_path,
                'checkpoint_epoch': job.checkpoint_epoch
            }
            
        except Exception as e:
            logger.error(f"Failed to get job progress: {e}")
            return None
    
    def get_job_info(self, job_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """获取训练任务详细信息
        
        Args:
            job_id: 任务ID
            tenant_id: 租户ID
            
        Returns:
            任务详细信息
        """
        try:
            job = self.job_repo.get_by_job_id(job_id, tenant_id)
            if not job:
                return None
            
            return job.to_dict() if hasattr(job, 'to_dict') else {
                'job_id': job.job_id,
                'name': job.name,
                'status': job.status
            }
            
        except Exception as e:
            logger.error(f"Failed to get job info: {e}")
            return None
    
    # ==================== 列表和统计 ====================
    
    def list_jobs(self, tenant_id: str, user_id: str = None,
                 status: str = None, scenario_type: str = None,
                 limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """列出训练任务
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID过滤
            status: 状态过滤
            scenario_type: 场景类型过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            任务列表
        """
        try:
            jobs = self.job_repo.list_by_tenant(
                tenant_id=tenant_id,
                user_id=user_id,
                status=status,
                scenario_type=scenario_type,
                limit=limit,
                offset=offset
            )
            
            return [
                job.to_dict() if hasattr(job, 'to_dict') else {
                    'job_id': job.job_id,
                    'name': job.name,
                    'status': job.status
                }
                for job in jobs
            ]
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return []
    
    def get_statistics(self, tenant_id: str, user_id: str = None) -> Dict[str, Any]:
        """获取训练统计信息
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            统计信息
        """
        try:
            stats = self.job_repo.get_statistics(tenant_id, user_id)
            
            # 添加调度器状态
            stats['scheduler_running'] = self._is_scheduler_running()
            stats['max_concurrent_jobs'] = self._get_max_concurrent_jobs()
            stats['queue_size'] = len([
                j for j in self._running_tasks.values()
                if j.get('status') == TrainingStatus.PENDING.value
            ])
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                'total_jobs': 0,
                'running_jobs': 0,
                'completed_jobs': 0,
                'failed_jobs': 0,
                'paused_jobs': 0,
                'cancelled_jobs': 0,
                'scheduler_running': False,
                'max_concurrent_jobs': 3,
                'queue_size': 0
            }
    
    # ==================== 日志查询 ====================
    
    def get_job_logs(self, job_id: str, tenant_id: str,
                    log_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取训练任务日志
        
        Args:
            job_id: 任务ID
            tenant_id: 租户ID
            log_type: 日志类型过滤
            limit: 限制数量
            
        Returns:
            日志列表
        """
        try:
            # 验证任务存在
            job = self.job_repo.get_by_job_id(job_id, tenant_id)
            if not job:
                return []
            
            logs = self.log_repo.list_by_job(job_id, log_type=log_type, limit=limit)
            
            return [
                log.to_dict() if hasattr(log, 'to_dict') else {
                    'job_id': log.job_id,
                    'log_type': log.log_type,
                    'message': log.message
                }
                for log in logs
            ]
            
        except Exception as e:
            logger.error(f"Failed to get job logs: {e}")
            return []
    
    # ==================== 进度更新（回调） ====================
    
    def update_progress(self, job_id: str, tenant_id: str,
                       progress: float, current_step: int = None,
                       current_epoch: int = None, metrics: Dict[str, Any] = None) -> bool:
        """更新训练进度（由训练器调用）
        
        Args:
            job_id: 任务ID
            tenant_id: 租户ID
            progress: 进度值
            current_step: 当前步骤
            current_epoch: 当前轮次
            metrics: 指标数据
            
        Returns:
            是否更新成功
        """
        try:
            job = self.job_repo.update_progress(
                job_id=job_id,
                progress=progress,
                current_step=current_step,
                current_epoch=current_epoch,
                metrics=metrics,
                tenant_id=tenant_id
            )
            
            if job:
                # 记录进度日志
                self._log_action(job_id, 'progress',
                               f"Progress: {progress:.1f}%, epoch: {current_epoch}, step: {current_step}",
                               epoch=current_epoch,
                               step=current_step,
                               metrics=metrics)
                
                # 触发进度回调
                if job_id in self._progress_callbacks:
                    try:
                        self._progress_callbacks[job_id](progress, metrics)
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update progress: {e}")
            return False
    
    def complete_job(self, job_id: str, tenant_id: str,
                    success: bool = True, result: Dict[str, Any] = None,
                    error_message: str = None) -> bool:
        """完成训练任务（由训练器调用）
        
        Args:
            job_id: 任务ID
            tenant_id: 租户ID
            success: 是否成功
            result: 训练结果
            error_message: 错误信息
            
        Returns:
            是否更新成功
        """
        try:
            status = TrainingStatus.COMPLETED if success else TrainingStatus.FAILED
            
            job = self.job_repo.update_status(
                job_id, status,
                tenant_id=tenant_id,
                completed_at=datetime.utcnow(),
                result=result,
                error_message=error_message,
                progress=100.0 if success else None
            )
            
            if job:
                # 记录日志
                self._log_action(job_id, 'status_change',
                               f"Training {'completed successfully' if success else 'failed'}: {error_message or 'No errors'}",
                               to_status=status.value,
                               details={'result': result})
                
                # 清理运行状态
                with self._lock:
                    if job_id in self._running_tasks:
                        del self._running_tasks[job_id]
                    if job_id in self._progress_callbacks:
                        del self._progress_callbacks[job_id]
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to complete job: {e}")
            return False
    
    # ==================== 私有方法 ====================
    
    def _log_action(self, job_id: str, log_type: str, message: str,
                   from_status: str = None, to_status: str = None,
                   operator_id: str = None, epoch: int = None,
                   step: int = None, metrics: Dict[str, Any] = None,
                   details: Dict[str, Any] = None):
        """记录操作日志"""
        try:
            log_data = {
                'job_id': job_id,
                'log_type': log_type,
                'log_level': 'info',
                'message': message,
                'from_status': from_status,
                'to_status': to_status,
                'operator_id': operator_id,
                'epoch': epoch,
                'step': step,
                'metrics': metrics,
                'details': details,
                'source': 'training_control_service'
            }
            
            self.log_repo.create(log_data)
            
        except Exception as e:
            logger.warning(f"Failed to log action: {e}")
    
    def _start_training_async(self, job: TrainingJob, from_checkpoint: bool = False):
        """异步启动训练 (优先通过 Launcher 执行)"""
        with self._lock:
            self._running_tasks[job.job_id] = {
                'status': TrainingStatus.RUNNING.value,
                'started_at': datetime.utcnow()
            }
        
        # 初始化进度跟踪
        if self._progress_manager:
            try:
                config = job.config or {}
                total_steps = config.get('total_steps', 1000)
                self._progress_manager.create_progress_tracker(job.job_id, total_steps)
                self._progress_manager.set_status(job.job_id, 'running')
            except Exception as e:
                logger.warning(f"Failed to initialize progress tracking: {e}")
        
        # 优先通过 Launcher 启动
        try:
            launcher_result = self._start_training_via_launcher(job, from_checkpoint)
            if launcher_result:
                logger.info(f"Training started via launcher: {job.job_id}")
                return
        except Exception as e:
            logger.warning(f"Launcher training failed, falling back: {e}")
        
        # 尝试通过场景管理器启动
        try:
            scenario_manager = self._get_scenario_manager()
            if scenario_manager:
                # 将 from_checkpoint 添加到配置中
                job_config = job.config.copy() if job.config else {}
                if from_checkpoint:
                    job_config['from_checkpoint'] = True
                
                scenario_manager.submit_job(
                    user_id=job.user_id,
                    scenario_type=job.scenario_type,
                    name=job.name,
                    config=job_config,
                    priority=job.priority,
                    description=job.description
                )
                return
        except Exception as e:
            logger.warning(f"Failed to start via scenario manager: {e}")
        
        # 尝试通过任务管理器启动
        try:
            task_manager = self._get_task_manager()
            if task_manager:
                task_manager.submit_training_task(job.job_id)
                return
        except Exception as e:
            logger.warning(f"Failed to start via task manager: {e}")
        
        # 直接在后台线程启动
        logger.info(f"Starting training in background thread: {job.job_id}")
        thread = threading.Thread(target=self._run_training, args=(job, from_checkpoint))
        thread.daemon = True
        thread.start()
    
    def _start_training_via_launcher(
        self,
        job: TrainingJob,
        from_checkpoint: bool = False
    ) -> bool:
        """通过 Launcher 启动训练
        
        Args:
            job: 训练任务对象
            from_checkpoint: 是否从检查点恢复
            
        Returns:
            是否成功启动
        """
        try:
            # 构建配置
            job_config = {
                'model_name': job.model_name,
                'model_id': job.model_id,
                'dataset_id': job.dataset_id,
                'scenario_type': job.scenario_type,
                'training_mode': job.training_mode or 'standard',
                'config': job.config or {}
            }
            
            if from_checkpoint and job.checkpoint_path:
                job_config['config']['resume_from_checkpoint'] = job.checkpoint_path
            
            launcher_config = self._build_launcher_config(job_config)
            
            # 创建启动器并启动训练
            launcher = TrainingSystemLauncher(launcher_config)
            
            # 在后台线程执行训练
            def run_launcher_training():
                try:
                    # 分析配置
                    analysis = launcher.analyze_config()
                    logger.info(f"Launcher analysis for job {job.job_id}: {analysis}")
                    
                    # 选择训练器
                    trainer = launcher.select_trainer(analysis)
                    
                    if trainer is None:
                        raise Exception("Launcher returned no trainer")
                    
                    # 设置进度回调
                    def progress_callback(progress: float, metrics: Dict[str, Any] = None):
                        self.update_progress(
                            job.job_id,
                            str(job.tenant_id),
                            progress=progress,
                            metrics=metrics
                        )
                    
                    if hasattr(trainer, 'set_progress_callback'):
                        trainer.set_progress_callback(progress_callback)
                    
                    # 执行训练
                    result = trainer.train()
                    
                    # 完成训练
                    self.complete_job(
                        job.job_id,
                        str(job.tenant_id),
                        success=result.get('success', True),
                        result=result,
                        error_message=result.get('error')
                    )
                    
                except Exception as e:
                    logger.error(f"Launcher training failed for job {job.job_id}: {e}")
                    self.complete_job(
                        job.job_id,
                        str(job.tenant_id),
                        success=False,
                        error_message=str(e)
                    )
            
            thread = threading.Thread(target=run_launcher_training, daemon=True)
            thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training via launcher: {e}")
            return False
    
    def launch_production_training(
        self,
        job_id: str,
        tenant_id: str,
        user_id: str,
        training_type: str = 'standard',
        **kwargs
    ) -> Dict[str, Any]:
        """使用生产级启动器启动训练
        
        Args:
            job_id: 任务ID
            tenant_id: 租户ID
            user_id: 用户ID
            training_type: 训练类型
            **kwargs: 额外配置
            
        Returns:
            启动结果
        """
        try:
            # 获取任务
            job = self.job_repo.get_by_job_id(job_id, tenant_id)
            if not job:
                raise ResourceNotFoundError(f"训练任务不存在: {job_id}")
            
            # 构建生产级配置
            from backend.modules.training.launcher import create_production_training_config
            
            config = job.config or {}
            production_config = create_production_training_config(
                training_type=training_type,
                output_dir=config.get('output_dir', f'./outputs/{job_id}'),
                model_name=job.model_name or 'production_model',
                num_epochs=config.get('epochs', kwargs.get('num_epochs', 10)),
                batch_size=config.get('batch_size', kwargs.get('batch_size', 16)),
                learning_rate=config.get('learning_rate', kwargs.get('learning_rate', 2e-5)),
                enable_checkpoint=kwargs.get('enable_checkpoint', True),
                enable_monitoring=kwargs.get('enable_monitoring', True),
            )
            
            # 更新任务状态
            old_status = job.status
            job = self.job_repo.update_status(
                job_id, TrainingStatus.RUNNING,
                tenant_id=tenant_id,
                started_at=datetime.utcnow()
            )
            
            # 记录日志
            self._log_action(job_id, 'status_change',
                           f"Production training started with type: {training_type}",
                           from_status=old_status,
                           to_status=TrainingStatus.RUNNING.value,
                           operator_id=user_id)
            
            # 在后台线程启动生产级训练
            def run_production():
                try:
                    launcher = ProductionTrainingLauncher(production_config)
                    result = launcher.launch_training()
                    
                    self.complete_job(
                        job_id, tenant_id,
                        success=result.get('success', False),
                        result=result,
                        error_message=result.get('error')
                    )
                except Exception as e:
                    logger.error(f"Production training failed: {e}")
                    self.complete_job(job_id, tenant_id, success=False, error_message=str(e))
            
            thread = threading.Thread(target=run_production, daemon=True)
            thread.start()
            
            with self._lock:
                self._running_tasks[job_id] = {
                    'status': TrainingStatus.RUNNING.value,
                    'started_at': datetime.utcnow(),
                    'launcher_type': 'production',
                    'training_type': training_type
                }
            
            return {
                'success': True,
                'job_id': job_id,
                'status': TrainingStatus.RUNNING.value,
                'training_type': training_type,
                'message': '生产级训练已启动'
            }
            
        except (ResourceNotFoundError, BusinessLogicError):
            raise
        except Exception as e:
            logger.error(f"Failed to launch production training: {e}")
            raise BusinessLogicError(f"启动生产级训练失败: {str(e)}", operation="launch_production_training")
    
    def get_launcher_diagnostics(self) -> Dict[str, Any]:
        """获取启动器诊断信息"""
        try:
            return {
                'available': True,
                'diagnostics': diagnose_launcher_module(),
                'module_availability': get_module_availability()
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def _run_training(self, job: TrainingJob, from_checkpoint: bool = False):
        """运行训练（后台线程）"""
        try:
            # 模拟训练过程
            import time
            total_steps = job.total_steps or 100
            
            for step in range(total_steps):
                # 检查是否被取消或暂停
                with self._lock:
                    task_info = self._running_tasks.get(job.job_id)
                    if not task_info:
                        logger.info(f"Training cancelled: {job.job_id}")
                        return
                    if task_info.get('status') == TrainingStatus.PAUSED.value:
                        logger.info(f"Training paused: {job.job_id}")
                        return
                
                # 更新进度
                progress = ((step + 1) / total_steps) * 100
                self.update_progress(
                    job.job_id, 
                    str(job.tenant_id),
                    progress=progress,
                    current_step=step + 1,
                    current_epoch=1,
                    metrics={'loss': 1.0 - (step / total_steps) * 0.9}
                )
                
                time.sleep(0.1)  # 模拟训练时间
            
            # 完成训练
            self.complete_job(
                job.job_id,
                str(job.tenant_id),
                success=True,
                result={'final_loss': 0.1, 'accuracy': 0.95}
            )
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.complete_job(
                job.job_id,
                str(job.tenant_id),
                success=False,
                error_message=str(e)
            )
    
    def _pause_training_async(self, job: TrainingJob):
        """异步暂停训练"""
        with self._lock:
            if job.job_id in self._running_tasks:
                self._running_tasks[job.job_id]['status'] = TrainingStatus.PAUSED.value
        
        try:
            scenario_manager = self._get_scenario_manager()
            if scenario_manager:
                scenario_manager.pause_job(job.job_id)
        except Exception as e:
            logger.warning(f"Failed to pause via scenario manager: {e}")
    
    def _resume_training_async(self, job: TrainingJob):
        """异步恢复训练"""
        with self._lock:
            if job.job_id in self._running_tasks:
                self._running_tasks[job.job_id]['status'] = TrainingStatus.RUNNING.value
        
        try:
            scenario_manager = self._get_scenario_manager()
            if scenario_manager:
                scenario_manager.resume_job(job.job_id)
        except Exception as e:
            logger.warning(f"Failed to resume via scenario manager: {e}")
    
    def _cancel_training_async(self, job: TrainingJob):
        """异步取消训练"""
        with self._lock:
            if job.job_id in self._running_tasks:
                del self._running_tasks[job.job_id]
        
        try:
            scenario_manager = self._get_scenario_manager()
            if scenario_manager:
                scenario_manager.cancel_job(job.job_id)
        except Exception as e:
            logger.warning(f"Failed to cancel via scenario manager: {e}")
    
    def _get_scenario_manager(self):
        """获取场景管理器"""
        if self._scenario_manager is None:
            try:
                
                self._scenario_manager = get_scenario_manager()
            except ImportError:
                pass
        return self._scenario_manager
    
    def _get_task_manager(self):
        """获取任务管理器"""
        if self._task_manager is None:
            try:
                from backend.modules.training.core.task_manager import get_training_task_manager
                self._task_manager = get_training_task_manager()
            except ImportError:
                pass
        return self._task_manager
    
    def _is_scheduler_running(self) -> bool:
        """检查调度器是否运行"""
        try:
            scenario_manager = self._get_scenario_manager()
            if scenario_manager:
                return scenario_manager.scheduler_running
        except Exception:
            pass
        return True  # 默认认为运行中
    
    def _get_max_concurrent_jobs(self) -> int:
        """获取最大并发任务数"""
        try:
            scenario_manager = self._get_scenario_manager()
            if scenario_manager:
                return scenario_manager.max_concurrent_jobs
        except Exception:
            pass
        return 3  # 默认值


# 全局实例
_training_control_service = None
_service_lock = threading.Lock()


def get_training_control_service(use_memory_storage: bool = False) -> TrainingControlService:
    """获取训练控制服务实例
    
    Args:
        use_memory_storage: 是否使用内存存储
        
    Returns:
        训练控制服务实例
    """
    global _training_control_service
    
    if _training_control_service is None:
        with _service_lock:
            if _training_control_service is None:
                _training_control_service = TrainingControlService(use_memory_storage)
    
    return _training_control_service


def shutdown_training_control_service():
    """关闭训练控制服务"""
    global _training_control_service
    
    if _training_control_service:
        _training_control_service = None
        logger.info("Training control service shutdown")

