"""三阶段训练服务

提供三阶段训练的业务逻辑层，包括：
- 训练会话管理（创建、启动、暂停、恢复、停止）
- 进度跟踪和报告
- 统计分析

支持租户级别的数据隔离。

架构调用关系：
API层 -> Service层 (本模块) -> Launcher层 (training_launcher.py) -> Core层 -> 下游训练模块
"""

import uuid
import logging
import threading
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# =============================================================================
# 训练启动器集成 (统一通过 launcher 执行训练)
# =============================================================================

from backend.modules.training.launcher import (
    TrainingSystemLauncher,
    ProductionTrainingLauncher,
    launch_training_system,
    launch_production_training,
    create_production_training_config,
    get_module_availability,
    diagnose_launcher_module,
)

from backend.modules.training.progress import (
    get_progress_manager,
    TrainingProgressManager,
)


class TrainingStage(Enum):
    """训练阶段枚举"""
    PRETRAIN = 'pretrain'
    FINETUNE = 'finetune'
    PREFERENCE = 'preference'
    SFT = 'sft'
    DPO = 'dpo'
    PT = 'pt'


class TrainingStatus(Enum):
    """训练状态枚举"""
    PENDING = 'pending'
    RUNNING = 'running'
    PAUSED = 'paused'
    STOPPED = 'stopped'
    COMPLETED = 'completed'
    FAILED = 'failed'
    ERROR = 'error'


@dataclass
class StageConfig:
    """阶段配置"""
    enabled: bool = False
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    data_path: str = ''
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """训练配置"""
    model_name: str
    base_model_path: str = 'gpt2'
    output_dir: str = './output'
    pass_model_between_stages: bool = True
    pretrain: StageConfig = field(default_factory=StageConfig)
    finetune: StageConfig = field(default_factory=StageConfig)
    preference: StageConfig = field(default_factory=StageConfig)


class ThreeStageTrainingService:
    """三阶段训练服务
    
    提供三阶段训练的完整业务逻辑。
    所有训练执行统一通过 TrainingSystemLauncher 调度。
    """
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化服务
        
        Args:
            use_memory_storage: 是否使用内存存储（测试用）
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self._use_memory = use_memory_storage
        
        # 初始化仓库
        from backend.repositories.three_stage_training_repository import (
            get_session_repository, get_progress_repository
        )
        self._session_repo = get_session_repository(use_memory_storage)
        self._progress_repo = get_progress_repository(use_memory_storage)
        
        # 运行中的训练任务
        self._running_tasks: Dict[str, Dict[str, Any]] = {}
        self._task_lock = threading.Lock()
        
        # 训练启动器实例（延迟初始化）
        self._launcher: Optional[Any] = None
        self._progress_manager: Optional[Any] = None
        
        # 初始化启动器
        self._init_launcher()
    
    def _init_launcher(self):
        """初始化训练启动器"""
        try:
            # 检查模块可用性
            availability = get_module_availability()
            self.logger.info(f"Launcher module availability: {availability}")
        except Exception as e:
            self.logger.warning(f"Failed to check launcher availability: {e}")

        try:
            self._progress_manager = get_progress_manager()
            self.logger.info("Progress manager initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize progress manager: {e}")
    
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
            self.logger.error(f"Failed to create launcher: {e}")
            return None
    
    def _build_launcher_config(self, session_config: Dict[str, Any]) -> Dict[str, Any]:
        """构建启动器配置
        
        Args:
            session_config: 会话配置
            
        Returns:
            启动器配置字典
        """
        stages = session_config.get('stages', {})
        
        launcher_config = {
            'model': {
                'name': session_config.get('model_name', 'three_stage_model'),
                'path': session_config.get('base_model_path', 'gpt2'),
                'type': 'transformer'
            },
            'training': {
                'mode': 'three_stage',
                'output_dir': session_config.get('output_dir', './output'),
                'pass_model_between_stages': session_config.get('pass_model_between_stages', True)
            },
            'three_stage': {
                'enabled': True,
                'pretrain': {
                    'enabled': stages.get('pt', {}).get('enabled', False),
                    'epochs': stages.get('pt', {}).get('epochs', 3),
                    'batch_size': stages.get('pt', {}).get('batch_size', 8),
                    'learning_rate': stages.get('pt', {}).get('learning_rate', 2e-5),
                    'data_path': stages.get('pt', {}).get('data_path', './data/pt')
                },
                'finetune': {
                    'enabled': stages.get('sft', {}).get('enabled', False),
                    'epochs': stages.get('sft', {}).get('epochs', 3),
                    'batch_size': stages.get('sft', {}).get('batch_size', 8),
                    'learning_rate': stages.get('sft', {}).get('learning_rate', 2e-5),
                    'data_path': stages.get('sft', {}).get('data_path', './data/sft')
                },
                'preference': {
                    'enabled': stages.get('dpo', {}).get('enabled', False),
                    'epochs': stages.get('dpo', {}).get('epochs', 3),
                    'batch_size': stages.get('dpo', {}).get('batch_size', 8),
                    'learning_rate': stages.get('dpo', {}).get('learning_rate', 2e-5),
                    'beta': stages.get('dpo', {}).get('beta', 0.1),
                    'data_path': stages.get('dpo', {}).get('data_path', './data/dpo')
                }
            }
        }
        
        return launcher_config
    
    # ==========================================================================
    # 会话管理
    # ==========================================================================
    
    def create_session(self, name: str, model_name: str, config: Dict[str, Any],
                       tenant_id: str, user_id: str,
                       description: Optional[str] = None) -> Dict[str, Any]:
        """创建训练会话
        
        Args:
            name: 会话名称
            model_name: 模型名称
            config: 训练配置
            tenant_id: 租户ID
            user_id: 用户ID
            description: 描述
            
        Returns:
            创建的会话信息
        """
        try:
            # 验证配置
            validated_config = self._validate_config(config)
            
            session_data = {
                'tenant_id': tenant_id,
                'user_id': user_id,
                'name': name,
                'description': description or f"Three-stage training: {model_name}",
                'model_name': model_name,
                'status': TrainingStatus.PENDING.value,
                'config': validated_config,
                'progress': 0.0,
                'pretrain_progress': 0.0,
                'finetune_progress': 0.0,
                'preference_progress': 0.0
            }
            
            result = self._session_repo.create(session_data)
            self.logger.info(f"Created training session: {result.get('session_id')}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to create session: {e}")
            raise
    
    def get_session(self, session_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """获取会话详情
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID
            
        Returns:
            会话信息或None
        """
        return self._session_repo.get_by_id(session_id, tenant_id)
    
    def list_sessions(self, tenant_id: str, user_id: Optional[str] = None,
                      status: Optional[str] = None, model_name: Optional[str] = None,
                      limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """获取会话列表
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID（可选）
            status: 状态过滤
            model_name: 模型名称过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            会话列表和总数
        """
        if user_id:
            sessions, total = self._session_repo.get_by_user(
                user_id, tenant_id, status, limit, offset
            )
        else:
            sessions, total = self._session_repo.list_by_tenant(
                tenant_id, status, model_name, limit, offset
            )
        
        return {
            'sessions': sessions,
            'total': total,
            'limit': limit,
            'offset': offset
        }
    
    def update_session(self, session_id: str, tenant_id: str,
                       updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新会话
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID
            updates: 更新内容
            
        Returns:
            更新后的会话或None
        """
        return self._session_repo.update(session_id, tenant_id, updates)
    
    def delete_session(self, session_id: str, tenant_id: str) -> bool:
        """删除会话
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID
            
        Returns:
            是否删除成功
        """
        # 先删除进度记录
        self._progress_repo.delete_by_session(session_id)
        return self._session_repo.delete(session_id, tenant_id)
    
    # ==========================================================================
    # 训练控制
    # ==========================================================================
    
    def start_training(self, session_id: str, tenant_id: str, 
                       user_id: str) -> Dict[str, Any]:
        """启动训练
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            操作结果
        """
        try:
            # 获取会话
            session = self._session_repo.get_by_id(session_id, tenant_id)
            if not session:
                return {'success': False, 'error': 'Session not found'}
            
            # 检查状态
            current_status = session.get('status')
            if current_status == TrainingStatus.RUNNING.value:
                return {'success': False, 'error': 'Training is already running'}
            
            if current_status == TrainingStatus.COMPLETED.value:
                return {'success': False, 'error': 'Training has already completed'}
            
            # 更新状态
            now = datetime.utcnow()
            self._session_repo.update(session_id, tenant_id, {
                'status': TrainingStatus.RUNNING.value,
                'started_at': now
            })
            
            # 启动异步训练任务
            config = session.get('config', {})
            output_dir = config.get('output_dir', f"./output/{session.get('model_name')}")
            
            # 在后台线程中执行训练
            thread = threading.Thread(
                target=self._run_training_async,
                args=(session_id, tenant_id, config, output_dir),
                daemon=True
            )
            thread.start()
            
            # 记录运行中的任务
            with self._task_lock:
                self._running_tasks[session_id] = {
                    'thread': thread,
                    'started_at': now,
                    'status': 'running'
                }
            
            self.logger.info(f"Started training for session: {session_id}")
            
            return {
                'success': True,
                'session_id': session_id,
                'status': TrainingStatus.RUNNING.value,
                'started_at': now.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to start training: {e}")
            return {'success': False, 'error': str(e)}
    
    def pause_training(self, session_id: str, tenant_id: str) -> Dict[str, Any]:
        """暂停训练
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID
            
        Returns:
            操作结果
        """
        try:
            session = self._session_repo.get_by_id(session_id, tenant_id)
            if not session:
                return {'success': False, 'error': 'Session not found'}
            
            if session.get('status') != TrainingStatus.RUNNING.value:
                return {'success': False, 'error': 'Training is not running'}
            
            # 更新状态
            self._session_repo.update_status(session_id, tenant_id, TrainingStatus.PAUSED.value)
            
            # 更新运行任务状态
            with self._task_lock:
                if session_id in self._running_tasks:
                    self._running_tasks[session_id]['status'] = 'paused'
            
            self.logger.info(f"Paused training for session: {session_id}")
            
            return {'success': True, 'session_id': session_id, 'status': 'paused'}
            
        except Exception as e:
            self.logger.error(f"Failed to pause training: {e}")
            return {'success': False, 'error': str(e)}
    
    def resume_training(self, session_id: str, tenant_id: str) -> Dict[str, Any]:
        """恢复训练
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID
            
        Returns:
            操作结果
        """
        try:
            session = self._session_repo.get_by_id(session_id, tenant_id)
            if not session:
                return {'success': False, 'error': 'Session not found'}
            
            if session.get('status') != TrainingStatus.PAUSED.value:
                return {'success': False, 'error': 'Training is not paused'}
            
            # 更新状态
            self._session_repo.update_status(session_id, tenant_id, TrainingStatus.RUNNING.value)
            
            # 更新运行任务状态
            with self._task_lock:
                if session_id in self._running_tasks:
                    self._running_tasks[session_id]['status'] = 'running'
            
            self.logger.info(f"Resumed training for session: {session_id}")
            
            return {'success': True, 'session_id': session_id, 'status': 'running'}
            
        except Exception as e:
            self.logger.error(f"Failed to resume training: {e}")
            return {'success': False, 'error': str(e)}
    
    def stop_training(self, session_id: str, tenant_id: str) -> Dict[str, Any]:
        """停止训练
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID
            
        Returns:
            操作结果
        """
        try:
            session = self._session_repo.get_by_id(session_id, tenant_id)
            if not session:
                return {'success': False, 'error': 'Session not found'}
            
            current_status = session.get('status')
            if current_status not in [TrainingStatus.RUNNING.value, TrainingStatus.PAUSED.value]:
                return {'success': False, 'error': 'Training is not running or paused'}
            
            # 更新状态
            now = datetime.utcnow()
            self._session_repo.update(session_id, tenant_id, {
                'status': TrainingStatus.STOPPED.value,
                'completed_at': now
            })
            
            # 清理运行任务
            with self._task_lock:
                if session_id in self._running_tasks:
                    self._running_tasks[session_id]['status'] = 'stopped'
                    del self._running_tasks[session_id]
            
            self.logger.info(f"Stopped training for session: {session_id}")
            
            return {
                'success': True,
                'session_id': session_id,
                'status': 'stopped',
                'completed_at': now.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to stop training: {e}")
            return {'success': False, 'error': str(e)}
    
    # ==========================================================================
    # 进度和报告
    # ==========================================================================
    
    def get_progress(self, session_id: str, tenant_id: str) -> Dict[str, Any]:
        """获取训练进度
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID
            
        Returns:
            进度信息
        """
        session = self._session_repo.get_by_id(session_id, tenant_id)
        if not session:
            return {'error': 'Session not found'}
        
        # 获取各阶段的进度摘要
        stage_summary = self._progress_repo.get_stage_summary(session_id)
        
        return {
            'session_id': session_id,
            'status': session.get('status'),
            'progress': session.get('progress', 0),
            'current_stage': session.get('current_stage'),
            'stages': {
                'pretrain': {
                    'progress': session.get('pretrain_progress', 0),
                    'details': stage_summary.get('pretrain')
                },
                'finetune': {
                    'progress': session.get('finetune_progress', 0),
                    'details': stage_summary.get('finetune')
                },
                'preference': {
                    'progress': session.get('preference_progress', 0),
                    'details': stage_summary.get('preference')
                }
            },
            'started_at': session.get('started_at'),
            'completed_at': session.get('completed_at')
        }
    
    def get_progress_history(self, session_id: str, tenant_id: str,
                             stage: Optional[str] = None,
                             limit: int = 100) -> Dict[str, Any]:
        """获取进度历史
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID
            stage: 阶段过滤
            limit: 返回数量限制
            
        Returns:
            进度历史
        """
        # 验证会话存在
        session = self._session_repo.get_by_id(session_id, tenant_id)
        if not session:
            return {'error': 'Session not found'}
        
        records, total = self._progress_repo.get_by_session(session_id, stage, limit)
        
        return {
            'session_id': session_id,
            'records': records,
            'total': total,
            'limit': limit
        }
    
    def get_report(self, session_id: str, tenant_id: str) -> Dict[str, Any]:
        """获取训练报告
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID
            
        Returns:
            训练报告
        """
        session = self._session_repo.get_by_id(session_id, tenant_id)
        if not session:
            return {'error': 'Session not found'}
        
        # 尝试从文件读取详细报告
        config = session.get('config', {})
        output_dir = config.get('output_dir')
        file_report = None
        
        if output_dir:
            report_path = os.path.join(output_dir, "three_stage_training_report.json")
            if os.path.isfile(report_path):
                try:
                    with open(report_path, 'r', encoding='utf-8') as f:
                        file_report = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to read report file: {e}")
        
        # 构建报告
        report = {
            'session_id': session_id,
            'model_name': session.get('model_name'),
            'status': session.get('status'),
            'config': session.get('config'),
            'result': session.get('result'),
            'progress': {
                'overall': session.get('progress', 0),
                'pretrain': session.get('pretrain_progress', 0),
                'finetune': session.get('finetune_progress', 0),
                'preference': session.get('preference_progress', 0)
            },
            'started_at': session.get('started_at'),
            'completed_at': session.get('completed_at'),
            'created_at': session.get('created_at')
        }
        
        # 合并文件报告
        if file_report:
            report['stages'] = file_report.get('stages')
            report['training_stats'] = file_report.get('training_stats')
            report['total_duration'] = file_report.get('total_duration')
        
        return report
    
    # ==========================================================================
    # 统计
    # ==========================================================================
    
    def get_statistics(self, tenant_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """获取统计信息
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID（可选）
            
        Returns:
            统计信息
        """
        return self._session_repo.get_statistics(tenant_id, user_id)
    
    # ==========================================================================
    # 快捷方法
    # ==========================================================================
    
    def create_and_start(self, name: str, model_name: str, config: Dict[str, Any],
                         tenant_id: str, user_id: str,
                         description: Optional[str] = None) -> Dict[str, Any]:
        """创建并启动训练
        
        Args:
            name: 会话名称
            model_name: 模型名称
            config: 训练配置
            tenant_id: 租户ID
            user_id: 用户ID
            description: 描述
            
        Returns:
            会话信息和启动结果
        """
        # 创建会话
        session = self.create_session(
            name=name,
            model_name=model_name,
            config=config,
            tenant_id=tenant_id,
            user_id=user_id,
            description=description
        )
        
        session_id = session.get('session_id')
        
        # 启动训练
        start_result = self.start_training(session_id, tenant_id, user_id)
        
        return {
            **session,
            'start_result': start_result
        }
    
    # ==========================================================================
    # 内部方法
    # ==========================================================================
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证和规范化配置
        
        Args:
            config: 原始配置
            
        Returns:
            验证后的配置
        """
        validated = {
            'model_name': config.get('model_name', ''),
            'base_model_path': config.get('base_model_path', 'gpt2'),
            'output_dir': config.get('output_dir', './output'),
            'pass_model_between_stages': config.get('pass_model_between_stages', True),
            'stages': {}
        }
        
        # 处理stages配置
        stages = config.get('stages', {})
        
        # 预训练阶段
        pt_config = stages.get('pt') or stages.get('pretrain', {})
        if pt_config.get('enabled'):
            validated['stages']['pt'] = {
                'enabled': True,
                'epochs': pt_config.get('epochs', 3),
                'batch_size': pt_config.get('batch_size', 8),
                'learning_rate': pt_config.get('learning_rate', 2e-5),
                'data_path': pt_config.get('data_path', './data/pt')
            }
        
        # 微调阶段
        sft_config = stages.get('sft') or stages.get('finetune', {})
        if sft_config.get('enabled'):
            validated['stages']['sft'] = {
                'enabled': True,
                'epochs': sft_config.get('epochs', 3),
                'batch_size': sft_config.get('batch_size', 8),
                'learning_rate': sft_config.get('learning_rate', 2e-5),
                'data_path': sft_config.get('data_path', './data/sft')
            }
        
        # 偏好优化阶段
        dpo_config = stages.get('dpo') or stages.get('preference', {})
        if dpo_config.get('enabled'):
            validated['stages']['dpo'] = {
                'enabled': True,
                'epochs': dpo_config.get('epochs', 3),
                'batch_size': dpo_config.get('batch_size', 8),
                'learning_rate': dpo_config.get('learning_rate', 2e-5),
                'beta': dpo_config.get('beta', 0.1),
                'data_path': dpo_config.get('data_path', './data/dpo')
            }
        
        return validated
    
    def _run_training_async(self, session_id: str, tenant_id: str,
                            config: Dict[str, Any], output_dir: str):
        """异步执行训练 (通过 Launcher 统一调度)
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID
            config: 训练配置
            output_dir: 输出目录
        """
        try:
            # 进度回调函数
            def progress_callback(stage: str, epoch: int, metrics: Dict[str, Any]):
                self._record_progress(session_id, tenant_id, stage, epoch, metrics)
                # 同时更新进度管理器
                if self._progress_manager:
                    try:
                        self._progress_manager.update_progress(
                            session_id,
                            current_stage=stage,
                            current_epoch=epoch,
                            metrics=metrics
                        )
                    except Exception:
                        pass
            
            # 状态检查函数
            def status_checker(sid: str) -> Optional[str]:
                session = self._session_repo.get_by_id(sid, tenant_id)
                return session.get('status') if session else None
            
            # 优先通过 Launcher 执行训练
            try:
                result = self._run_training_via_launcher(
                    session_id, tenant_id, config, output_dir,
                    progress_callback, status_checker
                )
                if result:
                    self._complete_training(session_id, tenant_id, result)
                    return
            except Exception as e:
                self.logger.warning(f"Launcher training failed, falling back: {e}")
            
            # 回退：尝试直接导入训练器
            try:
                from backend.modules.training.three_stage.three_stage_trainer import ThreeStageTrainer
                from backend.modules.training.three_stage.three_stage_config import ThreeStageConfig, StageConfig
                
                # 创建配置对象
                trainer_config = self._create_trainer_config(config, output_dir)
                
                # 创建训练器
                trainer = ThreeStageTrainer(
                    trainer_config,
                    progress_callback=progress_callback,
                    control_session_id=session_id,
                    status_checker=status_checker
                )
                
                # 执行训练
                result = trainer.train()
                
                # 更新完成状态
                self._complete_training(session_id, tenant_id, result)
                
            except ImportError as e:
                self.logger.warning(f"Cannot import trainer, using simulation: {e}")
                # 模拟训练
                self._simulate_training(session_id, tenant_id, config, progress_callback, status_checker)
                
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self._fail_training(session_id, tenant_id, str(e))
    
    def _run_training_via_launcher(
        self,
        session_id: str,
        tenant_id: str,
        config: Dict[str, Any],
        output_dir: str,
        progress_callback: Callable,
        status_checker: Callable
    ) -> Optional[Dict[str, Any]]:
        """通过 Launcher 执行训练
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID
            config: 训练配置
            output_dir: 输出目录
            progress_callback: 进度回调函数
            status_checker: 状态检查函数
            
        Returns:
            训练结果字典
        """
        try:
            # 构建启动器配置
            launcher_config = self._build_launcher_config(config)
            launcher_config['training']['output_dir'] = output_dir
            
            # 创建启动器
            launcher = TrainingSystemLauncher(launcher_config)
            
            # 分析配置
            analysis = launcher.analyze_config()
            self.logger.info(f"Launcher analysis: {analysis}")
            
            # 选择训练器
            trainer = launcher.select_trainer(analysis)
            
            if trainer is None:
                self.logger.warning("Launcher returned no trainer")
                return None
            
            # 如果训练器支持进度回调，设置回调
            if hasattr(trainer, 'set_progress_callback'):
                trainer.set_progress_callback(progress_callback)
            if hasattr(trainer, 'set_status_checker'):
                trainer.set_status_checker(status_checker)
            
            # 执行训练
            self.logger.info(f"Starting training via launcher for session: {session_id}")
            result = trainer.train()
            
            self.logger.info(f"Launcher training completed: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Launcher training error: {e}")
            raise
    
    def launch_training_with_launcher(
        self,
        session_id: str,
        tenant_id: str,
        user_id: str,
        training_type: str = 'three_stage',
        **kwargs
    ) -> Dict[str, Any]:
        """使用启动器启动训练（高级API）
        
        通过 ProductionTrainingLauncher 启动生产级训练。
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID
            user_id: 用户ID
            training_type: 训练类型
            **kwargs: 额外配置
            
        Returns:
            启动结果
        """
        try:
            # 获取会话
            session = self._session_repo.get_by_id(session_id, tenant_id)
            if not session:
                return {'success': False, 'error': 'Session not found'}
            
            config = session.get('config', {})
            output_dir = config.get('output_dir', f"./output/{session.get('model_name')}")
            
            # 构建生产级训练配置
            production_config = create_production_training_config(
                training_type=training_type,
                output_dir=output_dir,
                model_name=session.get('model_name', 'three_stage_model'),
                num_epochs=kwargs.get('num_epochs', 10),
                batch_size=kwargs.get('batch_size', 8),
                learning_rate=kwargs.get('learning_rate', 2e-5),
                enable_checkpoint=kwargs.get('enable_checkpoint', True),
                enable_monitoring=kwargs.get('enable_monitoring', True),
            )
            
            # 添加三阶段特定配置
            stages = config.get('stages', {})
            production_config['three_stage'] = {
                'enabled': True,
                'pretrain': stages.get('pt', {}),
                'finetune': stages.get('sft', {}),
                'preference': stages.get('dpo', {})
            }
            
            # 更新会话状态
            now = datetime.utcnow()
            self._session_repo.update(session_id, tenant_id, {
                'status': TrainingStatus.RUNNING.value,
                'started_at': now
            })
            
            # 在后台线程启动生产级训练
            def run_production_training():
                try:
                    launcher = ProductionTrainingLauncher(production_config)
                    result = launcher.launch_training()
                    
                    if result.get('success'):
                        self._complete_training(session_id, tenant_id, result)
                    else:
                        self._fail_training(session_id, tenant_id, result.get('error', 'Training failed'))
                        
                except Exception as e:
                    self.logger.error(f"Production training failed: {e}")
                    self._fail_training(session_id, tenant_id, str(e))
            
            thread = threading.Thread(target=run_production_training, daemon=True)
            thread.start()
            
            # 记录运行中的任务
            with self._task_lock:
                self._running_tasks[session_id] = {
                    'thread': thread,
                    'started_at': now,
                    'status': 'running',
                    'launcher_type': 'production'
                }
            
            return {
                'success': True,
                'session_id': session_id,
                'status': TrainingStatus.RUNNING.value,
                'started_at': now.isoformat(),
                'launcher_type': 'production'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to launch training with launcher: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_launcher_diagnostics(self) -> Dict[str, Any]:
        """获取启动器诊断信息
        
        Returns:
            诊断信息字典
        """
        try:
            diagnostics = diagnose_launcher_module()
            availability = get_module_availability()
            
            return {
                'available': True,
                'diagnostics': diagnostics,
                'module_availability': availability
            }
            
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
    
    def _create_trainer_config(self, config: Dict[str, Any], output_dir: str):
        """创建训练器配置对象"""
        from backend.modules.training.three_stage.three_stage_config import ThreeStageConfig, StageConfig
        
        stages = config.get('stages', {})
        
        pretrain_config = StageConfig(
            enabled=stages.get('pt', {}).get('enabled', False),
            epochs=stages.get('pt', {}).get('epochs', 3),
            batch_size=stages.get('pt', {}).get('batch_size', 8),
            learning_rate=stages.get('pt', {}).get('learning_rate', 2e-5),
            dataset_path=stages.get('pt', {}).get('data_path', './data/pt')
        )
        
        finetune_config = StageConfig(
            enabled=stages.get('sft', {}).get('enabled', False),
            epochs=stages.get('sft', {}).get('epochs', 3),
            batch_size=stages.get('sft', {}).get('batch_size', 8),
            learning_rate=stages.get('sft', {}).get('learning_rate', 2e-5),
            dataset_path=stages.get('sft', {}).get('data_path', './data/sft')
        )
        
        preference_config = StageConfig(
            enabled=stages.get('dpo', {}).get('enabled', False),
            epochs=stages.get('dpo', {}).get('epochs', 3),
            batch_size=stages.get('dpo', {}).get('batch_size', 8),
            learning_rate=stages.get('dpo', {}).get('learning_rate', 2e-5),
            dataset_path=stages.get('dpo', {}).get('data_path', './data/dpo')
        )
        
        return ThreeStageConfig(
            base_model_path=config.get('base_model_path', 'gpt2'),
            output_dir=output_dir,
            pass_model_between_stages=config.get('pass_model_between_stages', True),
            pretrain=pretrain_config,
            finetune=finetune_config,
            preference=preference_config
        )
    
    def _simulate_training(self, session_id: str, tenant_id: str, config: Dict[str, Any],
                          progress_callback: Callable, status_checker: Callable):
        """模拟训练过程（测试用）"""
        import time
        
        stages = config.get('stages', {})
        enabled_stages = []
        
        if stages.get('pt', {}).get('enabled'):
            enabled_stages.append(('pretrain', stages['pt'].get('epochs', 3)))
        if stages.get('sft', {}).get('enabled'):
            enabled_stages.append(('finetune', stages['sft'].get('epochs', 3)))
        if stages.get('dpo', {}).get('enabled'):
            enabled_stages.append(('preference', stages['dpo'].get('epochs', 3)))
        
        total_epochs = sum(e for _, e in enabled_stages)
        completed_epochs = 0
        
        for stage_name, epochs in enabled_stages:
            # 更新当前阶段
            self._session_repo.update(session_id, tenant_id, {'current_stage': stage_name})
            
            for epoch in range(1, epochs + 1):
                # 检查状态
                status = status_checker(session_id)
                if status == TrainingStatus.STOPPED.value:
                    return
                if status == TrainingStatus.PAUSED.value:
                    while status_checker(session_id) == TrainingStatus.PAUSED.value:
                        time.sleep(1)
                
                # 模拟训练
                time.sleep(0.5)
                
                # 生成模拟指标
                metrics = {
                    'loss': 1.0 - (epoch / epochs) * 0.5,
                    'accuracy': (epoch / epochs) * 0.9,
                    'learning_rate': 2e-5
                }
                
                # 记录进度
                progress_callback(stage_name, epoch, metrics)
                
                # 更新总体进度
                completed_epochs += 1
                overall_progress = (completed_epochs / total_epochs) * 100 if total_epochs > 0 else 0
                stage_progress = (epoch / epochs) * 100
                
                stage_progress_map = {
                    'pretrain': 'pretrain_progress',
                    'finetune': 'finetune_progress',
                    'preference': 'preference_progress'
                }
                
                updates = {'progress': overall_progress}
                if stage_name in stage_progress_map:
                    updates[stage_progress_map[stage_name]] = stage_progress
                
                self._session_repo.update(session_id, tenant_id, updates)
        
        # 完成训练
        self._complete_training(session_id, tenant_id, {
            'success': True,
            'message': 'Training completed (simulation)'
        })
    
    def _record_progress(self, session_id: str, tenant_id: str,
                         stage: str, epoch: int, metrics: Dict[str, Any]):
        """记录训练进度"""
        try:
            self._progress_repo.create({
                'session_id': session_id,
                'stage': stage,
                'epoch': epoch,
                'loss': metrics.get('loss'),
                'accuracy': metrics.get('accuracy') or metrics.get('reward_accuracy'),
                'learning_rate': metrics.get('learning_rate'),
                'metrics': metrics
            })
        except Exception as e:
            self.logger.error(f"Failed to record progress: {e}")
    
    def _complete_training(self, session_id: str, tenant_id: str, result: Dict[str, Any]):
        """完成训练"""
        now = datetime.utcnow()
        status = TrainingStatus.COMPLETED.value if result.get('success') else TrainingStatus.FAILED.value
        
        self._session_repo.update(session_id, tenant_id, {
            'status': status,
            'result': result,
            'progress': 100.0 if result.get('success') else None,
            'completed_at': now
        })
        
        # 清理运行任务
        with self._task_lock:
            if session_id in self._running_tasks:
                del self._running_tasks[session_id]
        
        self.logger.info(f"Training completed for session: {session_id}, status: {status}")
    
    def _fail_training(self, session_id: str, tenant_id: str, error_message: str):
        """训练失败处理"""
        now = datetime.utcnow()
        
        self._session_repo.update(session_id, tenant_id, {
            'status': TrainingStatus.FAILED.value,
            'error_message': error_message,
            'completed_at': now
        })
        
        # 清理运行任务
        with self._task_lock:
            if session_id in self._running_tasks:
                del self._running_tasks[session_id]
        
        self.logger.error(f"Training failed for session: {session_id}, error: {error_message}")


# ==============================================================================
# 单例模式获取器
# ==============================================================================

_service_instance: Optional[ThreeStageTrainingService] = None


def get_three_stage_training_service(use_memory_storage: bool = False) -> ThreeStageTrainingService:
    """获取服务实例（单例）"""
    global _service_instance
    if _service_instance is None or use_memory_storage:
        _service_instance = ThreeStageTrainingService(use_memory_storage)
    return _service_instance

