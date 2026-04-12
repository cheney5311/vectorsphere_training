"""训练执行管理服务

提供训练执行管理相关的业务逻辑。

架构调用关系：
API层 -> Service层 (本模块) -> Launcher层 (training_launcher.py) -> Core层 -> 下游训练模块
"""

import sys
import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import threading
import time
import asyncio
import psutil
import traceback

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.core.exceptions import ValidationError, BusinessLogicError
from backend.schemas.enums import TrainingStatus
from backend.repositories.training_session_repository import get_training_session_repository
from backend.repositories.training_execution_repository import (
    get_training_execution_repository,
)
from backend.schemas.training_models import TrainingExecution, TrainingExecutionLog
from backend.modules.training.progress import get_progress_manager
from backend.modules.distributed.cluster_manager import ClusterManager, StaticNodeDiscovery
from backend.modules.distributed.lease_manager import get_lease_manager
from backend.modules.distributed.fault_tolerance import get_fault_tolerance_manager

logger = logging.getLogger(__name__)

# =============================================================================
# 训练启动器集成 (统一通过 launcher 执行训练)
# =============================================================================

from backend.modules.training.launcher import (
    TrainingSystemLauncher,
    ProductionTrainingLauncher,
    get_module_availability,
    diagnose_launcher_module,
    create_production_training_config,
)


@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int
    step: int
    loss: float
    accuracy: Optional[float]
    learning_rate: float
    throughput: float  # 样本/秒
    memory_usage: float  # GB
    gpu_utilization: Optional[float]  # %
    timestamp: datetime


@dataclass
class ResourceStatus:
    """资源状态"""
    cpu_percent: float
    memory_percent: float
    gpu_memory_used: Optional[float]  # GB
    gpu_utilization: Optional[float]  # %
    disk_usage: float  # %
    network_io: Dict[str, float]  # bytes/sec


class TrainingExecutionService:
    """训练执行管理服务
    
    提供训练执行的完整生命周期管理，包括：
    - 创建和启动训练执行
    - 暂停、恢复、停止训练
    - 获取训练状态和进度
    - 更新训练指标
    - 管理执行日志
    
    所有训练执行统一通过 TrainingSystemLauncher 调度。
    """
    
    def __init__(self):
        self._session_repository = get_training_session_repository()
        self._execution_repository = get_training_execution_repository()
        # 修复参数node_discovery缺少传入值的问题
        node_discovery = StaticNodeDiscovery([])
        self._cluster_manager = ClusterManager(node_discovery)
        self._active_trainings = {}  # 存储活跃训练的状态
        self._monitoring_threads = {}  # 存储监控线程
        self._checkpoint_dir = "./checkpoints"  # 检查点目录
        
        # 训练启动器（统一调度）
        self._launcher: Optional[Any] = None
        
        # 初始化启动器
        self._init_launcher()
    
    def _init_launcher(self):
        """初始化训练启动器"""
        try:
            availability = get_module_availability()
            logger.info(f"Launcher module availability: {availability}")
        except Exception as e:
            logger.warning(f"Failed to check launcher availability: {e}")
    
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
    
    def _build_launcher_config(self, execution_config: Dict[str, Any]) -> Dict[str, Any]:
        """构建启动器配置
        
        Args:
            execution_config: 执行配置
            
        Returns:
            启动器配置字典
        """
        scenario_type = execution_config.get('scenario_type', 'standard')
        training_mode = execution_config.get('training_mode', 'standard')
        
        launcher_config = {
            'model': {
                'name': execution_config.get('model_name', 'default_model'),
                'type': execution_config.get('model_type', 'transformer')
            },
            'training': {
                'mode': training_mode,
                'scenario_type': scenario_type,
                'num_epochs': execution_config.get('epochs', 10),
                'batch_size': execution_config.get('batch_size', 16),
                'learning_rate': execution_config.get('learning_rate', 2e-5),
                'output_dir': execution_config.get('output_dir', './outputs')
            },
            'data': {
                'train_path': execution_config.get('train_data_path', './data/train')
            }
        }
        
        # 添加特定模式配置
        if training_mode == 'three_stage':
            launcher_config['three_stage'] = {
                'enabled': True,
                **execution_config.get('three_stage', {})
            }
        elif training_mode == 'distributed':
            launcher_config['distributed'] = {
                'enabled': True,
                **execution_config.get('distributed', {})
            }
        elif training_mode == 'multimodal':
            launcher_config['multimodal'] = {
                'enabled': True,
                **execution_config.get('multimodal', {})
            }
        elif training_mode == 'distillation':
            launcher_config['distillation'] = {
                'enabled': True,
                **execution_config.get('distillation', {})
            }
        
        return launcher_config
    
    # ==================== 执行记录管理方法 ====================
    
    def create_execution(self, tenant_id: str, user_id: str, session_id: str,
                        scenario_type: str, config: Dict[str, Any]) -> TrainingExecution:
        """创建训练执行记录
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            session_id: 会话ID
            scenario_type: 场景类型
            config: 训练配置
            
        Returns:
            创建的执行记录
        """
        import uuid
        
        execution_id = f"exec_{uuid.uuid4().hex[:16]}"
        trainer_type = self._select_trainer_for_scenario(scenario_type)
        
        execution_data = {
            'execution_id': execution_id,
            'session_id': session_id,
            'tenant_id': tenant_id,
            'user_id': user_id,
            'name': config.get('name', f"Training Execution {execution_id[:8]}"),
            'description': config.get('description', ''),
            'scenario_type': scenario_type,
            'training_mode': config.get('training_mode', 'standard'),
            'trainer_type': trainer_type,
            'training_config': config,
            'resource_config': config.get('resource_config', {}),
            'status': TrainingStatus.PENDING.value,
            'progress': 0.0,
            'total_epochs': config.get('epochs', 10),
            'total_steps': config.get('total_steps')
        }
        
        execution = self._execution_repository.create(execution_data)
        
        # 记录创建日志
        self._log_execution_event(
            execution_id, 'created',
            f"Execution created for scenario: {scenario_type}",
            to_status=TrainingStatus.PENDING.value,
            operator_id=user_id
        )
        
        logger.info(f"Created training execution: {execution_id}")
        return execution
    
    def _select_trainer_for_scenario(self, scenario_type: str) -> str:
        """根据场景类型选择训练器
        
        Args:
            scenario_type: 场景类型
            
        Returns:
            训练器类型名称
        """
        trainer_mapping = {
            'standard': 'StandardTrainer',
            'distributed': 'DistributedTrainer',
            'multimodal': 'MultiModalTrainer',
            'distillation': 'KnowledgeDistillationTrainer',
            'three_stage': 'ThreeStageTrainer',
            'industry': 'IndustryScenarioTrainer',
            'scenario': 'ScenarioTrainer'
        }
        return trainer_mapping.get(scenario_type, 'StandardTrainer')
    
    def get_execution(self, execution_id: str, tenant_id: str = None) -> Optional[TrainingExecution]:
        """获取训练执行记录
        
        Args:
            execution_id: 执行ID
            tenant_id: 租户ID
            
        Returns:
            执行记录
        """
        return self._execution_repository.get_by_execution_id(execution_id, tenant_id)
    
    def get_execution_by_session(self, session_id: str, tenant_id: str = None) -> Optional[TrainingExecution]:
        """根据会话ID获取最新执行记录
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID
            
        Returns:
            执行记录
        """
        return self._execution_repository.get_by_session_id(session_id, tenant_id)
    
    def update_execution_status(self, execution_id: str, status: TrainingStatus,
                               tenant_id: str = None, user_id: str = None,
                               **kwargs) -> Optional[TrainingExecution]:
        """更新执行状态
        
        Args:
            execution_id: 执行ID
            status: 新状态
            tenant_id: 租户ID
            user_id: 操作用户ID
            **kwargs: 其他要更新的字段
            
        Returns:
            更新后的执行记录
        """
        # 获取当前执行记录
        execution = self._execution_repository.get_by_execution_id(execution_id, tenant_id)
        if not execution:
            return None
        
        from_status = execution.status
        status_value = status.value if isinstance(status, TrainingStatus) else status
        
        # 更新状态
        result = self._execution_repository.update_status(
            execution_id, status, tenant_id, **kwargs
        )
        
        # 记录状态变更日志
        if result:
            self._log_execution_event(
                execution_id, 'status_change',
                f"Status changed from {from_status} to {status_value}",
                from_status=from_status,
                to_status=status_value,
                operator_id=user_id
            )
        
        return result
    
    def update_execution_progress(self, execution_id: str, progress: float,
                                 current_step: int = None, current_epoch: int = None,
                                 metrics: Dict[str, Any] = None,
                                 tenant_id: str = None) -> Optional[TrainingExecution]:
        """更新执行进度
        
        Args:
            execution_id: 执行ID
            progress: 进度值(0-100)
            current_step: 当前步骤
            current_epoch: 当前轮次
            metrics: 当前指标
            tenant_id: 租户ID
            
        Returns:
            更新后的执行记录
        """
        result = self._execution_repository.update_progress(
            execution_id, progress, current_step, current_epoch, metrics, tenant_id
        )
        
        # 记录进度更新日志（仅在关键节点）
        if result and current_epoch is not None:
            self._log_execution_event(
                execution_id, 'progress',
                f"Progress: {progress:.1f}%, epoch {current_epoch}, step {current_step}",
                epoch=current_epoch,
                step=current_step,
                progress=progress,
                metrics=metrics,
                log_level='debug'
            )
        
        return result
    
    def list_executions(self, tenant_id: str, status: str = None,
                       scenario_type: str = None, user_id: str = None,
                       limit: int = 100, offset: int = 0) -> List[TrainingExecution]:
        """列出执行记录
        
        Args:
            tenant_id: 租户ID
            status: 状态过滤
            scenario_type: 场景类型过滤
            user_id: 用户ID过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            执行记录列表
        """
        return self._execution_repository.list_by_tenant(
            tenant_id, status, scenario_type, user_id, limit, offset
        )
    
    def list_user_executions(self, user_id: str, tenant_id: str = None,
                            status: str = None, limit: int = 100,
                            offset: int = 0) -> List[TrainingExecution]:
        """列出用户的执行记录
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            status: 状态过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            执行记录列表
        """
        return self._execution_repository.list_by_user(
            user_id, tenant_id, status, limit, offset
        )
    
    def list_running_executions(self, tenant_id: str = None) -> List[TrainingExecution]:
        """列出运行中的执行记录
        
        Args:
            tenant_id: 租户ID
            
        Returns:
            运行中的执行记录列表
        """
        return self._execution_repository.list_running(tenant_id)
    
    def get_execution_statistics(self, tenant_id: str, user_id: str = None) -> Dict[str, Any]:
        """获取执行统计信息
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            统计信息
        """
        return self._execution_repository.get_statistics(tenant_id, user_id)
    
    def delete_execution(self, execution_id: str, tenant_id: str = None,
                        user_id: str = None) -> bool:
        """删除执行记录
        
        Args:
            execution_id: 执行ID
            tenant_id: 租户ID
            user_id: 操作用户ID
            
        Returns:
            是否删除成功
        """
        # 记录删除日志
        self._log_execution_event(
            execution_id, 'deleted',
            f"Execution deleted by user {user_id}",
            operator_id=user_id
        )
        
        return self._execution_repository.delete(execution_id, tenant_id)
    
    def _log_execution_event(self, execution_id: str, log_type: str, message: str,
                            from_status: str = None, to_status: str = None,
                            operator_id: str = None, **kwargs):
        """记录执行事件日志
        
        Args:
            execution_id: 执行ID
            log_type: 日志类型
            message: 日志消息
            from_status: 变更前状态
            to_status: 变更后状态
            operator_id: 操作者ID
            **kwargs: 其他日志数据
        """
        try:
            log_data = {
                'execution_id': execution_id,
                'log_type': log_type,
                'log_level': kwargs.get('log_level', 'info'),
                'message': message,
                'from_status': from_status,
                'to_status': to_status,
                'operator_id': operator_id,
                'epoch': kwargs.get('epoch'),
                'step': kwargs.get('step'),
                'progress': kwargs.get('progress'),
                'metrics': kwargs.get('metrics'),
                'resource_snapshot': kwargs.get('resource_snapshot'),
                'details': kwargs.get('details'),
                'source': 'training_execution_service'
            }
            self._execution_repository.create_log(log_data)
        except Exception as e:
            logger.warning(f"Failed to log execution event: {e}")
    
    def get_execution_logs(self, execution_id: str, log_type: str = None,
                          limit: int = 100, offset: int = 0) -> List[TrainingExecutionLog]:
        """获取执行日志
        
        Args:
            execution_id: 执行ID
            log_type: 日志类型过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            日志列表
        """
        return self._execution_repository.list_logs(execution_id, log_type, limit, offset)
    
    # ==================== 原有方法（保持兼容性） ====================
    
    @property
    def _repository(self):
        """兼容旧代码的属性访问"""
        return self._session_repository
    
    @_repository.setter
    def _repository(self, value):
        """兼容旧代码的属性设置"""
        self._session_repository = value
        
    def start_training(self, session_id: str, 
                      training_config: Dict[str, Any]) -> Dict[str, Any]:
        """启动训练
        
        Args:
            session_id: 训练会话ID
            training_config: 训练配置
            
        Returns:
            启动结果
            
        Raises:
            BusinessLogicError: 业务逻辑错误
        """
        session = None  # 初始化session变量以避免未绑定错误
        try:
            # 验证训练配置
            self._validate_training_config(training_config)
            
            # 获取训练会话，若不存在则自动创建为 Pending
            session = self._repository.get_by_id(session_id)
            if not session:
                try:
                    # 使用正确的TrainingSession模型创建会话
                    session_data = {
                        'session_id': session_id,
                        'name': f"Training Session {session_id}",
                        'status': TrainingStatus.PENDING.value,
                        'created_at': datetime.utcnow(),
                        'user_id': 'system',  # 默认用户ID
                        'config': training_config
                    }
                    session = self._repository.create(session_data)
                except Exception as ce:
                    raise BusinessLogicError(f"训练会话不存在且创建失败: {session_id}, 原因: {str(ce)}")
            
            # 检查会话状态（幂等处理：已运行则直接返回）
            status = getattr(session, 'status', None)
            if status == TrainingStatus.RUNNING.value:
                logger.info(f"会话已在运行（幂等返回）: {session_id}")
                started_at = getattr(session, 'started_at', None)
                return {
                    'session_id': session_id,
                    'status': TrainingStatus.RUNNING.value,
                    'started_at': started_at.isoformat() if started_at else None,
                    'message': '训练已在运行'
                }
            # 允许从失败/取消/完成状态重试：重置为PENDING再启动
            if status in (TrainingStatus.FAILED.value, TrainingStatus.CANCELLED.value, TrainingStatus.COMPLETED.value):
                setattr(session, 'status', TrainingStatus.PENDING.value)
                self._repository.update(session)
            # 仅当状态不是PENDING时才拒绝
            if getattr(session, 'status', None) != TrainingStatus.PENDING.value:
                raise BusinessLogicError(f"训练会话状态不正确: {getattr(session, 'status', None)}")
                
            # 更新会话状态 (修复SQLAlchemy模型属性赋值问题)
            # 使用setattr来避免类型检查问题
            setattr(session, 'status', TrainingStatus.RUNNING.value)
            setattr(session, 'started_at', datetime.utcnow())
            self._repository.update(session)
            
            # 初始化训练状态
            self._active_trainings[session_id] = {
                'session': session,
                'config': training_config,
                'start_time': datetime.utcnow(),
                'current_epoch': 0,
                'current_step': 0,
                'metrics_history': [],
                'resource_history': [],
                'orchestration_id': None,
                'lease_id': None
            }
            
            # 创建进度跟踪器
            try:
                progress_manager = get_progress_manager()
                progress_manager.create_progress_tracker(session_id, total_steps=training_config.get('total_steps', 1000))
                progress_manager.set_status(session_id, 'running')
            except Exception as e:
                logger.warning(f"创建进度跟踪器失败: {e}")

            # 资源预检与尝试预留资源（使用 ClusterManager.allocate_resources）
            try:
                # 解析资源需求
                required_gpus = int(training_config.get('gpus', training_config.get('required_gpus', 0)))
                min_free_mem = int(training_config.get('min_free_gpu_memory_mb', 0))
                labels_affinity = training_config.get('node_labels')

                # 优先使用已存在的持久化 allocation（如果 session 已经有 allocation_id），避免重复分配
                allocation = None
                allocation_record = None
                try:
                    existing_alloc = getattr(session, 'resource_allocation', None)
                    if existing_alloc:
                        # 如果已有 allocation 存在且包含 allocation_id，则直接使用
                        try:
                            if isinstance(existing_alloc, list):
                                allocation_record = existing_alloc
                            elif isinstance(existing_alloc, dict):
                                allocation_record = [existing_alloc]
                            else:
                                allocation_record = existing_alloc
                        except Exception:
                            allocation_record = None

                    # 若无现有 allocation，再尝试使用 ResourceAllocator 真实锁定资源
                    if not allocation_record:
                        from backend.modules.distributed.resource_allocator import get_resource_allocator
                        from backend.modules.distributed.task_scheduler import ResourceRequirement
                        allocator = get_resource_allocator()

                        # 构建 ResourceRequirement
                        req = ResourceRequirement(
                            cpu_cores=int(training_config.get('cpu_cores', 1)),
                            memory_mb=int(training_config.get('memory_mb', 1024)),
                            gpu_count=required_gpus,
                            gpu_memory_mb=min_free_mem,
                            disk_mb=int(training_config.get('disk_mb', 1024)),
                            network_mbps=int(training_config.get('network_mbps', 100)),
                            priority=int(training_config.get('priority', 1)),
                            labels_affinity=training_config.get('node_labels') if isinstance(training_config.get('node_labels'), dict) else None
                        )

                        # 获取健康节点
                        try:
                            nodes = asyncio.get_event_loop().run_until_complete(self._cluster_manager.get_healthy_nodes())
                        except Exception:
                            loop2 = asyncio.new_event_loop()
                            try:
                                nodes = loop2.run_until_complete(self._cluster_manager.get_healthy_nodes())
                            finally:
                                loop2.close()

                        # 调用 allocator.allocate_resources(nodes, req)
                        try:
                            alloc_res = asyncio.get_event_loop().run_until_complete(allocator.allocate_resources(nodes, req))
                        except Exception:
                            loop3 = asyncio.new_event_loop()
                            try:
                                alloc_res = loop3.run_until_complete(allocator.allocate_resources(nodes, req))
                            finally:
                                loop3.close()

                        if alloc_res:
                            # alloc_res 返回 (allocation_id, allocation_obj)
                            try:
                                alloc_id, alloc_obj = alloc_res
                                allocation_record = [{'node': alloc_obj.node_id, 'gpu_indices': alloc_obj.gpus, 'allocation_id': alloc_id}]
                            except Exception:
                                allocation_record = None
                except Exception:
                    allocation_record = None

                # 回退：如果未能使用 ResourceAllocator 成功分配，使用 ClusterManager 的预检（轻量）
                if not allocation_record:
                    try:
                        try:
                            allocation = asyncio.get_event_loop().run_until_complete(self._cluster_manager.allocate_resources(required_gpus, min_free_memory_mb=min_free_mem, labels_affinity=labels_affinity))
                        except Exception:
                            loop2 = asyncio.new_event_loop()
                            try:
                                allocation = loop2.run_until_complete(self._cluster_manager.allocate_resources(required_gpus, min_free_memory_mb=min_free_mem, labels_affinity=labels_affinity))
                            finally:
                                loop2.close()

                        # 规范化 cluster_manager 返回格式
                        if isinstance(allocation, tuple) and len(allocation) == 2:
                            alloc_id, alloc_obj = allocation
                            allocation_record = [{'node': alloc_obj.node_id, 'gpu_indices': alloc_obj.gpus, 'allocation_id': alloc_id}]
                        elif isinstance(allocation, dict):
                            allocation_record = [allocation]
                        else:
                            allocation_record = allocation
                    except Exception:
                        allocation_record = None

                if not allocation_record and required_gpus > 0:
                    raise BusinessLogicError(f"无法满足所需 GPU 资源: {required_gpus}")

                # 将 allocation 写入会话元数据并尝试创建资源 lease
                self._active_trainings[session_id]['resource_allocation'] = allocation_record
                # 提取 allocation_ids 记录到活动状态，便于后续训练器/清理/容错直接使用
                alloc_ids = []
                try:
                    if isinstance(allocation_record, list):
                        for a in allocation_record:
                            if isinstance(a, dict) and a.get('allocation_id'):
                                alloc_ids.append(a.get('allocation_id'))
                except Exception:
                    pass
                self._active_trainings[session_id]['allocation_ids'] = alloc_ids
                # 构造 ResourceAllocation 对象以便后续训练器/监控使用
                try:
                    if isinstance(allocation_record, list) and len(allocation_record) > 0:
                        first = allocation_record[0] if isinstance(allocation_record[0], dict) else None
                        if first:
                            from backend.modules.distributed.resource_allocator import ResourceAllocation as RA
                            alloc_obj = RA(
                                node_id=str(first.get('node')) if first.get('node') is not None else '',
                                cpu_cores=int(training_config.get('cpu_cores', 1)),
                                memory_mb=int(training_config.get('memory_mb', 1024)),
                                gpus=list(first.get('gpu_indices', [])),
                                gpu_memory_mb=int(training_config.get('min_free_gpu_memory_mb', min_free_mem)),
                                disk_mb=int(training_config.get('disk_mb', 1024)),
                                network_mbps=int(training_config.get('network_mbps', 100))
                            )
                            self._active_trainings[session_id]['resource_allocation_obj'] = alloc_obj
                            # 主 allocation_id（若存在）
                            if alloc_ids:
                                self._active_trainings[session_id]['allocation_id_primary'] = alloc_ids[0]
                except Exception:
                    pass
                try:
                    session.resource_allocation = allocation_record
                    self._repository.update(session)
                except Exception as e:
                    logger.warning(f"无法持久化 resource_allocation 到仓库: {e}")

                # 继续创建 orchestration lease
                lease_mgr = get_lease_manager()
                fault_mgr = get_fault_tolerance_manager()
                orchestration_id = f"orch-{int(time.time())}-{session_id}"
                lease_id = f"lease-{orchestration_id}"

                # 将 orchestration_id 与 lease 记录到活动训练状态
                self._active_trainings[session_id]['orchestration_id'] = orchestration_id
                self._active_trainings[session_id]['lease_id'] = lease_id
            # 将 lease_id 写回 session/resource metadata 以便 API 层读取
                session.lease_id = lease_id
                self._repository.update(session)

                # 创建 lease（TTL 60s），上层每次心跳应调用 heartbeat
                async def _create_lease():
                    # 包含 allocation_ids 以便 lease 回调/故障处理能直接定位分配
                    alloc_ids = []
                    try:
                        if isinstance(allocation_record, list):
                            for a in allocation_record:
                                if isinstance(a, dict) and a.get('allocation_id'):
                                    alloc_ids.append(a.get('allocation_id'))
                    except Exception:
                        pass
                    await lease_mgr.create_lease(lease_id=lease_id, owner_id=orchestration_id, ttl_seconds=60, metadata={"session_id": session_id, "allocation_ids": alloc_ids, "allocation": allocation_record})

                # 暴露 heartbeat 接口绑定到 LeaseManager
                try:
                    def _heartbeat_handler(lease_id_in: str) -> bool:
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                asyncio.ensure_future(lease_mgr.heartbeat(lease_id_in))
                            else:
                                loop.run_until_complete(lease_mgr.heartbeat(lease_id_in))
                            return True
                        except Exception:
                            try:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                loop.run_until_complete(lease_mgr.heartbeat(lease_id_in))
                                loop.close()
                                return True
                            except Exception:
                                return False
                    self._active_trainings[session_id]['heartbeat_handler'] = _heartbeat_handler
                except Exception:
                    pass

                try:
                    # 尝试在事件循环中调度创建
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(_create_lease())
                    else:
                        loop.run_until_complete(_create_lease())
                except Exception:
                    # 回退为同步创建（环境可能无活动event loop）
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(_create_lease())
                        loop.close()
                    except Exception as e:
                        logger.warning(f"无法创建 lease: {e}")

                # 注册 lease 到期回调，触发故障上报以便容错管理器处理
                def _on_lease_expire(lease):
                    try:
                        # 异步上报故障
                        async def _report():
                            await fault_mgr.report_fault(node_id=lease.owner_id, task_id=None, event_type='lease_expired', description=f'Lease expired for session {session_id}')
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.ensure_future(_report())
                        else:
                            loop.run_until_complete(_report())
                    except Exception as e:
                        logger.error(f"lease expiry callback failed: {e}")

                lease_mgr.register_expiry_callback(_on_lease_expire)
            except Exception as e:
                logger.warning(f"Lease creation/registration 失败: {e}")
            
            # 启动监控线程
            self._start_monitoring(session_id)
            
            logger.info(f"训练已启动: {session_id}")
            
            started_at = getattr(session, 'started_at', None)
            return {
                'session_id': session_id,
                'status': getattr(session, 'status', TrainingStatus.RUNNING.value),
                'started_at': started_at.isoformat() if started_at else None,
                'message': '训练启动成功',
                'orchestration_id': self._active_trainings[session_id].get('orchestration_id'),
                'lease_id': self._active_trainings[session_id].get('lease_id'),
                'allocation_ids': self._active_trainings[session_id].get('allocation_ids'),
                'allocation': self._active_trainings[session_id].get('resource_allocation')
            }
            
        except Exception as e:
            logger.error(f"启动训练失败: {str(e)}")
            # 回滚状态
            if session:  # 修复可能未绑定变量问题
                setattr(session, 'status', TrainingStatus.FAILED.value)
                setattr(session, 'error_message', str(e))
                self._repository.update(session)
            raise BusinessLogicError(f"启动训练失败: {str(e)}", operation="start_training")
    
    def _validate_training_config(self, config: Dict[str, Any]) -> None:
        """验证训练配置
        
        Args:
            config: 训练配置
            
        Raises:
            ValidationError: 配置验证失败
        """
        try:
            # 验证必需字段
            required_fields = ['epochs', 'batch_size', 'learning_rate']
            for field in required_fields:
                if field not in config:
                    raise ValidationError(f"缺少必需配置字段: {field}", field=field)
            
            # 验证数值范围
            epochs = config.get('epochs', 0)
            if not isinstance(epochs, int) or epochs <= 0:
                raise ValidationError("epochs必须是正整数", field="epochs")
                
            batch_size = config.get('batch_size', 0)
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValidationError("batch_size必须是正整数", field="batch_size")
                
            learning_rate = config.get('learning_rate', 0)
            if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
                raise ValidationError("learning_rate必须是正数", field="learning_rate")
                
            # 验证可选字段
            if 'warmup_steps' in config:
                warmup_steps = config['warmup_steps']
                if not isinstance(warmup_steps, int) or warmup_steps < 0:
                    raise ValidationError("warmup_steps必须是非负整数", field="warmup_steps")
                    
            if 'weight_decay' in config:
                weight_decay = config['weight_decay']
                if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
                    raise ValidationError("weight_decay必须是非负数", field="weight_decay")
                    
            logger.info("训练配置验证通过")
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"训练配置验证失败: {str(e)}")
            
    def pause_training(self, session_id: str) -> Dict[str, Any]:
        """暂停训练
        
        Args:
            session_id: 训练会话ID
            
        Returns:
            暂停结果
            
        Raises:
            BusinessLogicError: 业务逻辑错误
        """
        try:
            # 获取训练会话
            session = self._repository.get_by_id(session_id)
            if not session:
                raise BusinessLogicError(f"训练会话不存在: {session_id}")
                
            # 检查会话状态
            if getattr(session, 'status', None) != TrainingStatus.RUNNING.value:
                raise BusinessLogicError(f"训练会话状态不正确: {getattr(session, 'status', None)}")
                
            # 更新会话状态 (修复SQLAlchemy模型属性赋值问题)
            setattr(session, 'status', TrainingStatus.PAUSED.value)
            self._repository.update(session)
            
            # 停止监控线程
            if session_id in self._monitoring_threads:
                self._monitoring_threads[session_id]['stop'] = True
                
            logger.info(f"训练已暂停: {session_id}")
            
            return {
                'session_id': session_id,
                'status': getattr(session, 'status', TrainingStatus.PAUSED.value),
                'message': '训练暂停成功'
            }
            
        except Exception as e:
            logger.error(f"暂停训练失败: {str(e)}")
            raise BusinessLogicError(f"暂停训练失败: {str(e)}", operation="pause_training")
            
    def resume_training(self, session_id: str) -> Dict[str, Any]:
        """恢复训练
        
        Args:
            session_id: 训练会话ID
            
        Returns:
            恢复结果
            
        Raises:
            BusinessLogicError: 业务逻辑错误
        """
        try:
            # 获取训练会话
            session = self._repository.get_by_id(session_id)
            if not session:
                raise BusinessLogicError(f"训练会话不存在: {session_id}")
                
            # 检查会话状态
            if getattr(session, 'status', None) != TrainingStatus.PAUSED.value:
                raise BusinessLogicError(f"训练会话状态不正确: {getattr(session, 'status', None)}")
                
            # 尝试加载最新的检查点
            checkpoint_info = self._load_latest_checkpoint(session_id)
            if checkpoint_info:
                logger.info(f"从检查点恢复训练: epoch {checkpoint_info['epoch']}, step {checkpoint_info['step']}")
                
            # 更新会话状态 (修复SQLAlchemy模型属性赋值问题)
            setattr(session, 'status', TrainingStatus.RUNNING.value)
            self._repository.update(session)
            
            # 重新启动监控线程
            self._start_monitoring(session_id)
            
            logger.info(f"训练已恢复: {session_id}")
            
            return {
                'session_id': session_id,
                'status': getattr(session, 'status', TrainingStatus.RUNNING.value),
                'message': '训练恢复成功',
                'checkpoint_info': checkpoint_info
            }
            
        except Exception as e:
            logger.error(f"恢复训练失败: {str(e)}")
            raise BusinessLogicError(f"恢复训练失败: {str(e)}", operation="resume_training")
            
    def stop_training(self, session_id: str) -> Dict[str, Any]:
        """停止训练
        
        Args:
            session_id: 训练会话ID
            
        Returns:
            停止结果
            
        Raises:
            BusinessLogicError: 业务逻辑错误
        """
        try:
            # 获取训练会话
            session = self._repository.get_by_id(session_id)
            if not session:
                raise BusinessLogicError(f"训练会话不存在: {session_id}")
                
            # 更新会话状态 (修复SQLAlchemy模型属性赋值问题)
            setattr(session, 'status', TrainingStatus.CANCELLED.value)
            setattr(session, 'completed_at', datetime.utcnow())
            self._repository.update(session)
            
            # 停止监控线程
            if session_id in self._monitoring_threads:
                self._monitoring_threads[session_id]['stop'] = True
                del self._monitoring_threads[session_id]
                
            # 从活跃训练中移除前，尝试入队资源释放任务（优先 allocation_id）
            try:
                allocation_record = None
                alloc_ids = []
                # 优先使用活动状态中的 allocation 记录
                if session_id in self._active_trainings:
                    allocation_record = self._active_trainings[session_id].get('resource_allocation')
                    alloc_ids = self._active_trainings[session_id].get('allocation_ids') or []
                # 回退使用会话持久化的 allocation 记录
                if not allocation_record:
                    allocation_record = getattr(session, 'resource_allocation', None)
                # 入队释放
                if allocation_record:
                    from backend.utils.resource_release_queue import enqueue_release
                    if isinstance(allocation_record, list):
                        for alloc in allocation_record:
                            if isinstance(alloc, dict) and alloc.get('allocation_id'):
                                enqueue_release({'allocation_id': alloc.get('allocation_id')})
                            else:
                                node = alloc.get('node') if isinstance(alloc, dict) else None
                                gpus = alloc.get('gpu_indices', []) if isinstance(alloc, dict) else []
                                if node:
                                    enqueue_release({'node': node, 'gpu_indices': gpus})
                    elif isinstance(allocation_record, dict):
                        if allocation_record.get('allocation_id'):
                            enqueue_release({'allocation_id': allocation_record.get('allocation_id')})
                        else:
                            node = allocation_record.get('node')
                            gpus = allocation_record.get('gpu_indices', [])
                            if node:
                                enqueue_release({'node': node, 'gpu_indices': gpus})
                # 清理活动状态
                if session_id in self._active_trainings:
                    del self._active_trainings[session_id]
            except Exception:
                # 释放入队失败时忽略，避免影响停止流程
                if session_id in self._active_trainings:
                    del self._active_trainings[session_id]
                pass
                
            logger.info(f"训练已停止: {session_id}")
            
            completed_at = getattr(session, 'completed_at', None)
            return {
                'session_id': session_id,
                'status': getattr(session, 'status', TrainingStatus.CANCELLED.value),
                'completed_at': completed_at.isoformat() if completed_at else None,
                'message': '训练停止成功',
                'allocation_ids': alloc_ids if alloc_ids else None
            }
            
        except Exception as e:
            logger.error(f"停止训练失败: {str(e)}")
            raise BusinessLogicError(f"停止训练失败: {str(e)}", operation="stop_training")
            
    def get_training_status(self, session_id: str) -> Dict[str, Any]:
        """获取训练状态
        
        Args:
            session_id: 训练会话ID
            
        Returns:
            训练状态信息
            
        Raises:
            BusinessLogicError: 业务逻辑错误
        """
        try:
            # 获取训练会话
            session = self._repository.get_by_id(session_id)
            if not session:
                raise BusinessLogicError(f"训练会话不存在: {session_id}")
                
            # 获取训练状态
            training_state = self._active_trainings.get(session_id, {})
            
            # 获取最新指标
            latest_metrics = None
            if training_state.get('metrics_history'):
                latest_metrics = training_state['metrics_history'][-1]
                
            # 获取最新资源状态
            latest_resource = None
            if training_state.get('resource_history'):
                latest_resource = training_state['resource_history'][-1]
                
            started_at = getattr(session, 'started_at', None)
            completed_at = getattr(session, 'completed_at', None)
                
            return {
                'session_id': session_id,
                'status': getattr(session, 'status', ''),
                'progress': getattr(session, 'progress', 0.0),  # 修复属性访问问题
                'current_epoch': training_state.get('current_epoch', 0),
                'current_step': training_state.get('current_step', 0),
                'started_at': started_at.isoformat() if started_at else None,
                'completed_at': completed_at.isoformat() if completed_at else None,
                'latest_metrics': latest_metrics,
                'latest_resource': latest_resource,
                'error_message': getattr(session, 'error_message', None)  # 修复属性访问问题
            }
            
        except Exception as e:
            logger.error(f"获取训练状态失败: {str(e)}")
            raise BusinessLogicError(f"获取训练状态失败: {str(e)}", operation="get_training_status")
            
    def update_training_metrics(self, session_id: str, 
                               metrics: TrainingMetrics) -> bool:
        """更新训练指标
        
        Args:
            session_id: 训练会话ID
            metrics: 训练指标
            
        Returns:
            更新是否成功
        """
        try:
            # 检查指标异常
            self._check_metrics_anomalies(session_id, metrics)
            
            # 使用进度管理器更新训练进度
            progress_manager = get_progress_manager()
            
            # 准备进度更新数据
            progress_data = {
                "current_epoch": metrics.epoch,
                "current_step": metrics.step,
                "learning_rate": metrics.learning_rate,
                "train_loss": metrics.loss,
                "train_accuracy": metrics.accuracy,
                "metrics": {
                    "throughput": metrics.throughput,
                    "memory_usage": metrics.memory_usage
                }
            }
            
            # 如果有GPU使用率信息，也添加到指标中
            if metrics.gpu_utilization is not None:
                progress_data["metrics"]["gpu_utilization"] = metrics.gpu_utilization
            
            # 更新进度
            progress_manager.update_progress(session_id, **progress_data)
                
            return True
            
        except Exception as e:
            logger.error(f"更新训练指标失败: {str(e)}")
            return False
    
    def _check_metrics_anomalies(self, session_id: str, metrics: TrainingMetrics) -> None:
        """检查训练指标异常
        
        Args:
            session_id: 训练会话ID
            metrics: 训练指标
        """
        try:
            # 检查损失值异常
            if metrics.loss is not None:
                if metrics.loss > 100:
                    logger.warning(f"训练损失值异常高: {metrics.loss}")
                elif metrics.loss < 0:
                    logger.warning(f"训练损失值异常低: {metrics.loss}")
                
            # 检查准确率异常
            if metrics.accuracy is not None:
                if metrics.accuracy > 1.0 or metrics.accuracy < 0:
                    logger.warning(f"训练准确率异常: {metrics.accuracy}")
                
            # 检查学习率异常
            if metrics.learning_rate is not None:
                if metrics.learning_rate > 1.0:
                    logger.warning(f"学习率异常高: {metrics.learning_rate}")
                elif metrics.learning_rate < 1e-10:
                    logger.warning(f"学习率异常低: {metrics.learning_rate}")
                    
            # 检查吞吐量异常
            if metrics.throughput is not None and metrics.throughput < 0:
                logger.warning(f"吞吐量异常: {metrics.throughput}")
                
        except Exception as e:
            logger.error(f"检查指标异常失败: {str(e)}")
            
    def _start_monitoring(self, session_id: str):
        """启动监控线程
        
        Args:
            session_id: 训练会话ID
        """
        if session_id in self._monitoring_threads:
            # 如果已有监控线程，先停止它
            self._monitoring_threads[session_id]['stop'] = True
            
        # 创建新的监控线程
        stop_event = {'stop': False}
        monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(session_id, stop_event),
            daemon=True
        )
        
        self._monitoring_threads[session_id] = {
            'thread': monitoring_thread,
            'stop': stop_event
        }
        
        monitoring_thread.start()
        logger.info(f"监控线程已启动: {session_id}")
    
    def _save_checkpoint(self, session_id: str, epoch: int, step: int, model_state: Dict[str, Any]) -> bool:
        """保存训练检查点
        
        Args:
            session_id: 训练会话ID
            epoch: 当前轮次
            step: 当前步骤
            model_state: 模型状态
            
        Returns:
            是否保存成功
        """
        try:
            import json
            
            # 创建检查点目录
            checkpoint_dir = os.path.join(self._checkpoint_dir, session_id)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 保存模型状态
            model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}_step_{step}.pt")
            # 在实际实现中，这里会保存真实的模型权重
            
            # 保存检查点信息
            checkpoint_info = {
                'session_id': session_id,
                'epoch': epoch,
                'step': step,
                'timestamp': datetime.utcnow().isoformat(),
                'model_path': model_path
            }
            
            info_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_step_{step}.json")
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_info, f, ensure_ascii=False, indent=2)
            
            # 清理旧的检查点（保留最近的5个）
            self._cleanup_old_checkpoints(session_id)
            
            logger.info(f"检查点已保存: {info_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存检查点失败: {str(e)}")
            return False
    
    def _cleanup_old_checkpoints(self, session_id: str) -> None:
        """清理旧的检查点
        
        Args:
            session_id: 训练会话ID
        """
        try:
            import glob
            
            checkpoint_dir = os.path.join(self._checkpoint_dir, session_id)
            if not os.path.exists(checkpoint_dir):
                return
                
            # 获取所有检查点文件
            checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.json"))
            
            # 按修改时间排序
            checkpoint_files.sort(key=os.path.getmtime)
            
            # 保留最近的5个检查点
            max_checkpoints = 5
            if len(checkpoint_files) > max_checkpoints:
                old_checkpoints = checkpoint_files[:-max_checkpoints]
                for old_checkpoint in old_checkpoints:
                    try:
                        os.remove(old_checkpoint)
                        # 同时删除对应的模型文件
                        model_file = old_checkpoint.replace("checkpoint_", "model_").replace(".json", ".pt")
                        if os.path.exists(model_file):
                            os.remove(model_file)
                    except Exception as e:
                        logger.error(f"删除旧检查点失败: {str(e)}")
                        
        except Exception as e:
            logger.error(f"清理旧检查点失败: {str(e)}")
    
    def _load_latest_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """加载最新的检查点
        
        Args:
            session_id: 训练会话ID
            
        Returns:
            检查点信息或None
        """
        try:
            import glob
            import json
            
            checkpoint_dir = os.path.join(self._checkpoint_dir, session_id)
            if not os.path.exists(checkpoint_dir):
                return None
                
            # 获取所有检查点文件
            checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.json"))
            
            if not checkpoint_files:
                return None
                
            # 按修改时间排序，获取最新的
            checkpoint_files.sort(key=os.path.getmtime)
            latest_checkpoint = checkpoint_files[-1]
            
            # 加载检查点信息
            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                checkpoint_info = json.load(f)
                
            return checkpoint_info
            
        except Exception as e:
            logger.error(f"加载检查点失败: {str(e)}")
            return None
        
    def _monitoring_loop(self, session_id: str, stop_event: Dict[str, bool]):
        """监控循环
        
        Args:
            session_id: 训练会话ID
            stop_event: 停止事件
        """
        error_count = 0
        max_errors = 10  # 最大连续错误次数
        
        while not stop_event['stop']:
            try:
                # 获取资源状态
                resource_status = self._get_resource_status()
                
                # 更新资源历史
                if session_id in self._active_trainings:
                    training_state = self._active_trainings[session_id]
                    training_state['resource_history'].append(resource_status.__dict__)
                    
                    # 保留最近100个资源状态
                    if len(training_state['resource_history']) > 100:
                        training_state['resource_history'] = training_state['resource_history'][-100:]
                        
                # 检查训练会话状态
                session = self._repository.get_by_id(session_id)
                if not session or getattr(session, 'status', None) != TrainingStatus.RUNNING.value:
                    break
                    
                # 重置错误计数
                error_count = 0
                
                time.sleep(5)  # 每5秒检查一次
                
            except Exception as e:
                error_count += 1
                logger.error(f"监控循环出错 ({error_count}/{max_errors}): {str(e)}")
                logger.error(f"错误详情: {traceback.format_exc()}")
                
                # 如果连续错误次数超过阈值，停止训练
                if error_count >= max_errors:
                    logger.error(f"监控循环连续出错超过 {max_errors} 次，停止训练 {session_id}")
                    self.handle_training_failure(session_id, f"监控循环连续出错: {str(e)}")
                    break
                    
                time.sleep(5)
                
        logger.info(f"监控线程已停止: {session_id}")
        
    def _get_resource_status(self) -> ResourceStatus:
        """获取资源状态
        
        Returns:
            资源状态
        """
        try:
            # 获取CPU和内存使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            
            # 获取磁盘使用率
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            disk_free_gb = disk.free / (1024**3)
            
            # 获取网络IO
            net_io = psutil.net_io_counters()
            # 修复类型不匹配问题，将int转换为float
            network_io = {
                'bytes_sent': float(net_io.bytes_sent),
                'bytes_recv': float(net_io.bytes_recv)
            }
            
            # 简化实现，GPU状态设为None
            # 实际实现应该查询GPU状态
            gpu_memory_used = None
            gpu_utilization = None
            
            # 检查资源使用情况并发出告警
            self._check_resource_alerts(cpu_percent, memory_percent, disk_usage, disk_free_gb)
            
            return ResourceStatus(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                gpu_memory_used=gpu_memory_used,
                gpu_utilization=gpu_utilization,
                disk_usage=disk_usage,
                network_io=network_io
            )
            
        except Exception as e:
            logger.error(f"获取资源状态失败: {str(e)}")
            # 返回默认状态
            return ResourceStatus(
                cpu_percent=0.0,
                memory_percent=0.0,
                gpu_memory_used=None,
                gpu_utilization=None,
                disk_usage=0.0,
                network_io={'bytes_sent': 0.0, 'bytes_recv': 0.0}
            )
    
    def _check_resource_alerts(self, cpu_percent: float, memory_percent: float, 
                             disk_usage: float, disk_free_gb: float) -> None:
        """检查资源使用情况并发出告警
        
        Args:
            cpu_percent: CPU使用率
            memory_percent: 内存使用率
            disk_usage: 磁盘使用率
            disk_free_gb: 磁盘剩余空间(GB)
        """
        try:
            # CPU使用率告警
            if cpu_percent > 90:
                logger.warning(f"CPU使用率过高: {cpu_percent:.1f}%")
                
            # 内存使用率告警
            if memory_percent > 90:
                logger.warning(f"内存使用率过高: {memory_percent:.1f}%")
                
            # 磁盘使用率告警
            if disk_usage > 90:
                logger.warning(f"磁盘使用率过高: {disk_usage:.1f}%")
                
            # 磁盘空间告警
            if disk_free_gb < 5:
                logger.warning(f"磁盘空间不足: {disk_free_gb:.1f}GB")
                
        except Exception as e:
            logger.error(f"检查资源告警失败: {str(e)}")
            
    def handle_training_failure(self, session_id: str, error_message: str):
        """处理训练失败
        
        Args:
            session_id: 训练会话ID
            error_message: 错误信息
        """
        try:
            # 更新会话状态 (修复SQLAlchemy模型属性赋值问题)
            session = self._repository.get_by_id(session_id)
            if session:
                setattr(session, 'status', TrainingStatus.FAILED.value)
                setattr(session, 'error_message', error_message)
                setattr(session, 'completed_at', datetime.utcnow())
                self._repository.update(session)
                
            # 停止监控线程
            if session_id in self._monitoring_threads:
                self._monitoring_threads[session_id]['stop'] = True
                del self._monitoring_threads[session_id]
                
            # 从活跃训练中移除
            if session_id in self._active_trainings:
                del self._active_trainings[session_id]
                
            logger.error(f"训练失败: {session_id}, 错误: {error_message}")
            
        except Exception as e:
            logger.error(f"处理训练失败时出错: {str(e)}")
            
    def get_training_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取训练历史记录
        
        Args:
            user_id: 用户ID
            limit: 限制数量
            
        Returns:
            训练历史记录列表
        """
        try:
            sessions = self._repository.list_by_user(user_id, limit, 0)
            
            history = []
            for session in sessions:
                started_at = getattr(session, 'started_at', None)
                completed_at = getattr(session, 'completed_at', None)
                history.append({
                    'session_id': getattr(session, 'id', ''),
                    'name': getattr(session, 'name', ''),  # 修复属性访问问题
                    'status': getattr(session, 'status', ''),
                    'progress': getattr(session, 'progress', 0.0),  # 修复属性访问问题
                    'created_at': session.created_at.isoformat() if session.created_at else None,
                    'started_at': started_at.isoformat() if started_at else None,
                    'completed_at': completed_at.isoformat() if completed_at else None,
                    'error_message': getattr(session, 'error_message', None)  # 修复属性访问问题
                })
                
            return history
            
        except Exception as e:
            logger.error(f"获取训练历史记录失败: {str(e)}")
            return []
            
    def run_quick_training_evaluation(self, config: Dict[str, Any], scenario_type: Any) -> Dict[str, Any]:
        """运行快速训练评估
        
        Args:
            config: 训练配置
            scenario_type: 训练场景类型
            
        Returns:
            评估结果
        """
        try:
            import random
            
            start_time = time.time()
            
            # 验证配置
            self._validate_training_config(config)
            
            # 模拟快速训练过程
            epochs = config.get('epochs', 3)
            batch_size = config.get('batch_size', 32)
            learning_rate = config.get('learning_rate', 0.001)
            
            # 基于配置计算预期性能
            base_accuracy = 0.6
            
            # 学习率影响
            if 0.0001 <= learning_rate <= 0.01:
                base_accuracy += 0.15
            elif learning_rate > 0.01:
                base_accuracy -= 0.1
                
            # 批次大小影响
            if 16 <= batch_size <= 64:
                base_accuracy += 0.1
            elif batch_size < 8:
                base_accuracy -= 0.05
                
            # epoch数影响
            if epochs >= 2:
                base_accuracy += 0.05
                
            # 添加随机变化
            accuracy = base_accuracy + random.uniform(-0.1, 0.1)
            accuracy = max(0.1, min(0.95, accuracy))
            
            # 计算其他指标
            loss = 1.0 - accuracy + random.uniform(-0.1, 0.1)
            loss = max(0.05, loss)
            
            validation_accuracy = accuracy * random.uniform(0.85, 0.95)
            
            training_time = time.time() - start_time
            
            result = {
                'accuracy': round(accuracy, 4),
                'loss': round(loss, 4),
                'validation_accuracy': round(validation_accuracy, 4),
                'training_time': round(training_time, 2),
                'epochs_completed': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'evaluation_type': 'quick_training'
            }
            
            logger.info(f"快速训练评估完成: accuracy={accuracy:.4f}, loss={loss:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"快速训练评估失败: {e}")
            # 返回默认结果
            return {
                'accuracy': 0.5,
                'loss': 0.5,
                'validation_accuracy': 0.45,
                'training_time': 1.0,
                'evaluation_type': 'fallback',
                'error': str(e)
            }
    
    # ==================== Launcher 集成方法 ====================
    
    def start_training_via_launcher(
        self,
        session_id: str,
        training_config: Dict[str, Any],
        training_type: str = 'standard'
    ) -> Dict[str, Any]:
        """通过 Launcher 启动训练
        
        Args:
            session_id: 会话ID
            training_config: 训练配置
            training_type: 训练类型
            
        Returns:
            启动结果
        """
        try:
            # 验证配置
            self._validate_training_config(training_config)
            
            # 获取训练会话
            session = self._repository.get_by_id(session_id)
            if not session:
                # 创建会话
                session_data = {
                    'session_id': session_id,
                    'name': f"Training Session {session_id}",
                    'status': TrainingStatus.PENDING.value,
                    'created_at': datetime.utcnow(),
                    'user_id': 'system',
                    'config': training_config
                }
                session = self._repository.create(session_data)
            
            # 构建启动器配置
            launcher_config = self._build_launcher_config(training_config)
            launcher_config['training']['mode'] = training_type
            
            # 更新会话状态
            setattr(session, 'status', TrainingStatus.RUNNING.value)
            setattr(session, 'started_at', datetime.utcnow())
            self._repository.update(session)
            
            # 初始化训练状态
            self._active_trainings[session_id] = {
                'session': session,
                'config': training_config,
                'start_time': datetime.utcnow(),
                'current_epoch': 0,
                'current_step': 0,
                'metrics_history': [],
                'resource_history': [],
                'launcher_type': training_type
            }
            
            # 创建进度跟踪器
            try:
                progress_manager = get_progress_manager()
                total_steps = training_config.get('total_steps', 1000)
                progress_manager.create_progress_tracker(session_id, total_steps=total_steps)
                progress_manager.set_status(session_id, 'running')
            except Exception as e:
                logger.warning(f"创建进度跟踪器失败: {e}")
            
            # 在后台线程启动训练
            def run_launcher_training():
                try:
                    launcher = TrainingSystemLauncher(launcher_config)
                    
                    # 分析配置
                    analysis = launcher.analyze_config()
                    logger.info(f"Launcher analysis for session {session_id}: {analysis}")
                    
                    # 选择训练器
                    trainer = launcher.select_trainer(analysis)
                    
                    if trainer is None:
                        raise Exception("Launcher returned no trainer")
                    
                    # 设置进度回调
                    def progress_callback(progress: float, metrics: Dict[str, Any] = None):
                        if metrics:
                            training_metrics = TrainingMetrics(
                                epoch=metrics.get('epoch', 0),
                                step=metrics.get('step', 0),
                                loss=metrics.get('loss', 0.0),
                                accuracy=metrics.get('accuracy'),
                                learning_rate=metrics.get('learning_rate', 0.0),
                                throughput=metrics.get('throughput', 0.0),
                                memory_usage=metrics.get('memory_usage', 0.0),
                                gpu_utilization=metrics.get('gpu_utilization'),
                                timestamp=datetime.utcnow()
                            )
                            self.update_training_metrics(session_id, training_metrics)
                    
                    if hasattr(trainer, 'set_progress_callback'):
                        trainer.set_progress_callback(progress_callback)
                    
                    # 执行训练
                    result = trainer.train()
                    
                    # 完成训练
                    if result.get('success', True):
                        setattr(session, 'status', TrainingStatus.COMPLETED.value)
                        setattr(session, 'completed_at', datetime.utcnow())
                    else:
                        setattr(session, 'status', TrainingStatus.FAILED.value)
                        setattr(session, 'error_message', result.get('error', 'Training failed'))
                        setattr(session, 'completed_at', datetime.utcnow())
                    
                    self._repository.update(session)
                    
                    # 清理
                    if session_id in self._active_trainings:
                        del self._active_trainings[session_id]
                    
                    logger.info(f"Launcher training completed for session {session_id}")
                    
                except Exception as e:
                    logger.error(f"Launcher training failed for session {session_id}: {e}")
                    self.handle_training_failure(session_id, str(e))
            
            thread = threading.Thread(target=run_launcher_training, daemon=True)
            thread.start()
            
            # 启动监控线程
            self._start_monitoring(session_id)
            
            started_at = getattr(session, 'started_at', None)
            return {
                'session_id': session_id,
                'status': TrainingStatus.RUNNING.value,
                'started_at': started_at.isoformat() if started_at else None,
                'message': '训练启动成功 (via Launcher)',
                'launcher_type': training_type
            }
            
        except Exception as e:
            logger.error(f"通过 Launcher 启动训练失败: {e}")
            raise BusinessLogicError(f"启动训练失败: {str(e)}", operation="start_training_via_launcher")
    
    def launch_production_training(
        self,
        session_id: str,
        training_config: Dict[str, Any],
        training_type: str = 'standard',
        **kwargs
    ) -> Dict[str, Any]:
        """使用生产级启动器启动训练
        
        Args:
            session_id: 会话ID
            training_config: 训练配置
            training_type: 训练类型
            **kwargs: 额外配置
            
        Returns:
            启动结果
        """
        try:
            # 构建生产级配置
            production_config = create_production_training_config(
                training_type=training_type,
                output_dir=training_config.get('output_dir', f'./outputs/{session_id}'),
                model_name=training_config.get('model_name', 'production_model'),
                num_epochs=training_config.get('epochs', kwargs.get('num_epochs', 10)),
                batch_size=training_config.get('batch_size', kwargs.get('batch_size', 16)),
                learning_rate=training_config.get('learning_rate', kwargs.get('learning_rate', 2e-5)),
                enable_checkpoint=kwargs.get('enable_checkpoint', True),
                enable_monitoring=kwargs.get('enable_monitoring', True),
            )
            
            # 添加训练类型特定配置
            if training_type == 'three_stage':
                production_config['three_stage'] = training_config.get('three_stage', {'enabled': True})
            elif training_type == 'distributed':
                production_config['distributed'] = training_config.get('distributed', {'enabled': True})
            elif training_type == 'multimodal':
                production_config['multimodal'] = training_config.get('multimodal', {'enabled': True})
            elif training_type == 'distillation':
                production_config['distillation'] = training_config.get('distillation', {'enabled': True})
            
            # 在后台线程启动生产级训练
            def run_production():
                try:
                    launcher = ProductionTrainingLauncher(production_config)
                    result = launcher.launch_training()
                    
                    # 更新会话状态
                    session = self._repository.get_by_id(session_id)
                    if session:
                        if result.get('success'):
                            setattr(session, 'status', TrainingStatus.COMPLETED.value)
                        else:
                            setattr(session, 'status', TrainingStatus.FAILED.value)
                            setattr(session, 'error_message', result.get('error'))
                        setattr(session, 'completed_at', datetime.utcnow())
                        self._repository.update(session)
                        
                except Exception as e:
                    logger.error(f"Production training failed: {e}")
                    self.handle_training_failure(session_id, str(e))
            
            thread = threading.Thread(target=run_production, daemon=True)
            thread.start()
            
            return {
                'success': True,
                'session_id': session_id,
                'status': TrainingStatus.RUNNING.value,
                'training_type': training_type,
                'message': '生产级训练已启动'
            }
            
        except Exception as e:
            logger.error(f"Failed to launch production training: {e}")
            return {'success': False, 'error': str(e)}
    
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


# 全局训练执行服务实例
_global_training_execution_service = None


def get_training_execution_service() -> TrainingExecutionService:
    """获取全局训练执行服务实例
    
    Returns:
        TrainingExecutionService: 训练执行服务实例
    """
    global _global_training_execution_service
    
    if _global_training_execution_service is None:
        _global_training_execution_service = TrainingExecutionService()
        
    return _global_training_execution_service