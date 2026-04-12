"""训练服务

提供训练会话管理的核心业务逻辑。

架构调用关系：
API层 -> Service层 (本模块) -> Launcher层 (training_launcher.py) -> Core层 -> 下游训练模块
"""

import logging
import os
import sys
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 修复导入错误，使用正确的模块路径
from backend.core.exceptions import ValidationError, BusinessLogicError
from backend.schemas.training_models import TrainingSession
from backend.repositories.training_session_repository import get_training_session_repository
from backend.modules.training.progress.progress_manager import get_progress_manager

logger = logging.getLogger(__name__)

# =============================================================================
# 训练启动器集成 (统一通过 launcher 执行训练)
# =============================================================================

from backend.modules.training.launcher import (
    TrainingSystemLauncher,
    ProductionTrainingLauncher,
    launch_training_system,
    get_module_availability,
    diagnose_launcher_module,
    create_scenario_training_config,
    create_production_training_config,
)


class TrainingService:
    """训练服务"""
    
    def __init__(self):
        self._repository = get_training_session_repository()
        
    def create_training_session(
        self, 
        user_id: str, 
        name: str, 
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> TrainingSession:
        """创建训练会话
        
        Args:
            user_id: 用户ID
            name: 会话名称
            description: 会话描述
            config: 训练配置
            
        Returns:
            创建的训练会话对象
            
        Raises:
            ValidationError: 参数验证失败
            BusinessLogicError: 业务逻辑错误
        """
        try:
            # 验证参数
            if not user_id:
                raise ValidationError("用户ID不能为空", field="user_id")
                
            if not name:
                raise ValidationError("会话名称不能为空", field="name")
                
            # 创建训练会话
            session_config = config or {}
            
            # 手动设置时间戳
            current_time = datetime.utcnow()
            
            session = TrainingSession(
                session_id=str(uuid.uuid4()),
                user_id=user_id,
                training_type=session_config.get('training_method', 'standard'),
                config=session_config,
                tenant_id="default",  # 设置默认租户ID
                status="pending",  # 设置初始状态
                created_at=current_time,
                updated_at=current_time
            )
            # 设置名称和描述
            session.config = session.config or {}
            session.config['session_name'] = name
            if description:
                session.config['session_description'] = description
            
            # 保存到仓库
            saved_session = self._repository.create(session)
            
            logger.info(f"创建训练会话成功: {saved_session.session_id}")
            return saved_session
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"创建训练会话失败: {e}")
            raise BusinessLogicError(f"创建训练会话失败: {str(e)}", operation="create_training_session")
            
    def get_training_session(self, session_id: str) -> Optional[TrainingSession]:
        """获取训练会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            训练会话对象，如果不存在则返回None
            
        Raises:
            BusinessLogicError: 业务逻辑错误
        """
        try:
            session = self._repository.get_by_id(session_id)
            if session:
                logger.debug(f"获取训练会话成功: {session_id}")
            else:
                logger.debug(f"训练会话不存在: {session_id}")
            return session
        except Exception as e:
            logger.error(f"获取训练会话失败: {e}")
            raise BusinessLogicError(f"获取训练会话失败: {str(e)}", operation="get_training_session")
            
    def list_training_sessions(
        self, 
        user_id: str, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[TrainingSession]:
        """获取用户训练会话列表
        
        Args:
            user_id: 用户ID
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            训练会话列表
            
        Raises:
            BusinessLogicError: 业务逻辑错误
        """
        try:
            sessions = self._repository.list_by_user(user_id, limit, offset)
            logger.debug(f"获取用户训练会话列表成功: {user_id}, 数量: {len(sessions)}")
            return sessions
        except Exception as e:
            logger.error(f"获取用户训练会话列表失败: {e}")
            raise BusinessLogicError(f"获取用户训练会话列表失败: {str(e)}", operation="list_training_sessions")
            
    def update_training_session_progress(self, session_id: str, progress: float) -> TrainingSession:
        """更新训练会话进度
        
        Args:
            session_id: 会话ID
            progress: 进度值 (0-100)
            
        Returns:
            更新后的训练会话对象
            
        Raises:
            ValidationError: 参数验证失败
            BusinessLogicError: 业务逻辑错误
        """
        try:
            # 验证参数
            if progress < 0 or progress > 100:
                raise ValidationError("进度必须在0-100之间", field="progress")
                
            # 获取会话
            session = self._repository.get_by_id(session_id)
            if not session:
                raise ValidationError("训练会话不存在", field="session_id")
                
            # 更新进度
            session.progress = progress
            session.updated_at = datetime.utcnow()
            
            # 保存更新
            updated_session = self._repository.update(session)
            
            logger.info(f"更新训练会话进度成功: {session_id}, 进度: {progress}")
            return updated_session
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"更新训练会话进度失败: {e}")
            raise BusinessLogicError(f"更新训练会话进度失败: {str(e)}", operation="update_training_session_progress")
            
    def start_training_session(self, session_id: str) -> TrainingSession:
        """开始训练会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            更新后的训练会话对象
            
        Raises:
            ValidationError: 参数验证失败
            BusinessLogicError: 业务逻辑错误
        """
        try:
            # 获取会话
            session = self._repository.get_by_id(session_id)
            if not session:
                raise ValidationError("训练会话不存在", field="session_id")
                
            # 检查状态
            if session.status != "pending":
                raise ValidationError(f"训练会话状态不正确: {session.status}", field="status")
                
            # 开始训练
            session.status = "running"
            session.started_at = datetime.utcnow()
            session.updated_at = datetime.utcnow()
            
            # 创建进度跟踪器
            progress_manager = get_progress_manager()
            config = session.config or {}
            total_epochs = config.get('epochs', 10)
            total_steps = config.get('total_steps', total_epochs * 100)  # 估算步数
            progress_manager.create_progress_tracker(session_id, total_steps, total_epochs)
            
            # 保存更新
            updated_session = self._repository.update(session)
            
            logger.info(f"开始训练会话成功: {session_id}")
            return updated_session
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"开始训练会话失败: {e}")
            raise BusinessLogicError(f"开始训练会话失败: {str(e)}", operation="start_training_session")
            
    def complete_training_session(self, session_id: str, result: Optional[Dict[str, Any]] = None) -> TrainingSession:
        """完成训练会话
        
        Args:
            session_id: 会话ID
            result: 训练结果
            
        Returns:
            更新后的训练会话对象
            
        Raises:
            ValidationError: 参数验证失败
            BusinessLogicError: 业务逻辑错误
        """
        try:
            # 获取会话
            session = self._repository.get_by_id(session_id)
            if not session:
                raise ValidationError("训练会话不存在", field="session_id")
                
            # 检查状态
            if session.status != "running":
                raise ValidationError(f"训练会话状态不正确: {session.status}", field="status")
                
            # 完成训练
            session.status = "completed"
            session.progress = 100.0
            session.result = result
            session.completed_at = datetime.utcnow()
            session.updated_at = datetime.utcnow()
            
            # 保存更新
            updated_session = self._repository.update(session)

            # 释放资源（尝试撤销与该 session 关联的 lease，并清理 resource_allocation）
            try:
                from backend.modules.distributed.lease_manager import get_lease_manager
                import asyncio
                lease_mgr = get_lease_manager()
                async def _cleanup():
                    try:
                        leases = await lease_mgr.list_leases()
                        for lid, lease in leases.items():
                            if isinstance(lease.metadata, dict) and lease.metadata.get('session_id') == session_id:
                                try:
                                    await lease_mgr.revoke(lid)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(_cleanup())
                    else:
                        loop.run_until_complete(_cleanup())
                except Exception:
                    try:
                        loop2 = asyncio.new_event_loop()
                        loop2.run_until_complete(_cleanup())
                        loop2.close()
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"资源释放（lease revoke）遇到问题: {e}")

            # 清理持久化的 allocation 字段
            try:
                if hasattr(updated_session, 'resource_allocation') and updated_session.resource_allocation:
                    # 使用持久化释放队列入队释放任务（更可靠）
                    try:
                        from backend.utils.resource_release_queue import enqueue_release
                        allocation = updated_session.resource_allocation
                        if isinstance(allocation, list):
                            for alloc in allocation:
                                alloc_id = alloc.get('allocation_id')
                                if alloc_id:
                                    enqueue_release({'allocation_id': alloc_id})
                                    continue
                                node = alloc.get('node')
                                gpu_indices = alloc.get('gpu_indices', [])
                                enqueue_release({'node': node, 'gpu_indices': gpu_indices})
                        else:
                            alloc = allocation
                            alloc_id = alloc.get('allocation_id') if isinstance(alloc, dict) else None
                            if alloc_id:
                                enqueue_release({'allocation_id': alloc_id})
                            else:
                                node = alloc.get('node')
                                gpu_indices = alloc.get('gpu_indices', [])
                                enqueue_release({'node': node, 'gpu_indices': gpu_indices})
                    except Exception as e:
                        logger.debug(f"Failed to enqueue resource release: {e}")

                    # 兼容：如果没有持久化队列，可在此尝试直接调用释放
                    try:
                        from backend.modules.distributed.resource_allocator import get_resource_allocator
                        allocator = get_resource_allocator()
                        allocation = updated_session.resource_allocation
                        if isinstance(allocation, list):
                            for alloc in allocation:
                                alloc_id = alloc.get('allocation_id')
                                if alloc_id:
                                    try:
                                        asyncio.get_event_loop().run_until_complete(allocator.release_resources(alloc_id))
                                        continue
                                    except Exception:
                                        pass
                                node = alloc.get('node')
                                gpu_indices = alloc.get('gpu_indices', [])
                                if isinstance(node, str):
                                    allocation_ids = asyncio.get_event_loop().run_until_complete(allocator.find_allocations_by_node_and_gpus(node, gpu_indices))
                                    for aid in allocation_ids:
                                        try:
                                            asyncio.get_event_loop().run_until_complete(allocator.release_resources(aid))
                                        except Exception:
                                            pass
                                else:
                                    node_id = getattr(node, 'node_id', None)
                                    if node_id:
                                        allocation_ids = asyncio.get_event_loop().run_until_complete(allocator.find_allocations_by_node_and_gpus(node_id, gpu_indices))
                                        for aid in allocation_ids:
                                            try:
                                                asyncio.get_event_loop().run_until_complete(allocator.release_resources(aid))
                                            except Exception:
                                                pass
                        else:
                            alloc = allocation
                            alloc_id = alloc.get('allocation_id') if isinstance(alloc, dict) else None
                            if alloc_id:
                                try:
                                    asyncio.get_event_loop().run_until_complete(allocator.release_resources(alloc_id))
                                except Exception:
                                    pass
                            else:
                                node = alloc.get('node')
                                gpu_indices = alloc.get('gpu_indices', [])
                                node_id = node if isinstance(node, str) else getattr(node, 'node_id', None)
                                if node_id:
                                    allocation_ids = asyncio.get_event_loop().run_until_complete(allocator.find_allocations_by_node_and_gpus(node_id, gpu_indices))
                                    for aid in allocation_ids:
                                        try:
                                            asyncio.get_event_loop().run_until_complete(allocator.release_resources(aid))
                                        except Exception:
                                            pass
                    except Exception as e:
                        logger.debug(f"Direct resource release attempt failed: {e}")

                    except Exception as e:
                        logger.debug(f"ResourceAllocator release attempt failed: {e}")

                    updated_session.resource_allocation = None
                    self._repository.update(updated_session)
            except Exception as e:
                logger.warning(f"清理 resource_allocation 失败: {e}")

            logger.info(f"完成训练会话成功: {session_id}")
            return updated_session
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"完成训练会话失败: {e}")
            raise BusinessLogicError(f"完成训练会话失败: {str(e)}", operation="complete_training_session")
            
    def fail_training_session(self, session_id: str, error_message: str) -> TrainingSession:
        """训练会话失败
        
        Args:
            session_id: 会话ID
            error_message: 错误信息
            
        Returns:
            更新后的训练会话对象
            
        Raises:
            ValidationError: 参数验证失败
            BusinessLogicError: 业务逻辑错误
        """
        try:
            # 获取会话
            session = self._repository.get_by_id(session_id)
            if not session:
                raise ValidationError("训练会话不存在", field="session_id")
                
            # 标记失败
            session.status = "failed"
            session.error_message = error_message
            session.completed_at = datetime.utcnow()
            session.updated_at = datetime.utcnow()
            
            # 保存更新
            updated_session = self._repository.update(session)
            
            logger.info(f"训练会话失败: {session_id}, 错误: {error_message}")
            return updated_session
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"标记训练会话失败失败: {e}")
            raise BusinessLogicError(f"标记训练会话失败失败: {str(e)}", operation="fail_training_session")
            
    def delete_training_session(self, session_id: str) -> bool:
        """删除训练会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否删除成功
            
        Raises:
            ValidationError: 参数验证失败
            BusinessLogicError: 业务逻辑错误
        """
        try:
            # 获取会话
            session = self._repository.get_by_id(session_id)
            if not session:
                return False
                
            # 检查状态 - 只能删除已完成、已取消或失败的会话
            if session.status in ["running", "pending"]:
                raise ValidationError("无法删除正在运行或待处理的训练会话，请先取消", field="status")
                
            # 删除训练会话
            success = self._repository.delete(session_id)
            
            if success:
                logger.info(f"删除训练会话成功: {session_id}")
            else:
                logger.warning(f"删除训练会话失败: {session_id}")
                
            return success
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"删除训练会话失败: {e}")
            raise BusinessLogicError(f"删除训练会话失败: {str(e)}", operation="delete_training_session")

    def update_training_session(self, session_id: str, update_data: Dict[str, Any]) -> TrainingSession:
        """更新训练会话
        
        Args:
            session_id: 会话ID
            update_data: 更新数据
            
        Returns:
            更新后的训练会话对象
            
        Raises:
            ValidationError: 参数验证失败
            BusinessLogicError: 业务逻辑错误
        """
        try:
            # 获取会话
            session = self._repository.get_by_id(session_id)
            if not session:
                raise ValidationError("训练会话不存在", field="session_id")
                
            # 更新允许的字段
            allowed_fields = ['status', 'progress', 'config', 'error_message']
            
            for field, value in update_data.items():
                if field in allowed_fields and hasattr(session, field):
                    # 特殊处理config字段
                    if field == 'config':
                        if session.config is None:
                            session.config = {}
                        if isinstance(value, dict):
                            session.config.update(value)
                        else:
                            session.config = value
                    # 特殊处理名称和描述（存储在config中）
                    elif field == 'name':
                        if session.config is None:
                            session.config = {}
                        session.config['session_name'] = value
                    elif field == 'description':
                        if session.config is None:
                            session.config = {}
                        session.config['session_description'] = value
                    else:
                        setattr(session, field, value)
                        
            # 更新时间戳
            session.updated_at = datetime.utcnow()
            
            # 保存更新
            updated_session = self._repository.update(session)
            
            logger.info(f"更新训练会话成功: {session_id}")
            return updated_session
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"更新训练会话失败: {e}")
            raise BusinessLogicError(f"更新训练会话失败: {str(e)}", operation="update_training_session")
            
    def run_quick_evaluation(self, config: Dict[str, Any], scenario_type: Any) -> Dict[str, Any]:
        """运行快速评估
        
        Args:
            config: 训练配置
            scenario_type: 训练场景类型
            
        Returns:
            评估结果
        """
        try:
            # 尝试运行真实的快速评估
            result = self._run_real_quick_evaluation(config, scenario_type)
            if result:
                return result
                
            # 回退到基于配置的启发式评估
            return self._run_heuristic_evaluation(config, scenario_type)
            
        except Exception as e:
            logger.error(f"快速评估失败: {e}")
            # 返回基础评估结果
            return {
                'accuracy': 0.5,
                'loss': 0.5,
                'training_time': 10,
                'validation_accuracy': 0.45,
                'evaluation_type': 'fallback'
            }
            
    def _run_real_quick_evaluation(self, config: Dict[str, Any], scenario_type: Any) -> Optional[Dict[str, Any]]:
        """运行真实的快速评估
        
        Args:
            config: 训练配置
            scenario_type: 训练场景类型
            
        Returns:
            评估结果，如果失败则返回None
        """
        try:
            # 尝试导入训练执行服务
            from .training_execution_service import get_training_execution_service
            execution_service = get_training_execution_service()
            
            # 创建快速评估配置
            eval_config = config.copy()
            eval_config.update({
                'epochs': min(config.get('epochs', 10), 3),  # 最多3个epoch
                'batch_size': max(config.get('batch_size', 32), 16),  # 较大的batch size
                'early_stopping': True,
                'validation_split': 0.2,
                'quick_evaluation': True
            })
            
            # 运行快速训练评估
            result = execution_service.run_quick_training_evaluation(eval_config, scenario_type)
            
            return result
            
        except Exception as e:
            logger.warning(f"真实快速评估失败: {e}")
            return None
            
    def _run_heuristic_evaluation(self, config: Dict[str, Any], scenario_type: Any) -> Dict[str, Any]:
        """运行基于启发式的评估
        
        Args:
            config: 训练配置
            scenario_type: 训练场景类型
            
        Returns:
            评估结果
        """
        import random
        
        base_accuracy = 0.7
        
        # 根据学习率调整
        lr = config.get('learning_rate', 0.001)
        if isinstance(lr, (int, float)):
            if 0.001 <= lr <= 0.01:
                base_accuracy += 0.1
            elif lr < 0.0001 or lr > 0.1:
                base_accuracy -= 0.1
                
        # 根据批次大小调整
        batch_size = config.get('batch_size', 32)
        if isinstance(batch_size, (int, float)):
            if 16 <= batch_size <= 64:
                base_accuracy += 0.05
            elif batch_size < 8 or batch_size > 128:
                base_accuracy -= 0.05
                
        # 根据epoch数调整
        epochs = config.get('epochs', 10)
        if isinstance(epochs, (int, float)):
            if 5 <= epochs <= 20:
                base_accuracy += 0.05
            elif epochs < 3 or epochs > 50:
                base_accuracy -= 0.05
                
        # 添加随机噪声
        noise = random.uniform(-0.05, 0.05)
        final_accuracy = max(0.1, min(0.95, base_accuracy + noise))
        
        return {
            'accuracy': final_accuracy,
            'loss': 1.0 - final_accuracy + random.uniform(-0.1, 0.1),
            'training_time': random.uniform(5, 30),
            'validation_accuracy': final_accuracy * random.uniform(0.85, 0.95),
            'evaluation_type': 'heuristic'
        }
    
    # =========================================================================
    # 生产级训练启动器集成
    # =========================================================================
    
    def launch_production_training(
        self,
        session_id: str,
        training_type: str = 'standard',
        model=None,
        train_loader=None,
        val_loader=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用生产级启动器启动训练
        
        整合 launcher 模块提供的生产级训练能力：
        - 分布式训练管理 (orchestrator/pipeline/progress)
        - 策略组合执行
        - 检查点恢复
        - 资源监控
        - 容错机制
        
        Args:
            session_id: 训练会话ID
            training_type: 训练类型
                - standard: 标准训练
                - three_stage: 三阶段训练
                - industry: 行业模型训练
                - multimodal: 多模态训练
                - distillation: 知识蒸馏
                - distributed: 分布式训练
            model: PyTorch模型（可选）
            train_loader: 训练数据加载器（可选）
            val_loader: 验证数据加载器（可选）
            **kwargs: 额外配置参数
        
        Returns:
            训练结果字典
        
        Raises:
            ValidationError: 参数验证失败
            BusinessLogicError: 业务逻辑错误
        """
        try:
            # 获取训练会话
            session = self._repository.get_by_id(session_id)
            if not session:
                raise ValidationError("训练会话不存在", field="session_id")
            
            # 检查状态
            if session.status == "running":
                raise ValidationError("训练会话已在运行中", field="status")
            
            # 导入生产级启动器
            try:
                from backend.modules.training.launcher import (
                    ProductionTrainingLauncher,
                    create_production_training_config
                )
            except ImportError as e:
                logger.error(f"Failed to import ProductionTrainingLauncher: {e}")
                raise BusinessLogicError(
                    f"生产级训练模块不可用: {e}",
                    operation="launch_production_training"
                )
            
            # 获取会话配置
            session_config = session.config or {}
            
            # 构建生产级训练配置
            production_config = create_production_training_config(
                training_type=training_type,
                output_dir=session_config.get('output_dir', f'./outputs/{session_id}'),
                model_name=session_config.get('model_name', 'production_model'),
                num_epochs=session_config.get('epochs', kwargs.get('num_epochs', 10)),
                batch_size=session_config.get('batch_size', kwargs.get('batch_size', 16)),
                learning_rate=session_config.get('learning_rate', kwargs.get('learning_rate', 2e-5)),
                enable_checkpoint=kwargs.get('enable_checkpoint', True),
                enable_monitoring=kwargs.get('enable_monitoring', True),
                retry_on_failure=kwargs.get('retry_on_failure', 3),
                use_orchestrator=kwargs.get('use_orchestrator', False),
                pipeline_steps=kwargs.get('pipeline_steps'),
                **{k: v for k, v in session_config.items() 
                   if k not in ['output_dir', 'model_name', 'epochs', 'batch_size', 'learning_rate']}
            )
            
            # 添加训练类型特定配置
            if training_type == 'three_stage':
                production_config['three_stage'] = {
                    'enabled': True,
                    'pretrain_epochs': kwargs.get('pretrain_epochs', 3),
                    'finetune_epochs': kwargs.get('finetune_epochs', 5),
                    'preference_epochs': kwargs.get('preference_epochs', 2),
                }
            elif training_type == 'industry':
                production_config['industry'] = {
                    'enabled': True,
                    'type': kwargs.get('industry_type', 'manufacturing'),
                    'include_pretrain': kwargs.get('include_pretrain', True),
                    'include_align': kwargs.get('include_align', True),
                    'include_finetune': kwargs.get('include_finetune', True),
                }
            elif training_type == 'multimodal':
                production_config['multimodal'] = {
                    'enabled': True,
                    'modalities': kwargs.get('modalities', ['text', 'image']),
                }
            elif training_type == 'distillation':
                production_config['distillation'] = {
                    'enabled': True,
                    'scenario': kwargs.get('distillation_scenario', 'standard'),
                    'teacher_model_path': kwargs.get('teacher_model_path', 'mock'),
                    'student_model_path': kwargs.get('student_model_path', 'mock'),
                }
            elif training_type == 'distributed':
                production_config['distributed'] = {
                    'enabled': True,
                    'mode': kwargs.get('distributed_mode', 'ddp'),
                    'world_size': kwargs.get('world_size', 1),
                }
            
            # 更新会话状态为运行中
            session.status = "running"
            session.started_at = datetime.utcnow()
            session.updated_at = datetime.utcnow()
            self._repository.update(session)
            
            # 在后台线程中启动训练
            import threading
            
            def run_production_training():
                try:
                    # 创建生产级启动器
                    launcher = ProductionTrainingLauncher(production_config)
                    
                    # 启动训练
                    result = launcher.launch_training(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader
                    )
                    
                    # 更新会话结果
                    if result.get('success'):
                        self.complete_training_session(session_id, result)
                    else:
                        self.fail_training_session(
                            session_id,
                            result.get('error', 'Training failed')
                        )
                        
                except Exception as e:
                    logger.error(f"Production training failed for session {session_id}: {e}")
                    self.fail_training_session(session_id, str(e))
            
            thread = threading.Thread(target=run_production_training, daemon=True)
            thread.start()
            
            logger.info(f"Production training launched for session {session_id}")
            
            return {
                'success': True,
                'session_id': session_id,
                'training_type': training_type,
                'status': 'running',
                'message': '生产级训练已启动'
            }
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to launch production training: {e}")
            raise BusinessLogicError(
                f"启动生产级训练失败: {str(e)}",
                operation="launch_production_training"
            )
    
    def launch_pipeline_training(
        self,
        session_id: str,
        pipeline_steps: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用流水线启动多阶段训练
        
        Args:
            session_id: 训练会话ID
            pipeline_steps: 流水线步骤列表
            **kwargs: 额外配置参数
        
        Returns:
            训练结果字典
        """
        try:
            # 导入流水线模块
            try:
                from backend.modules.training.pipeline import (
                    create_pipeline, execute_pipeline
                )
            except ImportError as e:
                logger.error(f"Failed to import pipeline module: {e}")
                raise BusinessLogicError(
                    f"流水线模块不可用: {e}",
                    operation="launch_pipeline_training"
                )
            
            # 获取训练会话
            session = self._repository.get_by_id(session_id)
            if not session:
                raise ValidationError("训练会话不存在", field="session_id")
            
            # 创建流水线
            pipeline = create_pipeline(
                name=f'pipeline_{session_id}',
                steps=pipeline_steps,
                session_id=session_id,
                enable_rollback=kwargs.get('enable_rollback', True)
            )
            
            # 更新会话状态
            session.status = "running"
            session.started_at = datetime.utcnow()
            session.updated_at = datetime.utcnow()
            self._repository.update(session)
            
            # 在后台线程中执行流水线
            import threading
            
            def run_pipeline():
                try:
                    result = execute_pipeline(pipeline, session_id)
                    
                    if result.get('success'):
                        self.complete_training_session(session_id, result)
                    else:
                        self.fail_training_session(
                            session_id,
                            result.get('error', 'Pipeline execution failed')
                        )
                        
                except Exception as e:
                    logger.error(f"Pipeline execution failed for session {session_id}: {e}")
                    self.fail_training_session(session_id, str(e))
            
            thread = threading.Thread(target=run_pipeline, daemon=True)
            thread.start()
            
            logger.info(f"Pipeline training launched for session {session_id}")
            
            return {
                'success': True,
                'session_id': session_id,
                'pipeline_name': pipeline.name,
                'steps_count': len(pipeline_steps),
                'status': 'running',
                'message': '流水线训练已启动'
            }
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to launch pipeline training: {e}")
            raise BusinessLogicError(
                f"启动流水线训练失败: {str(e)}",
                operation="launch_pipeline_training"
            )
    
    def get_production_training_progress(self, session_id: str) -> Dict[str, Any]:
        """
        获取生产级训练进度
        
        Args:
            session_id: 训练会话ID
        
        Returns:
            进度信息字典
        """
        try:
            # 获取训练会话
            session = self._repository.get_by_id(session_id)
            if not session:
                raise ValidationError("训练会话不存在", field="session_id")
            
            # 获取进度管理器
            progress_manager = get_progress_manager()
            progress = progress_manager.get_progress(session_id)
            
            if progress:
                return {
                    'session_id': session_id,
                    'status': progress.status,
                    'progress': progress.progress,
                    'current_step': progress.current_step,
                    'total_steps': progress.total_steps,
                    'current_epoch': progress.current_epoch,
                    'total_epochs': progress.total_epochs,
                    'current_stage': progress.current_stage,
                    'stage_progress': progress.stage_progress,
                    'metrics': progress.metrics,
                    'cpu_usage': progress.cpu_usage,
                    'memory_usage': progress.memory_usage,
                    'gpu_usage': progress.gpu_usage,
                    'gpu_memory': progress.gpu_memory,
                    'throughput': progress.throughput,
                    'start_time': progress.start_time.isoformat() if progress.start_time else None,
                    'end_time': progress.end_time.isoformat() if progress.end_time else None,
                }
            
            # 回退到会话信息
            return {
                'session_id': session_id,
                'status': session.status,
                'progress': session.progress or 0.0,
            }
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to get production training progress: {e}")
            raise BusinessLogicError(
                f"获取生产级训练进度失败: {str(e)}",
                operation="get_production_training_progress"
            )
    
    def get_launcher_diagnostics(self) -> Dict[str, Any]:
        """获取启动器诊断信息
        
        Returns:
            诊断信息字典
        """
        try:
            return {
                'available': True,
                'diagnostics': diagnose_launcher_module(),
                'module_availability': get_module_availability()
            }
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
    
    def launch_training_via_launcher(
        self,
        session_id: str,
        training_type: str = 'standard',
        model=None,
        train_loader=None,
        val_loader=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        通过启动器启动训练（统一入口）
        
        整合 launcher 模块提供的训练能力，提供统一的训练启动接口。
        
        Args:
            session_id: 训练会话ID
            training_type: 训练类型
            model: PyTorch模型（可选）
            train_loader: 训练数据加载器（可选）
            val_loader: 验证数据加载器（可选）
            **kwargs: 额外配置参数
        
        Returns:
            训练结果字典
        """
        try:
            # 获取训练会话
            session = self._repository.get_by_id(session_id)
            if not session:
                raise ValidationError("训练会话不存在", field="session_id")
            
            # 获取会话配置
            session_config = session.config or {}
            
            # 使用通用启动器配置构建
            launcher_config = {
                'model': {
                    'name': session_config.get('model_name', 'training_model'),
                    'type': session_config.get('model_type', 'transformer')
                },
                'training': {
                    'mode': training_type,
                    'output_dir': session_config.get('output_dir', f'./outputs/{session_id}'),
                    'num_epochs': session_config.get('epochs', kwargs.get('num_epochs', 10)),
                    'batch_size': session_config.get('batch_size', kwargs.get('batch_size', 16)),
                    'learning_rate': session_config.get('learning_rate', kwargs.get('learning_rate', 2e-5)),
                },
                'data': {
                    'train_path': session_config.get('train_data_path', './data/train')
                }
            }
            
            # 添加训练类型特定配置
            if training_type == 'three_stage':
                launcher_config['three_stage'] = {
                    'enabled': True,
                    **session_config.get('three_stage', {})
                }
            elif training_type == 'distributed':
                launcher_config['distributed'] = {
                    'enabled': True,
                    **session_config.get('distributed', {})
                }
            elif training_type == 'multimodal':
                launcher_config['multimodal'] = {
                    'enabled': True,
                    **session_config.get('multimodal', {})
                }
            elif training_type == 'distillation':
                launcher_config['distillation'] = {
                    'enabled': True,
                    **session_config.get('distillation', {})
                }
            elif training_type == 'industry':
                launcher_config['industry'] = {
                    'enabled': True,
                    **session_config.get('industry', {})
                }
            
            # 更新会话状态
            session.status = "running"
            session.started_at = datetime.utcnow()
            session.updated_at = datetime.utcnow()
            self._repository.update(session)
            
            # 在后台线程启动训练
            import threading
            
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
                    
                    # 执行训练
                    result = trainer.train()
                    
                    # 更新会话结果
                    if result.get('success'):
                        self.complete_training_session(session_id, result)
                    else:
                        self.fail_training_session(session_id, result.get('error', 'Training failed'))
                        
                except Exception as e:
                    logger.error(f"Launcher training failed for session {session_id}: {e}")
                    self.fail_training_session(session_id, str(e))
            
            thread = threading.Thread(target=run_launcher_training, daemon=True)
            thread.start()
            
            logger.info(f"Launcher training started for session {session_id}")
            
            return {
                'success': True,
                'session_id': session_id,
                'training_type': training_type,
                'status': 'running',
                'message': '训练已通过 Launcher 启动'
            }
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to launch training via launcher: {e}")
            raise BusinessLogicError(
                f"通过 Launcher 启动训练失败: {str(e)}",
                operation="launch_training_via_launcher"
            )


# 全局训练服务实例
_global_training_service = TrainingService()


def get_training_service() -> TrainingService:
    """获取全局训练服务实例
    
    Returns:
        TrainingService: 训练服务实例
    """
    return _global_training_service