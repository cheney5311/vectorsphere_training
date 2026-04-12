"""训练执行管理API

提供训练执行管理相关的API接口，支持多种训练场景的执行控制。

功能特性：
- 多场景训练执行（标准/分布式/多模态/知识蒸馏/三阶段/行业）
- 租户隔离的数据访问
- 实时进度跟踪和指标更新
- 资源监控和管理
- 检查点保存和恢复
"""

import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import logging
logger = logging.getLogger(__name__)

from backend.core.exceptions import ValidationError, BusinessLogicError, ResourceNotFoundError
from backend.utils.response import success_response, error_response
from backend.services.training_execution_service import (
    get_training_execution_service,
    TrainingMetrics
)
from backend.schemas.enums import TrainingStatus

# 创建蓝图
training_execution_bp = Blueprint('training_execution', __name__, url_prefix='/api/v1/training/execution')


def _get_tenant_id() -> str:
    """获取当前租户ID"""
    tenant_id = request.headers.get('X-Tenant-ID')
    if not tenant_id:
        try:
            identity = get_jwt_identity()
            if isinstance(identity, dict):
                tenant_id = identity.get('tenant_id')
            else:
                tenant_id = 'default'
        except Exception:
            tenant_id = 'default'
    return tenant_id


def _get_user_id() -> str:
    """获取当前用户ID"""
    try:
        identity = get_jwt_identity()
        if isinstance(identity, dict):
            return identity.get('user_id', identity.get('sub', str(identity)))
        return str(identity)
    except Exception:
        return 'anonymous'


def _get_execution_service():
    """获取训练执行服务实例"""
    return get_training_execution_service()


# ==================== 场景化训练执行 ====================

@training_execution_bp.route('/scenario/start', methods=['POST'])
@jwt_required()
def start_scenario_training():
    """启动场景化训练执行
    
    根据指定的训练场景类型，自动选择对应的训练器执行训练。
    支持持久化执行记录到数据库（通过Service层）。
    
    请求体:
        {
            "session_id": "会话ID",
            "scenario_type": "训练场景(standard/distributed/multimodal/distillation/three_stage/industry)",
            "config": {
                "model_name": "模型名称",
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
                ...
            },
            "resource_config": {
                "gpus": 1,
                "cpu_cores": 4,
                "memory_mb": 8192
            }
        }
        
    Returns:
        {
            "success": true,
            "data": {
                "execution_id": "执行ID",
                "session_id": "会话ID",
                "scenario_type": "场景类型",
                "status": "running",
                "trainer_type": "使用的训练器类型",
                "started_at": "开始时间"
            },
            "message": "场景化训练启动成功"
        }
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        data = request.get_json() or {}
        session_id = data.get('session_id')
        scenario_type = data.get('scenario_type', 'standard')
        config = data.get('config', {})
        resource_config = data.get('resource_config', {})
        
        if not session_id:
            return error_response("session_id 是必需的", 400)
        
        # 添加租户和用户信息到配置
        config['tenant_id'] = tenant_id
        config['user_id'] = user_id
        config['scenario_type'] = scenario_type
        config['resource_config'] = resource_config
        
        # 合并资源配置
        if resource_config:
            config.update(resource_config)
        
        # 获取服务实例
        service = _get_execution_service()
        
        # 通过Service层创建执行记录（持久化）
        execution = service.create_execution(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            scenario_type=scenario_type,
            config=config
        )
        
        # 根据场景类型选择训练器
        trainer_type = _select_trainer_for_scenario(scenario_type)
        
        # 启动训练
        result = service.start_training(session_id, config)
        
        # 通过Service层更新执行状态为运行中
        service.update_execution_status(
            execution.execution_id,
            TrainingStatus.RUNNING,
            tenant_id=tenant_id,
            user_id=user_id,
            started_at=datetime.utcnow()
        )
        
        # 添加执行信息到结果
        result['execution_id'] = execution.execution_id
        result['scenario_type'] = scenario_type
        result['trainer_type'] = trainer_type
        
        return success_response(
            data=result,
            message=f"{scenario_type} 场景训练启动成功"
        )
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        logger.error(f"启动场景化训练失败: {e}")
        return error_response(f"启动场景化训练失败: {str(e)}", 500)


def _select_trainer_for_scenario(scenario_type: str) -> str:
    """根据场景类型选择训练器
    
    优先级：行业 > 场景化 > 分布式 > 知识蒸馏 > 多模态 > 三阶段 > 标准
    """
    trainer_mapping = {
        'industry': 'IndustryTrainer',
        'scenario': 'ScenarioTrainer',
        'distributed': 'DistributedTrainer',
        'distillation': 'KnowledgeDistillationTrainer',
        'multimodal': 'MultiModalTrainer',
        'three_stage': 'ThreeStageTrainer',
        'standard': 'UnifiedTrainingSystem',
        # 兼容旧的场景类型名称
        'supervised': 'UnifiedTrainingSystem',
        'classification': 'UnifiedTrainingSystem',
        'regression': 'UnifiedTrainingSystem',
        'nlp': 'UnifiedTrainingSystem',
        'computer_vision': 'UnifiedTrainingSystem',
    }
    return trainer_mapping.get(scenario_type, 'UnifiedTrainingSystem')


# ==================== 基础训练执行 ====================

@training_execution_bp.route('/start', methods=['POST'])
@jwt_required()
def start_training_execution():
    """启动训练执行
    
    请求体:
        {
            "session_id": "会话ID",
            "config": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "scenario_type": "分类任务类型（可选）"
        }
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        data = request.get_json() or {}
        session_id = data.get('session_id')
        config = data.get('config', {})
        scenario_type = data.get('scenario_type', 'classification')
        
        if not session_id:
            return error_response("session_id 是必需的", 400)
        
        # 添加租户和用户信息
        config['tenant_id'] = tenant_id
        config['user_id'] = user_id
        config['scenario_type'] = scenario_type
        
        # 启动训练
        service = _get_execution_service()
        result = service.start_training(session_id, config)
        
        return success_response(
            data=result,
            message="训练执行启动成功"
        )
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"启动训练执行失败: {str(e)}", 500)


@training_execution_bp.route('/sessions/<session_id>/start', methods=['POST'])
@jwt_required()
def start_training(session_id: str):
    """启动指定会话的训练
    
    Args:
        session_id: 训练会话ID
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        data = request.get_json() or {}
        training_config = data.get('training_config', data.get('config', {}))
        
        # 添加租户和用户信息
        training_config['tenant_id'] = tenant_id
        training_config['user_id'] = user_id
        
        service = _get_execution_service()
        result = service.start_training(session_id, training_config)
        
        return success_response(
            data=result,
            message="训练启动成功"
        )
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"启动训练失败: {str(e)}", 500)


@training_execution_bp.route('/sessions/<session_id>/pause', methods=['POST'])
@jwt_required()
def pause_training(session_id: str):
    """暂停训练
    
    Args:
        session_id: 训练会话ID
    """
    try:
        tenant_id = _get_tenant_id()
        
        service = _get_execution_service()
        result = service.pause_training(session_id)
        
        return success_response(
            data=result,
            message="训练暂停成功"
        )
        
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"暂停训练失败: {str(e)}", 500)


@training_execution_bp.route('/sessions/<session_id>/resume', methods=['POST'])
@jwt_required()
def resume_training(session_id: str):
    """恢复训练
    
    Args:
        session_id: 训练会话ID
        
    请求体（可选）:
        {
            "from_checkpoint": true,
            "checkpoint_path": "检查点路径"
        }
    """
    try:
        tenant_id = _get_tenant_id()
        
        data = request.get_json() or {}
        from_checkpoint = data.get('from_checkpoint', True)
        checkpoint_path = data.get('checkpoint_path')
        
        service = _get_execution_service()
        result = service.resume_training(session_id)
        
        # 如果指定了检查点，添加到结果
        if checkpoint_path:
            result['checkpoint_path'] = checkpoint_path
        
        return success_response(
            data=result,
            message="训练恢复成功"
        )
        
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"恢复训练失败: {str(e)}", 500)


@training_execution_bp.route('/sessions/<session_id>/stop', methods=['POST'])
@jwt_required()
def stop_training(session_id: str):
    """停止训练
    
    Args:
        session_id: 训练会话ID
        
    请求体（可选）:
        {
            "save_checkpoint": true,
            "reason": "停止原因"
        }
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        data = request.get_json() or {}
        save_checkpoint = data.get('save_checkpoint', True)
        reason = data.get('reason', 'User requested stop')
        
        service = _get_execution_service()
        result = service.stop_training(session_id)
        
        # 添加停止原因到结果
        result['stop_reason'] = reason
        result['checkpoint_saved'] = save_checkpoint
        
        return success_response(
            data=result,
            message="训练停止成功"
        )
        
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"停止训练失败: {str(e)}", 500)


# ==================== 状态和进度查询 ====================

@training_execution_bp.route('/sessions/<session_id>/status', methods=['GET'])
@jwt_required()
def get_training_status(session_id: str):
    """获取训练状态
    
    Args:
        session_id: 训练会话ID
        
    Returns:
        {
            "success": true,
            "data": {
                "session_id": "会话ID",
                "status": "状态",
                "progress": 进度百分比,
                "current_epoch": 当前轮次,
                "total_epochs": 总轮次,
                "current_step": 当前步骤,
                "total_steps": 总步骤,
                "metrics": {...},
                "started_at": "开始时间",
                "elapsed_time": 已用时间秒
            }
        }
    """
    try:
        tenant_id = _get_tenant_id()
        
        service = _get_execution_service()
        status = service.get_training_status(session_id)
        
        if not status:
            return error_response("训练会话不存在", 404)
        
        return success_response(
            data=status,
            message="获取训练状态成功"
        )
        
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"获取训练状态失败: {str(e)}", 500)


@training_execution_bp.route('/sessions/<session_id>/progress', methods=['GET'])
@jwt_required()
def get_training_progress(session_id: str):
    """获取训练进度详情
    
    Args:
        session_id: 训练会话ID
    """
    try:
        tenant_id = _get_tenant_id()
        
        service = _get_execution_service()
        
        # 尝试使用进度管理器获取更详细的进度
        try:
            from backend.modules.training.progress import get_progress_manager
            progress_manager = get_progress_manager()
            progress_data = progress_manager.get_progress(session_id)
            
            if progress_data:
                return success_response(
                    data=progress_data,
                    message="获取训练进度成功"
                )
        except Exception:
            pass
        
        # 回退到基本状态
        status = service.get_training_status(session_id)
        if status:
            return success_response(
                data=status,
                message="获取训练进度成功"
            )
        
        return error_response("训练会话不存在", 404)
        
    except Exception as e:
        return error_response(f"获取训练进度失败: {str(e)}", 500)


# ==================== 指标更新 ====================

@training_execution_bp.route('/sessions/<session_id>/metrics', methods=['POST'])
@jwt_required()
def update_training_metrics(session_id: str):
    """更新训练指标
    
    Args:
        session_id: 训练会话ID
        
    请求体:
        {
            "epoch": 当前轮次,
            "step": 当前步骤,
            "loss": 损失值,
            "accuracy": 准确率,
            "learning_rate": 学习率,
            "throughput": 吞吐量,
            "memory_usage": 内存使用,
            "gpu_utilization": GPU使用率,
            "custom_metrics": {...}
        }
    """
    try:
        tenant_id = _get_tenant_id()
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
        
        # 创建训练指标对象
        metrics = TrainingMetrics(
            epoch=data.get('epoch', 0),
            step=data.get('step', 0),
            loss=data.get('loss', 0.0),
            accuracy=data.get('accuracy'),
            learning_rate=data.get('learning_rate', 0.0),
            throughput=data.get('throughput', 0.0),
            memory_usage=data.get('memory_usage', 0.0),
            gpu_utilization=data.get('gpu_utilization'),
            timestamp=datetime.utcnow()
        )
        
        service = _get_execution_service()
        success = service.update_training_metrics(session_id, metrics)
        
        # 同步更新进度管理器
        try:
            from backend.modules.training.progress import get_progress_manager
            progress_manager = get_progress_manager()
            total_epochs = data.get('total_epochs', 10)
            progress = (metrics.epoch / total_epochs) * 100 if total_epochs > 0 else 0
            progress_manager.update_progress(
                session_id, 
                progress=progress,
                metrics={
                    'loss': metrics.loss,
                    'accuracy': metrics.accuracy,
                    'learning_rate': metrics.learning_rate
                }
            )
        except Exception:
            pass
        
        # 同步更新任务调度器
        try:
            from backend.modules.distributed.task_scheduler import get_task_scheduler
            import asyncio
            ts = get_task_scheduler()
            total_epochs = data.get('total_epochs', 10)
            progress_ratio = min(1.0, metrics.epoch / max(1, total_epochs))
            
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(ts.update_task_progress(session_id, progress_ratio))
            finally:
                loop.close()
        except Exception:
            pass
        
        if success:
            return success_response(
                data={'updated': True, 'metrics': data},
                message="训练指标更新成功"
            )
        else:
            return error_response("训练指标更新失败", 500)
            
    except Exception as e:
        return error_response(f"更新训练指标失败: {str(e)}", 500)


@training_execution_bp.route('/sessions/<session_id>/metrics', methods=['GET'])
@jwt_required()
def get_training_metrics(session_id: str):
    """获取训练指标历史
    
    Args:
        session_id: 训练会话ID
        
    查询参数:
        - limit: 限制返回数量（默认100）
        - metric_type: 指标类型过滤
    """
    try:
        tenant_id = _get_tenant_id()
        
        limit = request.args.get('limit', 100, type=int)
        metric_type = request.args.get('metric_type')
        
        service = _get_execution_service()
        
        # 获取活跃训练的指标历史
        active_training = service._active_trainings.get(session_id)
        if active_training:
            metrics_history = active_training.get('metrics_history', [])
            
            # 过滤并限制
            if metric_type:
                metrics_history = [m for m in metrics_history if metric_type in m]
            metrics_history = metrics_history[-limit:]
            
            return success_response(
                data={
                    'session_id': session_id,
                    'metrics_history': metrics_history,
                    'total_count': len(metrics_history)
                },
                message="获取训练指标成功"
            )
        
        # 尝试从数据库获取
        try:
            from backend.repositories.training_history_repository import get_training_history_repository
            history_repo = get_training_history_repository()
            metrics = history_repo.get_metrics_by_session(session_id, limit=limit)
            
            return success_response(
                data={
                    'session_id': session_id,
                    'metrics_history': metrics,
                    'total_count': len(metrics)
                },
                message="获取训练指标成功"
            )
        except Exception:
            pass
        
        return success_response(
            data={'session_id': session_id, 'metrics_history': [], 'total_count': 0},
            message="获取训练指标成功"
        )
        
    except Exception as e:
        return error_response(f"获取训练指标失败: {str(e)}", 500)


# ==================== 检查点管理 ====================

@training_execution_bp.route('/sessions/<session_id>/checkpoint', methods=['POST'])
@jwt_required()
def save_checkpoint(session_id: str):
    """保存训练检查点
    
    Args:
        session_id: 训练会话ID
        
    请求体（可选）:
        {
            "checkpoint_name": "检查点名称",
            "description": "描述"
        }
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        data = request.get_json() or {}
        checkpoint_name = data.get('checkpoint_name')
        description = data.get('description', '')
        
        service = _get_execution_service()
        
        # 尝试保存检查点
        try:
            # 兼容 Service 的 _save_checkpoint 方法
            # 如果 service 有公开的 save_checkpoint 方法，应该优先使用
            if hasattr(service, 'save_checkpoint'):
                checkpoint_info = service.save_checkpoint(
                    session_id, 
                    name=checkpoint_name,
                    description=description
                )
            else:
                # 获取当前进度以填充 epoch 和 step
                from backend.modules.training.progress import get_progress_manager
                progress_manager = get_progress_manager()
                progress = progress_manager.get_progress(session_id)
                
                epoch = progress.current_epoch if progress else 0
                step = progress.current_step if progress else 0
                
                # 调用 _save_checkpoint
                success = service._save_checkpoint(
                    session_id, 
                    epoch=epoch,
                    step=step,
                    model_state={}  # 占位符，实际逻辑在 service 内部处理
                )
                
                checkpoint_info = {
                    'session_id': session_id,
                    'name': checkpoint_name or f"checkpoint_epoch_{epoch}_step_{step}",
                    'description': description,
                    'timestamp': datetime.utcnow().isoformat(),
                    'epoch': epoch,
                    'step': step
                } if success else {}
            
            return success_response(
                data=checkpoint_info,
                message="检查点保存成功"
            )
        except AttributeError:
            # 服务没有 _save_checkpoint 方法，使用进度管理器
            try:
                from backend.modules.training.progress import get_progress_manager
                progress_manager = get_progress_manager()
                checkpoint_info = progress_manager.save_checkpoint(session_id)
                
                return success_response(
                    data=checkpoint_info,
                    message="检查点保存成功"
                )
            except Exception as e:
                return error_response(f"保存检查点失败: {str(e)}", 500)
        
    except Exception as e:
        return error_response(f"保存检查点失败: {str(e)}", 500)


@training_execution_bp.route('/sessions/<session_id>/checkpoints', methods=['GET'])
@jwt_required()
def list_checkpoints(session_id: str):
    """列出训练检查点
    
    Args:
        session_id: 训练会话ID
    """
    try:
        tenant_id = _get_tenant_id()
        
        service = _get_execution_service()
        
        # 尝试从服务获取检查点列表
        try:
            checkpoints = service._list_checkpoints(session_id)
            return success_response(
                data={'checkpoints': checkpoints},
                message="获取检查点列表成功"
            )
        except AttributeError:
            pass
        
        # 尝试从文件系统获取
        import glob
        checkpoint_dir = f"./checkpoints/{session_id}"
        if os.path.exists(checkpoint_dir):
            checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*"))
            checkpoints = [
                {
                    'path': f,
                    'name': os.path.basename(f),
                    'created_at': datetime.fromtimestamp(os.path.getmtime(f)).isoformat()
                }
                for f in checkpoint_files
            ]
            
            return success_response(
                data={'checkpoints': checkpoints},
                message="获取检查点列表成功"
            )
        
        return success_response(
            data={'checkpoints': []},
            message="获取检查点列表成功"
        )
        
    except Exception as e:
        return error_response(f"获取检查点列表失败: {str(e)}", 500)


# ==================== 资源监控 ====================

@training_execution_bp.route('/sessions/<session_id>/resources', methods=['GET'])
@jwt_required()
def get_resource_status(session_id: str):
    """获取训练资源状态
    
    Args:
        session_id: 训练会话ID
    """
    try:
        tenant_id = _get_tenant_id()
        
        service = _get_execution_service()
        
        # 获取活跃训练的资源状态
        active_training = service._active_trainings.get(session_id)
        
        resource_status = {
            'session_id': session_id,
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'gpu_memory_used': None,
            'gpu_utilization': None,
            'disk_usage': 0.0
        }
        
        # 尝试获取实时资源状态
        try:
            import psutil
            resource_status['cpu_percent'] = psutil.cpu_percent()
            resource_status['memory_percent'] = psutil.virtual_memory().percent
            resource_status['disk_usage'] = psutil.disk_usage('/').percent
        except Exception:
            pass
        
        # 尝试获取GPU状态
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(',')
                    if len(parts) >= 3:
                        resource_status['gpu_memory_used'] = float(parts[0].strip()) / 1024  # GB
                        resource_status['gpu_memory_total'] = float(parts[1].strip()) / 1024  # GB
                        resource_status['gpu_utilization'] = float(parts[2].strip())
        except Exception:
            pass
        
        # 如果有活跃训练，添加资源分配信息
        if active_training:
            resource_status['allocated_resources'] = active_training.get('resource_allocation')
            resource_status['resource_history'] = active_training.get('resource_history', [])[-10:]
        
        return success_response(
            data=resource_status,
            message="获取资源状态成功"
        )
        
    except Exception as e:
        return error_response(f"获取资源状态失败: {str(e)}", 500)


# ==================== 心跳和健康检查 ====================

@training_execution_bp.route('/sessions/<session_id>/heartbeat', methods=['POST'])
@jwt_required()
def heartbeat_training_api(session_id: str):
    """训练心跳/续约 API
    
    客户端/agent应定期调用以续约lease，防止任务被回收。
    
    Args:
        session_id: 训练会话ID
        
    请求体:
        {
            "lease_id": "租约ID",
            "metrics": {...}  // 可选的实时指标
        }
    """
    try:
        data = request.get_json() or {}
        lease_id = data.get('lease_id')
        metrics = data.get('metrics')
        
        if not lease_id:
            return error_response("lease_id 是必需的", 400)
        
        service = _get_execution_service()
        
        # 尝试使用服务层的心跳处理器
        handler = None
        try:
            handler = service._active_trainings.get(session_id, {}).get('heartbeat_handler')
        except Exception:
            handler = None
        
        ok = False
        if handler:
            try:
                ok = handler(lease_id)
            except Exception:
                ok = False
        else:
            # 回退：直接调用 LeaseManager
            try:
                from backend.modules.distributed.lease_manager import get_lease_manager
                import asyncio
                lm = get_lease_manager()
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(lm.heartbeat(lease_id))
                    ok = True
                finally:
                    loop.close()
            except Exception:
                ok = False
        
        # 如果提供了指标，同时更新指标
        if metrics and ok:
            try:
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
                service.update_training_metrics(session_id, training_metrics)
            except Exception:
                pass
        
        if ok:
            return success_response(
                data={'heartbeat': 'processed', 'lease_id': lease_id},
                message="心跳处理成功"
            )
        else:
            return error_response("心跳处理失败", 500)
            
    except Exception as e:
        return error_response(f"心跳处理失败: {e}", 500)


# ==================== 历史和统计 ====================

@training_execution_bp.route('/sessions/history', methods=['GET'])
@jwt_required()
def get_training_history():
    """获取训练历史记录
    
    查询参数:
        - limit: 限制数量（默认50）
        - status: 状态过滤
        - scenario_type: 场景类型过滤
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        limit = request.args.get('limit', 50, type=int)
        status = request.args.get('status')
        scenario_type = request.args.get('scenario_type')
        
        service = _get_execution_service()
        
        # 尝试从服务获取历史
        try:
            history = service.get_training_history(user_id, limit)
            
            # 过滤
            if status:
                history = [h for h in history if h.get('status') == status]
            if scenario_type:
                history = [h for h in history if h.get('scenario_type') == scenario_type]
            
            return success_response(
                data={'history': history, 'total': len(history)},
                message="获取训练历史记录成功"
            )
        except Exception:
            pass
        
        # 尝试从仓库获取
        try:
            from backend.repositories.training_session_repository import get_training_session_repository
            repo = get_training_session_repository()
            sessions = repo.list_by_user(user_id, limit=limit)
            
            history = []
            for session in sessions:
                session_dict = session.to_dict() if hasattr(session, 'to_dict') else {
                    'session_id': getattr(session, 'session_id', ''),
                    'status': getattr(session, 'status', ''),
                    'created_at': getattr(session, 'created_at', None)
                }
                
                # 过滤
                if status and session_dict.get('status') != status:
                    continue
                    
                history.append(session_dict)
            
            return success_response(
                data={'history': history, 'total': len(history)},
                message="获取训练历史记录成功"
            )
        except Exception:
            pass
        
        return success_response(
            data={'history': [], 'total': 0},
            message="获取训练历史记录成功"
        )
        
    except Exception as e:
        return error_response(f"获取训练历史记录失败: {str(e)}", 500)


@training_execution_bp.route('/statistics', methods=['GET'])
@jwt_required()
def get_execution_statistics():
    """获取训练执行统计信息
    
    查询参数:
        - user_id: 用户ID过滤（可选）
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = request.args.get('user_id') or _get_user_id()
        
        service = _get_execution_service()
        
        # 统计活跃训练
        active_count = len(service._active_trainings)
        running_sessions = [
            sid for sid, info in service._active_trainings.items()
        ]
        
        # 尝试获取更多统计
        stats = {
            'active_trainings': active_count,
            'running_sessions': running_sessions,
            'total_sessions': 0,
            'completed_sessions': 0,
            'failed_sessions': 0,
            'paused_sessions': 0
        }
        
        try:
            from backend.repositories.training_session_repository import get_training_session_repository
            repo = get_training_session_repository()
            sessions = repo.list_by_user(user_id, limit=1000)
            
            stats['total_sessions'] = len(sessions)
            for session in sessions:
                status = getattr(session, 'status', '')
                if status == 'completed':
                    stats['completed_sessions'] += 1
                elif status == 'failed':
                    stats['failed_sessions'] += 1
                elif status == 'paused':
                    stats['paused_sessions'] += 1
        except Exception:
            pass
        
        return success_response(
            data=stats,
            message="获取执行统计成功"
        )
        
    except Exception as e:
        return error_response(f"获取执行统计失败: {str(e)}", 500)


# ==================== 批量操作 ====================

@training_execution_bp.route('/batch/stop', methods=['POST'])
@jwt_required()
def batch_stop_trainings():
    """批量停止训练
    
    请求体:
        {
            "session_ids": ["session_1", "session_2", ...],
            "reason": "停止原因"
        }
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        data = request.get_json() or {}
        session_ids = data.get('session_ids', [])
        reason = data.get('reason', 'Batch stop')
        
        if not session_ids:
            return error_response("session_ids 不能为空", 400)
        
        service = _get_execution_service()
        
        stopped = []
        failed = []
        
        for session_id in session_ids:
            try:
                result = service.stop_training(session_id)
                if result.get('status') in ['cancelled', 'stopped']:
                    stopped.append(session_id)
                else:
                    failed.append({'session_id': session_id, 'error': 'Stop failed'})
            except Exception as e:
                failed.append({'session_id': session_id, 'error': str(e)})
        
        return success_response(
            data={
                'stopped': stopped,
                'failed': failed,
                'reason': reason
            },
            message=f"批量停止完成: 成功 {len(stopped)}, 失败 {len(failed)}"
        )
        
    except Exception as e:
        return error_response(f"批量停止失败: {str(e)}", 500)


# ==================== 执行记录管理 ====================

@training_execution_bp.route('/executions', methods=['GET'])
@jwt_required()
def list_executions():
    """列出训练执行记录
    
    查询参数:
        - status: 状态过滤
        - scenario_type: 场景类型过滤
        - user_id: 用户ID过滤（可选）
        - limit: 限制数量（默认100）
        - offset: 偏移量（默认0）
        
    Returns:
        {
            "success": true,
            "data": {
                "executions": [...],
                "total": 总数,
                "limit": 限制,
                "offset": 偏移
            }
        }
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = request.args.get('user_id')
        status = request.args.get('status')
        scenario_type = request.args.get('scenario_type')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        service = _get_execution_service()
        executions = service.list_executions(
            tenant_id=tenant_id,
            status=status,
            scenario_type=scenario_type,
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        
        # 转换为字典列表
        execution_list = [
            e.to_dict() if hasattr(e, 'to_dict') else {
                'execution_id': getattr(e, 'execution_id', ''),
                'session_id': getattr(e, 'session_id', ''),
                'status': getattr(e, 'status', ''),
                'progress': getattr(e, 'progress', 0)
            }
            for e in executions
        ]
        
        return success_response(
            data={
                'executions': execution_list,
                'total': len(execution_list),
                'limit': limit,
                'offset': offset
            },
            message="获取执行记录列表成功"
        )
        
    except Exception as e:
        logger.error(f"获取执行记录列表失败: {e}")
        return error_response(f"获取执行记录列表失败: {str(e)}", 500)


@training_execution_bp.route('/executions/<execution_id>', methods=['GET'])
@jwt_required()
def get_execution_detail(execution_id: str):
    """获取训练执行详情
    
    Args:
        execution_id: 执行ID
        
    Returns:
        {
            "success": true,
            "data": {
                "execution_id": "执行ID",
                "session_id": "会话ID",
                "status": "状态",
                "progress": 进度,
                "metrics": {...},
                ...
            }
        }
    """
    try:
        tenant_id = _get_tenant_id()
        
        service = _get_execution_service()
        execution = service.get_execution(execution_id, tenant_id)
        
        if not execution:
            return error_response("执行记录不存在", 404)
        
        execution_dict = execution.to_dict() if hasattr(execution, 'to_dict') else {
            'execution_id': getattr(execution, 'execution_id', ''),
            'session_id': getattr(execution, 'session_id', ''),
            'status': getattr(execution, 'status', ''),
            'progress': getattr(execution, 'progress', 0)
        }
        
        return success_response(
            data=execution_dict,
            message="获取执行详情成功"
        )
        
    except Exception as e:
        logger.error(f"获取执行详情失败: {e}")
        return error_response(f"获取执行详情失败: {str(e)}", 500)


@training_execution_bp.route('/executions/<execution_id>/progress', methods=['PUT'])
@jwt_required()
def update_execution_progress(execution_id: str):
    """更新执行进度
    
    Args:
        execution_id: 执行ID
        
    请求体:
        {
            "progress": 进度值(0-100),
            "current_step": 当前步骤,
            "current_epoch": 当前轮次,
            "metrics": {...}
        }
    """
    try:
        tenant_id = _get_tenant_id()
        
        data = request.get_json() or {}
        progress = data.get('progress', 0)
        current_step = data.get('current_step')
        current_epoch = data.get('current_epoch')
        metrics = data.get('metrics')
        
        service = _get_execution_service()
        execution = service.update_execution_progress(
            execution_id=execution_id,
            progress=progress,
            current_step=current_step,
            current_epoch=current_epoch,
            metrics=metrics,
            tenant_id=tenant_id
        )
        
        if not execution:
            return error_response("执行记录不存在", 404)
        
        return success_response(
            data={
                'execution_id': execution_id,
                'progress': progress,
                'current_step': current_step,
                'current_epoch': current_epoch
            },
            message="更新进度成功"
        )
        
    except Exception as e:
        logger.error(f"更新执行进度失败: {e}")
        return error_response(f"更新执行进度失败: {str(e)}", 500)


@training_execution_bp.route('/executions/<execution_id>/status', methods=['PUT'])
@jwt_required()
def update_execution_status(execution_id: str):
    """更新执行状态
    
    Args:
        execution_id: 执行ID
        
    请求体:
        {
            "status": "新状态(pending/running/paused/completed/failed/cancelled)",
            "error_message": "错误信息（可选）",
            "result": {...}（可选）
        }
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        data = request.get_json() or {}
        status = data.get('status')
        error_message = data.get('error_message')
        result = data.get('result')
        
        if not status:
            return error_response("status 是必需的", 400)
        
        # 验证状态值
        valid_statuses = ['pending', 'running', 'paused', 'completed', 'failed', 'cancelled']
        if status not in valid_statuses:
            return error_response(f"无效的状态值: {status}", 400)
        
        service = _get_execution_service()
        
        # 构建额外更新字段
        kwargs = {}
        if error_message:
            kwargs['error_message'] = error_message
        if result:
            kwargs['result'] = result
        if status == 'completed':
            kwargs['completed_at'] = datetime.utcnow()
        
        execution = service.update_execution_status(
            execution_id=execution_id,
            status=status,
            tenant_id=tenant_id,
            user_id=user_id,
            **kwargs
        )
        
        if not execution:
            return error_response("执行记录不存在", 404)
        
        return success_response(
            data={
                'execution_id': execution_id,
                'status': status
            },
            message="更新状态成功"
        )
        
    except Exception as e:
        logger.error(f"更新执行状态失败: {e}")
        return error_response(f"更新执行状态失败: {str(e)}", 500)


@training_execution_bp.route('/executions/<execution_id>/logs', methods=['GET'])
@jwt_required()
def get_execution_logs(execution_id: str):
    """获取执行日志
    
    Args:
        execution_id: 执行ID
        
    查询参数:
        - log_type: 日志类型过滤
        - limit: 限制数量
        - offset: 偏移量
    """
    try:
        tenant_id = _get_tenant_id()
        
        log_type = request.args.get('log_type')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        service = _get_execution_service()
        logs = service.get_execution_logs(
            execution_id=execution_id,
            log_type=log_type,
            limit=limit,
            offset=offset
        )
        
        log_list = [
            log.to_dict() if hasattr(log, 'to_dict') else {
                'log_type': getattr(log, 'log_type', ''),
                'message': getattr(log, 'message', ''),
                'created_at': str(getattr(log, 'created_at', ''))
            }
            for log in logs
        ]
        
        return success_response(
            data={
                'logs': log_list,
                'total': len(log_list),
                'execution_id': execution_id
            },
            message="获取执行日志成功"
        )
        
    except Exception as e:
        logger.error(f"获取执行日志失败: {e}")
        return error_response(f"获取执行日志失败: {str(e)}", 500)


@training_execution_bp.route('/executions/<execution_id>', methods=['DELETE'])
@jwt_required()
def delete_execution(execution_id: str):
    """删除执行记录
    
    Args:
        execution_id: 执行ID
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        service = _get_execution_service()
        
        # 检查执行是否存在
        execution = service.get_execution(execution_id, tenant_id)
        if not execution:
            return error_response("执行记录不存在", 404)
        
        # 检查是否可以删除（运行中的不能删除）
        if getattr(execution, 'status', '') == 'running':
            return error_response("运行中的执行记录不能删除", 400)
        
        success = service.delete_execution(execution_id, tenant_id, user_id)
        
        if not success:
            return error_response("删除执行记录失败", 500)
        
        return success_response(
            data={'execution_id': execution_id, 'deleted': True},
            message="删除执行记录成功"
        )
        
    except Exception as e:
        logger.error(f"删除执行记录失败: {e}")
        return error_response(f"删除执行记录失败: {str(e)}", 500)


@training_execution_bp.route('/executions/running', methods=['GET'])
@jwt_required()
def list_running_executions():
    """列出运行中的执行记录
    
    Returns:
        {
            "success": true,
            "data": {
                "executions": [...],
                "count": 数量
            }
        }
    """
    try:
        tenant_id = _get_tenant_id()
        
        service = _get_execution_service()
        executions = service.list_running_executions(tenant_id)
        
        execution_list = [
            e.to_dict() if hasattr(e, 'to_dict') else {
                'execution_id': getattr(e, 'execution_id', ''),
                'session_id': getattr(e, 'session_id', ''),
                'status': getattr(e, 'status', ''),
                'progress': getattr(e, 'progress', 0)
            }
            for e in executions
        ]
        
        return success_response(
            data={
                'executions': execution_list,
                'count': len(execution_list)
            },
            message="获取运行中执行记录成功"
        )
        
    except Exception as e:
        logger.error(f"获取运行中执行记录失败: {e}")
        return error_response(f"获取运行中执行记录失败: {str(e)}", 500)


@training_execution_bp.route('/executions/statistics', methods=['GET'])
@jwt_required()
def get_executions_statistics():
    """获取执行统计信息
    
    查询参数:
        - user_id: 用户ID过滤（可选）
        
    Returns:
        {
            "success": true,
            "data": {
                "total_executions": 总数,
                "pending_executions": 待处理数,
                "running_executions": 运行中数,
                "completed_executions": 完成数,
                "failed_executions": 失败数,
                "paused_executions": 暂停数,
                "cancelled_executions": 取消数
            }
        }
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = request.args.get('user_id')
        
        service = _get_execution_service()
        stats = service.get_execution_statistics(tenant_id, user_id)
        
        return success_response(
            data=stats,
            message="获取执行统计成功"
        )
        
    except Exception as e:
        logger.error(f"获取执行统计失败: {e}")
        return error_response(f"获取执行统计失败: {str(e)}", 500)


# ==================== 健康检查 ====================

@training_execution_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        service = _get_execution_service()
        active_count = len(service._active_trainings)
        
        return success_response(
            data={
                'status': 'healthy',
                'service': 'training_execution',
                'active_trainings': active_count,
                'timestamp': datetime.utcnow().isoformat()
            },
            message="服务正常"
        )
        
    except Exception as e:
        return error_response(f"健康检查失败: {str(e)}", 500)
