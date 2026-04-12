# -*- coding: utf-8 -*-
"""调度器 API 接口

提供生产级的调度任务管理 API，包括任务调度、模板管理、执行监控等功能。
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from flask import Blueprint, request, jsonify, g
from functools import wraps

logger = logging.getLogger(__name__)

# 创建蓝图
scheduler_bp = Blueprint('scheduler', __name__, url_prefix='/api/v1/scheduler')


# ==================== Pydantic 请求模型 ====================

try:
    from pydantic import BaseModel, Field, ValidationError
    try:
        from pydantic import validator
    except ImportError:
        from pydantic import field_validator as validator
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object


if HAS_PYDANTIC:
    class ScheduleTaskRequest(BaseModel):
        """调度任务请求"""
        task_config: Dict[str, Any] = Field(..., description="任务配置")
        schedule_time: str = Field(..., description="调度时间，ISO格式")
        name: Optional[str] = Field(None, description="任务名称")
        priority: str = Field(default="normal", description="优先级: low, normal, high, urgent, critical")
        task_type: str = Field(default="training", description="任务类型")
        task_id: Optional[str] = Field(None, description="任务ID")
        template_id: Optional[str] = Field(None, description="模板ID")
        max_retries: int = Field(default=3, description="最大重试次数")
        timeout_seconds: Optional[int] = Field(None, description="超时秒数")
        depends_on: Optional[List[str]] = Field(None, description="依赖的任务ID列表")
        tags: Optional[List[str]] = Field(None, description="标签")
        metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")

        @validator('schedule_time')
        @classmethod
        def validate_schedule_time(cls, v):
            try:
                dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
                return dt
            except ValueError as e:
                raise ValueError(f'Invalid schedule time format: {e}')

        @validator('priority')
        @classmethod
        def validate_priority(cls, v):
            valid = ['low', 'normal', 'high', 'urgent', 'critical']
            if v not in valid:
                raise ValueError(f'Priority must be one of: {valid}')
            return v

        @validator('task_type')
        @classmethod
        def validate_task_type(cls, v):
            valid = ['training', 'evaluation', 'inference', 'data_processing', 
                    'model_export', 'cleanup', 'backup', 'custom']
            if v not in valid:
                raise ValueError(f'Task type must be one of: {valid}')
            return v


    class ScheduleRecurringTaskRequest(BaseModel):
        """调度周期性任务请求"""
        task_config: Dict[str, Any] = Field(..., description="任务配置")
        cron_expression: Optional[str] = Field(None, description="Cron表达式")
        interval_seconds: Optional[int] = Field(None, description="间隔秒数")
        name: Optional[str] = Field(None, description="任务名称")
        priority: str = Field(default="normal", description="优先级")
        task_type: str = Field(default="training", description="任务类型")


    class UpdateTaskRequest(BaseModel):
        """更新任务请求"""
        name: Optional[str] = None
        description: Optional[str] = None
        priority: Optional[str] = None
        config: Optional[Dict[str, Any]] = None
        schedule_time: Optional[str] = None
        tags: Optional[List[str]] = None
        metadata: Optional[Dict[str, Any]] = None
        timeout_seconds: Optional[int] = None
        max_retries: Optional[int] = None


    class CreateTemplateRequest(BaseModel):
        """创建模板请求"""
        name: str = Field(..., description="模板名称")
        config_template: Dict[str, Any] = Field(..., description="配置模板")
        description: Optional[str] = Field("", description="描述")
        category: Optional[str] = Field(None, description="分类")
        task_type: str = Field(default="training", description="任务类型")
        default_priority: str = Field(default="normal", description="默认优先级")
        default_timeout_seconds: Optional[int] = Field(None, description="默认超时")
        default_max_retries: int = Field(default=3, description="默认最大重试")
        parameters: Optional[Dict[str, Any]] = Field(None, description="参数定义")
        tags: Optional[List[str]] = Field(None, description="标签")


# ==================== 辅助函数 ====================

def _get_scheduler_service():
    """获取调度器服务"""
    try:
        from backend.services.scheduler_service import get_scheduler_service
        return get_scheduler_service()
    except ImportError:
        # 回退到原始调度器
        from backend.services.scheduler import get_scheduler
        return get_scheduler()


def _get_tenant_id() -> Optional[str]:
    """获取租户ID"""
    return getattr(g, 'tenant_id', None) or request.headers.get('X-Tenant-ID')


def _get_user_id() -> Optional[str]:
    """获取用户ID"""
    if hasattr(g, 'current_user') and g.current_user:
        return g.current_user.get('id')
    return request.headers.get('X-User-ID')


def token_required(f):
    """JWT 认证装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        # 简化认证，实际应使用 JWT 验证
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            # 模拟用户信息
            g.current_user = {'id': 'user_001', 'role': 'admin'}
        else:
            g.current_user = None
        return f(*args, **kwargs)
    return decorated


def task_to_response(task: Any) -> Dict[str, Any]:
    """将任务对象转换为响应格式"""
    if isinstance(task, dict):
        # 转换时间字段
        result = task.copy()
        for key in ['schedule_time', 'created_at', 'updated_at', 'started_at', 
                   'completed_at', 'next_run_time', 'last_run_time']:
            if key in result and result[key]:
                if isinstance(result[key], datetime):
                    result[key] = result[key].isoformat()
        return result
    else:
        # 兼容原始 ScheduledTask 对象
        return {
            'id': task.id,
            'config': task.config,
            'schedule_time': task.schedule_time.isoformat() if task.schedule_time else None,
            'status': task.status.value if hasattr(task.status, 'value') else task.status,
            'priority': task.priority.value if hasattr(task.priority, 'value') else task.priority,
            'created_at': task.created_at.isoformat() if task.created_at else None,
            'updated_at': task.updated_at.isoformat() if task.updated_at else None,
            'result': task.result,
            'error_message': task.error_message
        }


# ==================== 调度器控制 API ====================

@scheduler_bp.route('/status', methods=['GET'])
@token_required
def get_scheduler_status():
    """获取调度器状态"""
    try:
        service = _get_scheduler_service()
        
        if hasattr(service, 'get_status'):
            status = service.get_status()
        else:
            status = {
                'running': service.running,
                'scheduled_count': len(service.scheduled_tasks) if hasattr(service, 'scheduled_tasks') else 0
            }
        
        return jsonify({
            'success': True,
            'data': status
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get scheduler status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@scheduler_bp.route('/start', methods=['POST'])
@token_required
def start_scheduler():
    """启动调度器"""
    try:
        service = _get_scheduler_service()
        service.start()
        
        return jsonify({
            'success': True,
            'message': 'Scheduler started'
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@scheduler_bp.route('/stop', methods=['POST'])
@token_required
def stop_scheduler():
    """停止调度器"""
    try:
        service = _get_scheduler_service()
        service.stop()
        
        return jsonify({
            'success': True,
            'message': 'Scheduler stopped'
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to stop scheduler: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== 任务调度 API ====================

@scheduler_bp.route('/tasks', methods=['POST'])
@token_required
def schedule_task():
    """调度任务"""
    try:
        data = request.get_json() or {}
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        # 验证请求
        if HAS_PYDANTIC:
            try:
                req = ScheduleTaskRequest(**data)
                schedule_time = req.schedule_time
                task_config = req.task_config
                name = req.name
                priority = req.priority
                task_type = req.task_type
                task_id = req.task_id
                template_id = req.template_id
                max_retries = req.max_retries
                timeout_seconds = req.timeout_seconds
                depends_on = req.depends_on
                tags = req.tags
                metadata = req.metadata
            except ValidationError as e:
                return jsonify({
                    'success': False,
                    'error': 'Validation error',
                    'details': e.errors()
                }), 400
        else:
            schedule_time_str = data.get('schedule_time')
            if not schedule_time_str:
                return jsonify({'success': False, 'error': 'schedule_time is required'}), 400
            
            try:
                schedule_time = datetime.fromisoformat(schedule_time_str.replace('Z', '+00:00'))
            except ValueError:
                return jsonify({'success': False, 'error': 'Invalid schedule_time format'}), 400
            
            task_config = data.get('task_config', {})
            name = data.get('name')
            priority = data.get('priority', 'normal')
            task_type = data.get('task_type', 'training')
            task_id = data.get('task_id')
            template_id = data.get('template_id')
            max_retries = data.get('max_retries', 3)
            timeout_seconds = data.get('timeout_seconds')
            depends_on = data.get('depends_on')
            tags = data.get('tags')
            metadata = data.get('metadata')
        
        service = _get_scheduler_service()
        
        # 使用新服务
        if hasattr(service, 'schedule_task') and hasattr(service, '_task_repo'):
            success, message, task = service.schedule_task(
                task_config=task_config,
                schedule_time=schedule_time,
                name=name,
                priority=priority,
                task_type=task_type,
                task_id=task_id,
                tenant_id=tenant_id,
                user_id=user_id,
                template_id=template_id,
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
                depends_on=depends_on,
                tags=tags,
                metadata=metadata
            )
            
            if success:
                return jsonify({
                    'success': True,
                    'data': task_to_response(task),
                    'message': message
                }), 201
            else:
                return jsonify({
                    'success': False,
                    'error': message
                }), 400
        else:
            # 回退到原始调度器
            from backend.modules.scheduler.models import TaskPriority
            
            if not service.running:
                service.start()
            
            result_id = service.schedule_task(
                task_config=task_config,
                schedule_time=schedule_time,
                priority=TaskPriority(priority),
                task_id=task_id
            )
            
            task = service.get_task(result_id)
            
            return jsonify({
                'success': True,
                'data': task_to_response(task)
            }), 201
        
    except Exception as e:
        logger.error(f"Failed to schedule task: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@scheduler_bp.route('/tasks/recurring', methods=['POST'])
@token_required
def schedule_recurring_task():
    """调度周期性任务"""
    try:
        data = request.get_json() or {}
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        service = _get_scheduler_service()
        
        if not hasattr(service, 'schedule_recurring_task'):
            return jsonify({
                'success': False,
                'error': 'Recurring tasks not supported'
            }), 400
        
        success, message, task = service.schedule_recurring_task(
            task_config=data.get('task_config', {}),
            cron_expression=data.get('cron_expression'),
            interval_seconds=data.get('interval_seconds'),
            name=data.get('name'),
            priority=data.get('priority', 'normal'),
            task_type=data.get('task_type', 'training'),
            tenant_id=tenant_id,
            user_id=user_id,
            tags=data.get('tags'),
            metadata=data.get('metadata')
        )
        
        if success:
            return jsonify({
                'success': True,
                'data': task_to_response(task),
                'message': message
            }), 201
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 400
        
    except Exception as e:
        logger.error(f"Failed to schedule recurring task: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@scheduler_bp.route('/tasks', methods=['GET'])
@token_required
def list_tasks():
    """列出任务"""
    try:
        tenant_id = _get_tenant_id()
        status = request.args.get('status')
        priority = request.args.get('priority')
        task_type = request.args.get('task_type')
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        
        service = _get_scheduler_service()
        
        if hasattr(service, 'list_tasks'):
            tasks = service.list_tasks(
                tenant_id=tenant_id,
                status=status,
                priority=priority,
                task_type=task_type,
                limit=limit,
                offset=offset
            )
        else:
            # 回退到原始调度器
            from backend.modules.scheduler.models import TaskStatus
            
            if status:
                try:
                    status_enum = TaskStatus(status)
                    tasks = service.get_scheduled_tasks(status=status_enum)
                except ValueError:
                    return jsonify({
                        'success': False,
                        'error': f'Invalid status: {status}'
                    }), 400
            else:
                tasks = service.get_scheduled_tasks()
        
        return jsonify({
            'success': True,
            'data': [task_to_response(t) for t in tasks],
            'count': len(tasks),
            'limit': limit,
            'offset': offset
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@scheduler_bp.route('/tasks/<task_id>', methods=['GET'])
@token_required
def get_task(task_id: str):
    """获取任务详情"""
    try:
        tenant_id = _get_tenant_id()
        service = _get_scheduler_service()
        
        if hasattr(service, 'get_task'):
            task = service.get_task(task_id, tenant_id)
        else:
            task = service.get_task(task_id)
        
        if not task:
            return jsonify({
                'success': False,
                'error': 'Task not found'
            }), 404
        
        return jsonify({
            'success': True,
            'data': task_to_response(task)
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get task {task_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@scheduler_bp.route('/tasks/<task_id>', methods=['PUT'])
@token_required
def update_task(task_id: str):
    """更新任务"""
    try:
        data = request.get_json() or {}
        tenant_id = _get_tenant_id()
        
        service = _get_scheduler_service()
        
        if not hasattr(service, 'update_task'):
            return jsonify({
                'success': False,
                'error': 'Update not supported'
            }), 400
        
        success, message = service.update_task(task_id, data, tenant_id)
        
        if success:
            task = service.get_task(task_id, tenant_id)
            return jsonify({
                'success': True,
                'data': task_to_response(task) if task else None,
                'message': message
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 400
        
    except Exception as e:
        logger.error(f"Failed to update task {task_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@scheduler_bp.route('/tasks/<task_id>', methods=['DELETE'])
@token_required
def delete_task(task_id: str):
    """删除任务"""
    try:
        tenant_id = _get_tenant_id()
        service = _get_scheduler_service()
        
        if hasattr(service, 'delete_task'):
            success, message = service.delete_task(task_id, tenant_id)
        else:
            success = service.cancel_task(task_id)
            message = "Deleted" if success else "Not found"
        
        if success:
            return jsonify({
                'success': True,
                'message': message
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 404
        
    except Exception as e:
        logger.error(f"Failed to delete task {task_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@scheduler_bp.route('/tasks/<task_id>/cancel', methods=['POST'])
@token_required
def cancel_task(task_id: str):
    """取消任务"""
    try:
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        service = _get_scheduler_service()
        
        if hasattr(service, 'cancel_task'):
            if hasattr(service, '_task_repo'):
                success, message = service.cancel_task(task_id, tenant_id, user_id)
            else:
                success = service.cancel_task(task_id)
                message = "Cancelled" if success else "Not found"
        else:
            return jsonify({
                'success': False,
                'error': 'Cancel not supported'
            }), 400
        
        if success:
            return jsonify({
                'success': True,
                'message': message
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 400
        
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@scheduler_bp.route('/tasks/<task_id>/pause', methods=['POST'])
@token_required
def pause_task(task_id: str):
    """暂停任务"""
    try:
        tenant_id = _get_tenant_id()
        service = _get_scheduler_service()
        
        if not hasattr(service, 'pause_task'):
            return jsonify({
                'success': False,
                'error': 'Pause not supported'
            }), 400
        
        success, message = service.pause_task(task_id, tenant_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': message
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 400
        
    except Exception as e:
        logger.error(f"Failed to pause task {task_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@scheduler_bp.route('/tasks/<task_id>/resume', methods=['POST'])
@token_required
def resume_task(task_id: str):
    """恢复任务"""
    try:
        tenant_id = _get_tenant_id()
        service = _get_scheduler_service()
        
        if not hasattr(service, 'resume_task'):
            return jsonify({
                'success': False,
                'error': 'Resume not supported'
            }), 400
        
        success, message = service.resume_task(task_id, tenant_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': message
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 400
        
    except Exception as e:
        logger.error(f"Failed to resume task {task_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@scheduler_bp.route('/tasks/<task_id>/retry', methods=['POST'])
@token_required
def retry_task(task_id: str):
    """重试任务"""
    try:
        tenant_id = _get_tenant_id()
        service = _get_scheduler_service()
        
        if not hasattr(service, 'retry_task'):
            return jsonify({
                'success': False,
                'error': 'Retry not supported'
            }), 400
        
        success, message = service.retry_task(task_id, tenant_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': message
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 400
        
    except Exception as e:
        logger.error(f"Failed to retry task {task_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@scheduler_bp.route('/tasks/<task_id>/logs', methods=['GET'])
@token_required
def get_task_logs(task_id: str):
    """获取任务执行日志"""
    try:
        limit = int(request.args.get('limit', 100))
        service = _get_scheduler_service()
        
        if not hasattr(service, 'get_task_logs'):
            return jsonify({
                'success': False,
                'error': 'Logs not supported'
            }), 400
        
        logs = service.get_task_logs(task_id, limit)
        
        return jsonify({
            'success': True,
            'data': logs,
            'count': len(logs)
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get logs for task {task_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== 模板管理 API ====================

@scheduler_bp.route('/templates', methods=['GET'])
@token_required
def list_templates():
    """列出模板"""
    try:
        tenant_id = _get_tenant_id()
        category = request.args.get('category')
        include_system = request.args.get('include_system', 'true').lower() == 'true'
        
        service = _get_scheduler_service()
        
        if hasattr(service, 'list_templates'):
            templates = service.list_templates(
                tenant_id=tenant_id,
                category=category,
                include_system=include_system
            )
        else:
            # 回退到原始模板管理器
            template_manager = service.get_template_manager()
            template_names = template_manager.list_templates()
            templates = [{
                'id': f'builtin_{name}',
                'name': name,
                'description': template_manager.get_template_description(name) if hasattr(template_manager, 'get_template_description') else name,
                'is_system': True
            } for name in template_names]
        
        return jsonify({
            'success': True,
            'data': templates,
            'count': len(templates)
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@scheduler_bp.route('/templates', methods=['POST'])
@token_required
def create_template():
    """创建模板"""
    try:
        data = request.get_json() or {}
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        service = _get_scheduler_service()
        
        if not hasattr(service, 'create_template'):
            return jsonify({
                'success': False,
                'error': 'Template creation not supported'
            }), 400
        
        success, message, template = service.create_template(
            name=data.get('name'),
            config_template=data.get('config_template', {}),
            description=data.get('description', ''),
            category=data.get('category'),
            task_type=data.get('task_type', 'training'),
            tenant_id=tenant_id,
            user_id=user_id,
            default_priority=data.get('default_priority', 'normal'),
            default_timeout_seconds=data.get('default_timeout_seconds'),
            default_max_retries=data.get('default_max_retries', 3),
            parameters=data.get('parameters'),
            tags=data.get('tags')
        )
        
        if success:
            return jsonify({
                'success': True,
                'data': template,
                'message': message
            }), 201
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 400
        
    except Exception as e:
        logger.error(f"Failed to create template: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@scheduler_bp.route('/templates/<template_id>', methods=['GET'])
@token_required
def get_template(template_id: str):
    """获取模板"""
    try:
        tenant_id = _get_tenant_id()
        service = _get_scheduler_service()
        
        if hasattr(service, 'get_template'):
            template = service.get_template(template_id, tenant_id)
        else:
            # 回退处理
            template_manager = service.get_template_manager()
            name = template_id.replace('builtin_', '')
            try:
                config = template_manager.get_template(name)
                template = {
                    'id': template_id,
                    'name': name,
                    'config_template': config,
                    'is_system': True
                }
            except ValueError:
                template = None
        
        if not template:
            return jsonify({
                'success': False,
                'error': 'Template not found'
            }), 404
        
        return jsonify({
            'success': True,
            'data': template
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get template {template_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@scheduler_bp.route('/templates/<template_id>', methods=['PUT'])
@token_required
def update_template(template_id: str):
    """更新模板"""
    try:
        data = request.get_json() or {}
        tenant_id = _get_tenant_id()
        
        service = _get_scheduler_service()
        
        if not hasattr(service, 'update_template'):
            return jsonify({
                'success': False,
                'error': 'Template update not supported'
            }), 400
        
        success, message = service.update_template(template_id, data, tenant_id)
        
        if success:
            template = service.get_template(template_id, tenant_id)
            return jsonify({
                'success': True,
                'data': template,
                'message': message
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 400
        
    except Exception as e:
        logger.error(f"Failed to update template {template_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@scheduler_bp.route('/templates/<template_id>', methods=['DELETE'])
@token_required
def delete_template(template_id: str):
    """删除模板"""
    try:
        tenant_id = _get_tenant_id()
        service = _get_scheduler_service()
        
        if not hasattr(service, 'delete_template'):
            return jsonify({
                'success': False,
                'error': 'Template deletion not supported'
            }), 400
        
        success, message = service.delete_template(template_id, tenant_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': message
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 400
        
    except Exception as e:
        logger.error(f"Failed to delete template {template_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== 统计 API ====================

@scheduler_bp.route('/statistics', methods=['GET'])
@token_required
def get_statistics():
    """获取统计信息"""
    try:
        tenant_id = _get_tenant_id()
        service = _get_scheduler_service()
        
        if hasattr(service, 'get_statistics'):
            stats = service.get_statistics(tenant_id)
        else:
            # 回退处理
            tasks = service.get_scheduled_tasks()
            stats = {
                'total': len(tasks),
                'by_status': {},
                'by_priority': {}
            }
            for task in tasks:
                status = task.status.value if hasattr(task.status, 'value') else str(task.status)
                priority = task.priority.value if hasattr(task.priority, 'value') else str(task.priority)
                stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
                stats['by_priority'][priority] = stats['by_priority'].get(priority, 0) + 1
        
        return jsonify({
            'success': True,
            'data': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@scheduler_bp.route('/metrics', methods=['GET'])
@token_required
def get_metrics():
    """获取指标"""
    try:
        tenant_id = _get_tenant_id()
        period_type = request.args.get('period_type')
        limit = int(request.args.get('limit', 100))
        
        service = _get_scheduler_service()
        
        if not hasattr(service, 'get_metrics'):
            return jsonify({
                'success': False,
                'error': 'Metrics not supported'
            }), 400
        
        metrics = service.get_metrics(
            tenant_id=tenant_id,
            period_type=period_type,
            limit=limit
        )
        
        return jsonify({
            'success': True,
            'data': metrics,
            'count': len(metrics)
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== 健康检查 ====================

@scheduler_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        service = _get_scheduler_service()
        running = service.running if hasattr(service, 'running') else True
        
        return jsonify({
            'success': True,
            'status': 'healthy' if running else 'stopped',
            'scheduler_running': running,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500
