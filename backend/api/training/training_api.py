"""训练API模块

提供训练相关的REST API接口。
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime

# 修复导入错误，使用绝对导入路径
from backend.core.exceptions import ValidationError, BusinessLogicError
from backend.core.validation import validate_json_schema
from backend.utils.response import success_response, error_response
from backend.services.training_service import get_training_service
from backend.modules.training.progress import update_progress, get_progress
from backend.modules.scheduler.training_scheduler import schedule_training_task, cancel_training_task, get_scheduled_training_tasks
from backend.modules.training.scenarios import get_scenario_manager

# 创建蓝图
training_bp = Blueprint('training', __name__, url_prefix='/api/v1/training')
training_service = get_training_service()


@training_bp.route('/sessions', methods=['POST'])
@jwt_required()
@validate_json_schema('backend/api/schemas/training_create_session.json')
def create_training_session():
    """创建训练会话
    
    Request:
        POST /api/v1/training/sessions
        {
            "name": "训练会话名称",
            "description": "训练会话描述",
            "config": {...}  # 训练配置
        }
        
    Response:
        {
            "success": true,
            "data": {
                "id": "会话ID",
                "user_id": "用户ID",
                "name": "训练会话名称",
                "description": "训练会话描述",
                "status": "pending",
                "progress": 0.0,
                "config": {...},
                "created_at": "创建时间",
                "updated_at": "更新时间"
            },
            "message": "创建成功"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
            
        # 验证必需字段
        user_id = data.get('user_id', current_user_id)
        name = data.get('name', '')
        
        # 数据验证
        if not user_id or user_id == "":
            return error_response("用户ID不能为空", 400)
        if not name or name == "":
            return error_response("训练会话名称不能为空", 400)
            
        # 验证配置参数
        config = data.get('config', {})
        if config:
            epochs = config.get('epochs')
            batch_size = config.get('batch_size')
            
            if epochs is not None and epochs < 0:
                return error_response("训练轮数不能为负数", 400)
            if batch_size is not None and batch_size <= 0:
                return error_response("批次大小必须大于0", 400)
            
        # 创建训练会话
        session = training_service.create_training_session(
            user_id=user_id,
            name=name,
            description=data.get('description'),
            config=config
        )
        
        # 返回包含session_id的格式，兼容测试脚本
        response_data = session.to_dict()
        # 确保包含session_id字段
        response_data['session_id'] = session.session_id
        
        # 直接返回数据，不包装在data字段中，以兼容测试脚本
        return jsonify(response_data), 201
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"内部服务器错误: {str(e)}", 500)


@training_bp.route('/sessions/<session_id>', methods=['GET'])
@jwt_required()
def get_training_session(session_id: str):
    """获取训练会话
    
    Request:
        GET /api/v1/training/sessions/{session_id}
        
    Response:
        {
            "success": true,
            "data": {
                "id": "会话ID",
                "user_id": "用户ID",
                "name": "训练会话名称",
                "description": "训练会话描述",
                "status": "会话状态",
                "progress": 0.0,
                "config": {...},
                "created_at": "创建时间",
                "updated_at": "更新时间",
                "started_at": "开始时间",
                "completed_at": "完成时间",
                "result": {...},
                "error_message": "错误信息"
            },
            "message": "获取成功"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        
        # 获取训练会话
        session = training_service.get_training_session(session_id)
        
        if not session:
            return error_response("训练会话不存在", 404)
            
        # 检查权限
        if session.user_id != current_user_id:
            return error_response("无权限访问该训练会话", 403)
            
        return success_response(session.to_dict(), "获取训练会话成功")
        
    except Exception as e:
        return error_response(f"内部服务器错误: {str(e)}", 500)


@training_bp.route('/sessions', methods=['GET'])
@jwt_required()
def list_training_sessions():
    """获取训练会话列表
    
    Request:
        GET /api/v1/training/sessions
        Query Parameters:
            - limit: 限制数量 (默认: 50)
            - offset: 偏移量 (默认: 0)
            - status: 状态过滤 (可选)
        
    Response:
        {
            "success": true,
            "data": [
                {
                    "id": "会话ID",
                    "user_id": "用户ID",
                    "name": "训练会话名称",
                    "description": "训练会话描述",
                    "status": "会话状态",
                    "progress": 0.0,
                    "config": {...},
                    "created_at": "创建时间",
                    "updated_at": "更新时间"
                }
            ],
            "message": "获取成功"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        
        # 获取查询参数
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        status = request.args.get('status', None)
        
        # 限制参数范围
        limit = min(max(limit, 1), 100)
        offset = max(offset, 0)
        
        # 获取训练会话列表
        sessions = training_service.list_training_sessions(
            user_id=current_user_id,
            limit=limit,
            offset=offset
        )
        
        # 根据状态过滤
        if status:
            sessions = [s for s in sessions if s.status == status]
        
        # 转换为字典列表
        sessions_data = [session.to_dict() for session in sessions]
        
        return success_response(sessions_data, "获取训练会话列表成功")
        
    except Exception as e:
        return error_response(f"内部服务器错误: {str(e)}", 500)


@training_bp.route('/sessions/<session_id>/start', methods=['POST'])
@jwt_required()
def start_training_session(session_id: str):
    """开始训练会话
    
    Request:
        POST /api/v1/training/sessions/{session_id}/start
        
    Response:
        {
            "success": true,
            "data": {
                "id": "会话ID",
                "user_id": "用户ID",
                "name": "训练会话名称",
                "description": "训练会话描述",
                "status": "running",
                "progress": 0.0,
                "config": {...},
                "created_at": "创建时间",
                "updated_at": "更新时间",
                "started_at": "开始时间"
            },
            "message": "训练会话已开始"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        
        # 获取训练会话
        session = training_service.get_training_session(session_id)
        
        if not session:
            return error_response("训练会话不存在", 404)
            
        # 检查权限
        if session.user_id != current_user_id:
            return error_response("无权限操作该训练会话", 403)
            
        # 开始训练会话
        updated_session = training_service.start_training_session(session_id)
        
        return success_response(updated_session.to_dict(), "训练会话已开始")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"内部服务器错误: {str(e)}", 500)


@training_bp.route('/sessions/<session_id>', methods=['PUT'])
@jwt_required()
def update_training_session(session_id: str):
    """更新训练会话
    
    Request:
        PUT /api/v1/training/sessions/{session_id}
        {
            "name": "新的训练会话名称",
            "description": "新的训练会话描述",
            "status": "running",
            "progress": 50.0,
            "config": {...}  # 新的训练配置
        }
        
    Response:
        {
            "success": true,
            "data": {
                "id": "会话ID",
                "user_id": "用户ID",
                "name": "训练会话名称",
                "description": "训练会话描述",
                "status": "会话状态",
                "progress": 50.0,
                "config": {...},
                "created_at": "创建时间",
                "updated_at": "更新时间"
            },
            "message": "更新成功"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        # 获取训练会话
        session = training_service.get_training_session(session_id)
        
        if not session:
            return error_response("训练会话不存在", 404)
            
        # 检查权限
        if session.user_id != current_user_id:
            return error_response("无权限操作该训练会话", 403)
            
        # 更新训练会话
        updated_session = training_service.update_training_session(session_id, data)
        
        return success_response(updated_session.to_dict(), "训练会话更新成功")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"内部服务器错误: {str(e)}", 500)


@training_bp.route('/sessions/<session_id>', methods=['DELETE'])
@jwt_required()
def delete_training_session(session_id: str):
    """删除训练会话
    
    Request:
        DELETE /api/v1/training/sessions/{session_id}
        
    Response:
        {
            "success": true,
            "data": null,
            "message": "训练会话删除成功"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        
        # 获取训练会话
        session = training_service.get_training_session(session_id)
        
        if not session:
            return error_response("训练会话不存在", 404)
            
        # 检查权限
        if session.user_id != current_user_id:
            return error_response("无权限操作该训练会话", 403)
            
        # 检查状态 - 只能删除已完成、已取消或失败的会话
        if session.status in ["running", "pending"]:
            return error_response("无法删除正在运行或待处理的训练会话，请先取消", 400)
            
        # 删除训练会话
        success = training_service.delete_training_session(session_id)
        
        if not success:
            return error_response("删除训练会话失败", 500)
        
        return success_response(None, "训练会话删除成功")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"内部服务器错误: {str(e)}", 500)


@training_bp.route('/sessions/<session_id>/complete', methods=['POST'])
@jwt_required()
def complete_training_session(session_id: str):
    """完成训练会话
    
    Request:
        POST /api/v1/training/sessions/{session_id}/complete
        {
            "result": {...}  # 训练结果
        }
        
    Response:
        {
            "success": true,
            "data": {
                "id": "会话ID",
                "user_id": "用户ID",
                "name": "训练会话名称",
                "description": "训练会话描述",
                "status": "completed",
                "progress": 100.0,
                "config": {...},
                "created_at": "创建时间",
                "updated_at": "更新时间",
                "started_at": "开始时间",
                "completed_at": "完成时间",
                "result": {...}
            },
            "message": "训练会话已完成"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        # 获取训练会话
        session = training_service.get_training_session(session_id)
        
        if not session:
            return error_response("训练会话不存在", 404)
            
        # 检查权限
        if session.user_id != current_user_id:
            return error_response("无权限操作该训练会话", 403)
            
        # 完成训练会话
        result = data.get('result') if data else None
        updated_session = training_service.complete_training_session(session_id, result)
        
        return success_response(updated_session.to_dict(), "训练会话已完成")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"内部服务器错误: {str(e)}", 500)


@training_bp.route('/sessions/<session_id>/fail', methods=['POST'])
@jwt_required()
def fail_training_session(session_id: str):
    """标记训练会话失败
    
    Request:
        POST /api/v1/training/sessions/{session_id}/fail
        {
            "error_message": "错误信息"
        }
        
    Response:
        {
            "success": true,
            "data": {
                "id": "会话ID",
                "user_id": "用户ID",
                "name": "训练会话名称",
                "description": "训练会话描述",
                "status": "failed",
                "progress": 0.0,
                "config": {...},
                "created_at": "创建时间",
                "updated_at": "更新时间",
                "started_at": "开始时间",
                "completed_at": "完成时间",
                "error_message": "错误信息"
            },
            "message": "训练会话已标记为失败"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        # 获取训练会话
        session = training_service.get_training_session(session_id)
        
        if not session:
            return error_response("训练会话不存在", 404)
            
        # 检查权限
        if session.user_id != current_user_id:
            return error_response("无权限操作该训练会话", 403)
            
        # 标记训练会话失败
        error_message = data.get('error_message', '未知错误')
        updated_session = training_service.fail_training_session(session_id, error_message)
        
        return success_response(updated_session.to_dict(), "训练会话已标记为失败")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"内部服务器错误: {str(e)}", 500)


@training_bp.route('/sessions/<session_id>/progress', methods=['PUT'])
@jwt_required()
def update_training_progress_api(session_id: str):
    """更新训练进度
    
    Request:
        PUT /api/v1/training/sessions/{session_id}/progress
        {
            "progress": 50.0,
            "current_step": 100,
            "total_steps": 200,
            "train_loss": 0.5,
            "eval_loss": 0.6,
            "learning_rate": 1e-5
        }
        
    Response:
        {
            "success": true,
            "data": {
                "id": "会话ID",
                "user_id": "用户ID",
                "name": "训练会话名称",
                "description": "训练会话描述",
                "status": "running",
                "progress": 50.0,
                "config": {...},
                "created_at": "创建时间",
                "updated_at": "更新时间",
                "started_at": "开始时间"
            },
            "message": "进度更新成功"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        # 获取训练会话
        session = training_service.get_training_session(session_id)
        
        if not session:
            return error_response("训练会话不存在", 404)
            
        # 检查权限
        if session.user_id != current_user_id:
            return error_response("无权限操作该训练会话", 403)
            
        # 验证参数
        progress = data.get('progress')
        if progress is not None and (progress < 0 or progress > 100):
            return error_response("进度必须在0-100之间", 400)
        
        # 更新进度
        update_params = {
            'progress': progress,
            'current_step': data.get('current_step'),
            'total_steps': data.get('total_steps'),
            'train_loss': data.get('train_loss'),
            'eval_loss': data.get('eval_loss'),
            'learning_rate': data.get('learning_rate')
        }
        
        # 移除None值
        update_params = {k: v for k, v in update_params.items() if v is not None}
        
        # 更新进度
        updated_session = training_service.update_training_session_progress(session_id, progress)
        
        # 同时更新详细进度信息
        if update_params:
            update_progress(session_id, **update_params)
        
        return success_response(updated_session.to_dict(), "进度更新成功")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"内部服务器错误: {str(e)}", 500)


@training_bp.route('/sessions/<session_id>/progress', methods=['GET'])
@jwt_required()
def get_training_progress_api(session_id: str):
    """获取训练进度详情
    
    Request:
        GET /api/v1/training/sessions/{session_id}/progress
        
    Response:
        {
            "success": true,
            "data": {
                "session_id": "会话ID",
                "total_steps": 1000,
                "current_step": 500,
                "current_epoch": 2,
                "total_epochs": 10,
                "progress": 50.0,
                "learning_rate": 1e-5,
                "train_loss": 0.5,
                "eval_loss": 0.6,
                "train_accuracy": 0.85,
                "eval_accuracy": 0.83,
                "status": "running",
                "start_time": "开始时间",
                "metrics": {...}
            },
            "message": "获取进度成功"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        
        # 获取训练会话
        session = training_service.get_training_session(session_id)
        
        if not session:
            return error_response("训练会话不存在", 404)
            
        # 检查权限
        if session.user_id != current_user_id:
            return error_response("无权限访问该训练会话", 403)
            
        # 获取详细进度信息
        progress = get_progress(session_id)
        
        if progress:
            progress_data = {
                'session_id': progress.session_id,
                'total_steps': progress.total_steps,
                'current_step': progress.current_step,
                'current_epoch': progress.current_epoch,
                'total_epochs': progress.total_epochs,
                'progress': progress.progress,
                'learning_rate': progress.learning_rate,
                'train_loss': progress.train_loss,
                'eval_loss': progress.eval_loss,
                'train_accuracy': progress.train_accuracy,
                'eval_accuracy': progress.eval_accuracy,
                'status': progress.status,
                'start_time': progress.start_time.isoformat() if progress.start_time else None,
                'end_time': progress.end_time.isoformat() if progress.end_time else None,
                'error_message': progress.error_message,
                'metrics': progress.metrics
            }
        else:
            # 如果没有详细进度，返回会话的基本信息
            progress_data = {
                'session_id': session.id,
                'progress': session.progress,
                'status': session.status,
                'created_at': session.created_at.isoformat() if session.created_at else None,
                'updated_at': session.updated_at.isoformat() if session.updated_at else None,
                'started_at': session.started_at.isoformat() if session.started_at else None,
                'completed_at': session.completed_at.isoformat() if session.completed_at else None
            }
        
        return success_response(progress_data, "获取进度成功")
        
    except Exception as e:
        return error_response(f"内部服务器错误: {str(e)}", 500)


@training_bp.route('/sessions/<session_id>/cancel', methods=['POST'])
@jwt_required()
def cancel_training_session(session_id: str):
    """取消训练会话
    
    Request:
        POST /api/v1/training/sessions/{session_id}/cancel
        
    Response:
        {
            "success": true,
            "data": {
                "id": "会话ID",
                "user_id": "用户ID",
                "name": "训练会话名称",
                "description": "训练会话描述",
                "status": "cancelled",
                "progress": 0.0,
                "config": {...},
                "created_at": "创建时间",
                "updated_at": "更新时间",
                "started_at": "开始时间",
                "completed_at": "完成时间"
            },
            "message": "训练会话已取消"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        
        # 获取训练会话
        session = training_service.get_training_session(session_id)
        
        if not session:
            return error_response("训练会话不存在", 404)
            
        # 检查权限
        if session.user_id != current_user_id:
            return error_response("无权限操作该训练会话", 403)
            
        # 检查状态
        if session.status not in ["pending", "running"]:
            return error_response("只能取消待处理或运行中的训练会话", 400)
            
        # 更新会话状态为取消
        session.status = "cancelled"
        session.completed_at = datetime.utcnow()
        session.updated_at = datetime.utcnow()
        updated_session = training_service._repository.update(session)
        
        # 如果有进度跟踪器，也更新其状态
        try:
            from backend.modules.training.progress import get_progress_manager
            progress_manager = get_progress_manager()
            progress = progress_manager.get_progress(session_id)
            if progress:
                progress_manager.cancel_training(session_id)
        except Exception:
            pass  # 进度跟踪器可能不存在，忽略错误
        
        return success_response(updated_session.to_dict(), "训练会话已取消")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"内部服务器错误: {str(e)}", 500)


@training_bp.route('/scheduler/tasks', methods=['POST'])
@jwt_required()
def schedule_training_task_api():
    """调度训练任务
    
    Request:
        POST /api/v1/training/scheduler/tasks
        {
            "task_config": {...},  # 任务配置
            "schedule_time": "2023-10-01T10:00:00Z"  # 调度时间
        }
        
    Response:
        {
            "success": true,
            "data": {
                "task_id": "任务ID",
                "task_config": {...},
                "schedule_time": "2023-10-01T10:00:00Z",
                "status": "scheduled",
                "created_at": "创建时间"
            },
            "message": "任务调度成功"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        task_config = data.get('task_config')
        schedule_time_str = data.get('schedule_time')
        
        if not task_config:
            return error_response("任务配置不能为空", 400)
        
        if not schedule_time_str:
            return error_response("调度时间不能为空", 400)
        
        # 解析调度时间
        from datetime import datetime
        try:
            schedule_time = datetime.fromisoformat(schedule_time_str.replace('Z', '+00:00'))
        except ValueError:
            return error_response("调度时间格式错误", 400)
        
        # 调度任务
        task_id = schedule_training_task(task_config, schedule_time)
        
        # 返回任务信息
        task_info = {
            'task_id': task_id,
            'task_config': task_config,
            'schedule_time': schedule_time.isoformat(),
            'status': 'scheduled',
            'created_at': datetime.utcnow().isoformat()
        }
        
        return success_response(task_info, "任务调度成功", 201)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"内部服务器错误: {str(e)}", 500)


@training_bp.route('/scheduler/tasks', methods=['GET'])
@jwt_required()
def list_scheduled_tasks_api():
    """获取调度任务列表
    
    Request:
        GET /api/v1/training/scheduler/tasks
        
    Response:
        {
            "success": true,
            "data": [
                {
                    "task_id": "任务ID",
                    "task_config": {...},
                    "schedule_time": "调度时间",
                    "status": "任务状态",
                    "created_at": "创建时间"
                }
            ],
            "message": "获取成功"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        
        # 获取所有调度任务
        tasks = get_scheduled_training_tasks()
        
        # 转换为字典列表
        tasks_data = []
        for task in tasks:
            task_data = {
                'task_id': task.get('id', task.get('task_id')),
                'task_config': task.get('config', {}),
                'schedule_time': task.get('schedule_time', '').isoformat() if task.get('schedule_time') else None,
                'status': task.get('status', 'unknown'),
                'created_at': task.get('created_at', '').isoformat() if task.get('created_at') else None
            }
            tasks_data.append(task_data)
        
        return success_response(tasks_data, "获取调度任务列表成功")
        
    except Exception as e:
        return error_response(f"内部服务器错误: {str(e)}", 500)


@training_bp.route('/scheduler/tasks/<task_id>/cancel', methods=['POST'])
@jwt_required()
def cancel_scheduled_task_api(task_id: str):
    """取消调度任务
    
    Request:
        POST /api/v1/training/scheduler/tasks/{task_id}/cancel
        
    Response:
        {
            "success": true,
            "data": null,
            "message": "任务已取消"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        
        # 取消任务
        success = cancel_training_task(task_id)
        
        if success:
            return success_response(None, "任务已取消")
        else:
            return error_response("任务取消失败或任务不存在", 404)
        
    except Exception as e:
        return error_response(f"内部服务器错误: {str(e)}", 500)


@training_bp.route('/scenarios/submit', methods=['POST'])
@jwt_required()
def submit_training_scenario_api():
    """提交训练场景任务
    
    Request:
        POST /api/v1/training/scenarios/submit
        {
            "scenario_type": "basic_model",
            "config": {...}  # 场景配置
        }
        
    Response:
        {
            "success": true,
            "data": {
                "job_id": "任务ID",
                "scenario_type": "basic_model",
                "status": "submitted",
                "created_at": "创建时间"
            },
            "message": "场景任务提交成功"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        scenario_type = data.get('scenario_type')
        config = data.get('config', {})
        
        if not scenario_type:
            return error_response("场景类型不能为空", 400)
        
        # 获取场景管理器
        scenario_manager = get_scenario_manager()
        
        # 创建场景配置
        from backend.modules.training.scenarios import ScenarioConfig, TrainingScenario
        try:
            scenario_enum = TrainingScenario(scenario_type)
        except ValueError:
            return error_response(f"不支持的场景类型: {scenario_type}", 400)
        
        # 将参数转换为字典
        scenario_config_dict = {
            'scenario': scenario_enum,
            'name': f"{scenario_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'output_dir': config.get('output_dir', './outputs'),
            'custom_config': {
                'model_name': config.get('model_name', 'gpt2'),
                'train_data_path': config.get('train_data_path', './data/train'),
                'val_data_path': config.get('val_data_path'),
                'test_data_path': config.get('test_data_path'),
                'num_epochs': config.get('num_epochs', 10),
                'batch_size': config.get('batch_size', 16),
                'learning_rate': config.get('learning_rate', 2e-5)
            }
        }
        
        # 提交任务
        job_id = scenario_manager.submit_job(
            user_id=current_user_id,
            scenario_type=scenario_type,
            name=scenario_config_dict['name'],
            config=scenario_config_dict
        )
        
        # 返回任务信息
        job_info = {
            'job_id': job_id,
            'scenario_type': scenario_type,
            'status': 'submitted',
            'created_at': datetime.utcnow().isoformat()
        }
        
        return success_response(job_info, "场景任务提交成功", 201)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"内部服务器错误: {str(e)}", 500)


@training_bp.route('/scenarios/jobs/<job_id>', methods=['GET'])
@jwt_required()
def get_training_job_status_api(job_id: str):
    """获取训练任务状态
    
    Request:
        GET /api/v1/training/scenarios/jobs/{job_id}
        
    Response:
        {
            "success": true,
            "data": {
                "job_id": "任务ID",
                "status": "任务状态",
                "result": {...},  # 任务结果（如果已完成）
                "error": "错误信息（如果失败）",
                "created_at": "创建时间"
            },
            "message": "获取任务状态成功"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        
        # 获取场景管理器
        scenario_manager = get_scenario_manager()
        
        # 获取任务状态
        job_status = scenario_manager.get_job_status(job_id)
        
        if not job_status:
            return error_response("任务不存在", 404)
        
        return success_response(job_status, "获取任务状态成功")
        
    except Exception as e:
        return error_response(f"内部服务器错误: {str(e)}", 500)