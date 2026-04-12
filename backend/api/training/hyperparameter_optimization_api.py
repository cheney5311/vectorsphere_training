"""超参数优化API

提供超参数优化任务管理、试验记录、搜索空间模板等完整的API接口。
支持租户维度的数据隔离和持久化，通过Service层访问Repository层。
"""

import sys
import os
import logging
from flask import Blueprint, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.core.exceptions import ValidationError, BusinessLogicError
from backend.utils.response import success_response, error_response
from backend.schemas.enums import TrainingScenario

logger = logging.getLogger(__name__)

# 创建蓝图
hyperparameter_optimization_bp = Blueprint(
    'hyperparameter_optimization', 
    __name__, 
    url_prefix='/api/v1/training/hyperparameter'
)


def _get_tenant_id(user_id: str) -> str:
    """获取用户的租户ID
    
    优先从请求头获取，否则使用默认值
    """
    tenant_id = request.headers.get('X-Tenant-ID')
    if not tenant_id:
        tenant_id = f"tenant_{user_id}"
    return tenant_id


def _get_service():
    """获取超参数优化服务（延迟加载）"""
    try:
        from backend.services.hyperparameter_optimization_service import (
            get_hyperparameter_optimization_service
        )
        return get_hyperparameter_optimization_service(use_memory_storage=True)
    except ImportError as e:
        logger.error(f"Failed to import HyperparameterOptimizationService: {e}")
        return None


# ============================================================================
# 优化任务管理
# ============================================================================

@hyperparameter_optimization_bp.route('/optimizations', methods=['POST'])
@jwt_required()
def create_optimization():
    """创建超参数优化任务
    
    Request Body:
        {
            "name": "string",
            "description": "string" (optional),
            "scenario_type": "string",
            "optimization_method": "string" (random, grid, bayesian),
            "search_space": [
                {
                    "name": "string",
                    "type": "string" (int, float, categorical),
                    "low": number (optional),
                    "high": number (optional),
                    "choices": [] (optional),
                    "default": any (optional)
                }
            ],
            "training_config": {} (optional),
            "max_trials": number (default: 10),
            "model_id": "string" (optional),
            "dataset_id": "string" (optional)
        }
    
    Returns:
        {
            "optimization": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
        
        # 验证必填字段
        if not data.get('name'):
            return error_response("优化任务名称不能为空", 400)
        if not data.get('search_space'):
            return error_response("搜索空间不能为空", 400)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        optimization = service.create_optimization(tenant_id, user_id, data)
        
        logger.info(f"Created optimization: {optimization.get('id')} for user {user_id}")
        
        return success_response({
            "optimization": optimization
        }, "超参数优化任务创建成功", 201)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to create optimization: {e}")
        return error_response(f"创建优化任务失败: {str(e)}", 500)


@hyperparameter_optimization_bp.route('/optimizations', methods=['GET'])
@jwt_required()
def list_optimizations():
    """获取优化任务列表
    
    Query Parameters:
        - status: 状态过滤 (pending, running, completed, failed, cancelled)
        - scenario_type: 场景类型过滤
        - page: 页码 (默认: 1)
        - limit: 每页数量 (默认: 20)
    
    Returns:
        {
            "optimizations": [...],
            "pagination": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        status = request.args.get('status')
        scenario_type = request.args.get('scenario_type')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        result = service.list_optimizations(
            tenant_id=tenant_id,
            user_id=user_id,
            status=status,
            scenario_type=scenario_type,
            page=page,
            page_size=limit
        )
        
        return success_response({
            "optimizations": result.get('items', []),
            "pagination": {
                "page": result.get('page', page),
                "limit": result.get('page_size', limit),
                "total": result.get('total', 0),
                "pages": result.get('total_pages', 0)
            }
        }, "获取优化任务列表成功")
        
    except Exception as e:
        logger.error(f"Failed to list optimizations: {e}")
        return error_response(f"获取优化任务列表失败: {str(e)}", 500)


@hyperparameter_optimization_bp.route('/optimizations/<optimization_id>', methods=['GET'])
@jwt_required()
def get_optimization(optimization_id):
    """获取优化任务详情
    
    Returns:
        {
            "optimization": {...},
            "trials": [...]
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        optimization = service.get_optimization(optimization_id, tenant_id)
        if not optimization:
            return error_response("优化任务不存在", 404)
        
        # 验证权限
        if optimization.get('user_id') != user_id:
            return error_response("无权限访问此优化任务", 403)
        
        # 获取试验记录
        trials_result = service.get_trials(optimization_id, tenant_id)
        
        return success_response({
            "optimization": optimization,
            "trials": trials_result.get('trials', []),
            "best_trial": trials_result.get('best_trial')
        }, "获取优化任务详情成功")
        
    except Exception as e:
        logger.error(f"Failed to get optimization: {e}")
        return error_response(f"获取优化任务详情失败: {str(e)}", 500)


@hyperparameter_optimization_bp.route('/optimizations/<optimization_id>', methods=['PUT'])
@jwt_required()
def update_optimization(optimization_id):
    """更新优化任务
    
    Request Body:
        {
            "name": "string" (optional),
            "description": "string" (optional),
            "tags": [] (optional)
        }
    
    Returns:
        {
            "optimization": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        # 验证权限
        optimization = service.get_optimization(optimization_id, tenant_id)
        if not optimization:
            return error_response("优化任务不存在", 404)
        if optimization.get('user_id') != user_id:
            return error_response("无权限更新此优化任务", 403)
        
        # 只允许更新特定字段
        update_data = {}
        for field in ['name', 'description', 'tags']:
            if field in data:
                update_data[field] = data[field]
        
        updated = service.update_optimization(optimization_id, tenant_id, update_data)
        
        return success_response({
            "optimization": updated
        }, "优化任务更新成功")
        
    except Exception as e:
        logger.error(f"Failed to update optimization: {e}")
        return error_response(f"更新优化任务失败: {str(e)}", 500)


@hyperparameter_optimization_bp.route('/optimizations/<optimization_id>', methods=['DELETE'])
@jwt_required()
def delete_optimization(optimization_id):
    """删除优化任务
    
    Returns:
        {
            "message": "string"
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        optimization = service.get_optimization(optimization_id, tenant_id)
        if not optimization:
            return error_response("优化任务不存在", 404)
        
        # 验证权限
        if optimization.get('user_id') != user_id:
            return error_response("无权限删除此优化任务", 403)
        
        success = service.delete_optimization(optimization_id, tenant_id)
        if not success:
            return error_response("删除优化任务失败", 500)
        
        logger.info(f"Deleted optimization: {optimization_id}")
        
        return success_response({}, "优化任务已删除")
        
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to delete optimization: {e}")
        return error_response(f"删除优化任务失败: {str(e)}", 500)


# ============================================================================
# 优化执行控制
# ============================================================================

@hyperparameter_optimization_bp.route('/optimizations/<optimization_id>/start', methods=['POST'])
@jwt_required()
def start_optimization(optimization_id):
    """启动优化任务
    
    Returns:
        {
            "optimization": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        result = service.start_optimization(optimization_id, tenant_id, user_id)
        
        logger.info(f"Started optimization: {optimization_id}")
        
        return success_response({
            "optimization": result
        }, "优化任务已启动")
        
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to start optimization: {e}")
        return error_response(f"启动优化任务失败: {str(e)}", 500)


@hyperparameter_optimization_bp.route('/optimizations/<optimization_id>/cancel', methods=['POST'])
@jwt_required()
def cancel_optimization(optimization_id):
    """取消优化任务
    
    Returns:
        {
            "optimization": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        result = service.cancel_optimization(optimization_id, tenant_id, user_id)
        if not result:
            return error_response("优化任务不存在", 404)
        
        logger.info(f"Cancelled optimization: {optimization_id}")
        
        return success_response({
            "optimization": result
        }, "优化任务已取消")
        
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to cancel optimization: {e}")
        return error_response(f"取消优化任务失败: {str(e)}", 500)


# ============================================================================
# 试验管理
# ============================================================================

@hyperparameter_optimization_bp.route('/optimizations/<optimization_id>/trials', methods=['GET'])
@jwt_required()
def get_trials(optimization_id):
    """获取优化任务的试验记录
    
    Query Parameters:
        - status: 状态过滤
    
    Returns:
        {
            "trials": [...],
            "best_trial": {...},
            "total": number
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        status = request.args.get('status')
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        # 验证权限
        optimization = service.get_optimization(optimization_id, tenant_id)
        if not optimization:
            return error_response("优化任务不存在", 404)
        if optimization.get('user_id') != user_id:
            return error_response("无权限访问此优化任务", 403)
        
        result = service.get_trials(optimization_id, tenant_id, status)
        
        return success_response(result, "获取试验记录成功")
        
    except Exception as e:
        logger.error(f"Failed to get trials: {e}")
        return error_response(f"获取试验记录失败: {str(e)}", 500)


@hyperparameter_optimization_bp.route('/optimizations/<optimization_id>/trials/<trial_id>', methods=['GET'])
@jwt_required()
def get_trial(optimization_id, trial_id):
    """获取试验详情
    
    Returns:
        {
            "trial": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        # 验证权限
        optimization = service.get_optimization(optimization_id, tenant_id)
        if not optimization:
            return error_response("优化任务不存在", 404)
        if optimization.get('user_id') != user_id:
            return error_response("无权限访问此优化任务", 403)
        
        trial = service.get_trial(optimization_id, trial_id, tenant_id)
        if not trial:
            return error_response("试验记录不存在", 404)
        
        return success_response({
            "trial": trial
        }, "获取试验详情成功")
        
    except Exception as e:
        logger.error(f"Failed to get trial: {e}")
        return error_response(f"获取试验详情失败: {str(e)}", 500)


# ============================================================================
# 搜索空间操作
# ============================================================================

@hyperparameter_optimization_bp.route('/search-space/validate', methods=['POST'])
@jwt_required()
def validate_search_space():
    """验证搜索空间定义
    
    Request Body:
        {
            "params": [...]
        }
    
    Returns:
        {
            "valid": boolean,
            "search_space": [...],
            "errors": []
        }
    """
    try:
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
            
        params = data.get('params', [])
        if not params:
            return error_response("参数列表不能为空", 400)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        try:
            search_space = service.define_search_space(params)
            search_space_data = [space.to_dict() for space in search_space]
            
            return success_response({
                "valid": True,
                "search_space": search_space_data,
                "errors": []
            }, "搜索空间验证成功")
            
        except ValidationError as e:
            return success_response({
                "valid": False,
                "search_space": [],
                "errors": [str(e)]
            }, "搜索空间验证失败")
        
    except Exception as e:
        logger.error(f"Failed to validate search space: {e}")
        return error_response(f"验证搜索空间失败: {str(e)}", 500)


@hyperparameter_optimization_bp.route('/search-space/suggest', methods=['POST'])
@jwt_required()
def suggest_params():
    """建议下一组超参数
    
    Request Body:
        {
            "search_space": [...],
            "method": "string" (random, grid, bayesian),
            "count": number (default: 1)
        }
    
    Returns:
        {
            "suggestions": [...]
        }
    """
    try:
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
        
        search_space_data = data.get('search_space', [])
        method = data.get('method', 'random')
        count = min(data.get('count', 1), 10)
        
        if not search_space_data:
            return error_response("搜索空间不能为空", 400)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        # 转换搜索空间
        search_space = service.define_search_space(search_space_data)
        
        # 生成建议
        suggestions = []
        for _ in range(count):
            params = service.suggest_next_params(search_space, method)
            suggestions.append(params)
        
        return success_response({
            "suggestions": suggestions,
            "method": method
        }, "超参数建议成功")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to suggest params: {e}")
        return error_response(f"建议超参数失败: {str(e)}", 500)


# ============================================================================
# 搜索空间模板
# ============================================================================

@hyperparameter_optimization_bp.route('/templates', methods=['GET'])
@jwt_required()
def list_templates():
    """获取搜索空间模板列表
    
    Query Parameters:
        - scenario_type: 场景类型过滤
        - page: 页码 (默认: 1)
        - limit: 每页数量 (默认: 20)
    
    Returns:
        {
            "templates": [...],
            "pagination": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        scenario_type = request.args.get('scenario_type')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        result = service.list_templates(
            tenant_id=tenant_id,
            scenario_type=scenario_type,
            page=page,
            page_size=limit
        )
        
        return success_response({
            "templates": result.get('items', []),
            "pagination": {
                "page": result.get('page', page),
                "limit": result.get('page_size', limit),
                "total": result.get('total', 0),
                "pages": result.get('total_pages', 0)
            }
        }, "获取模板列表成功")
        
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        return error_response(f"获取模板列表失败: {str(e)}", 500)


@hyperparameter_optimization_bp.route('/templates/<template_id>', methods=['GET'])
@jwt_required()
def get_template(template_id):
    """获取模板详情
    
    Returns:
        {
            "template": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        template = service.get_template(template_id, tenant_id)
        if not template:
            return error_response("模板不存在", 404)
        
        return success_response({
            "template": template
        }, "获取模板详情成功")
        
    except Exception as e:
        logger.error(f"Failed to get template: {e}")
        return error_response(f"获取模板详情失败: {str(e)}", 500)


@hyperparameter_optimization_bp.route('/templates', methods=['POST'])
@jwt_required()
def create_template():
    """创建搜索空间模板
    
    Request Body:
        {
            "name": "string",
            "description": "string" (optional),
            "scenario_type": "string",
            "parameters": [...],
            "recommended_method": "string" (optional),
            "recommended_trials": number (optional),
            "is_public": boolean (default: false)
        }
    
    Returns:
        {
            "template": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
        
        if not data.get('name'):
            return error_response("模板名称不能为空", 400)
        if not data.get('parameters'):
            return error_response("参数列表不能为空", 400)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        template = service.create_template(tenant_id, user_id, data)
        
        logger.info(f"Created template: {template.get('id')}")
        
        return success_response({
            "template": template
        }, "模板创建成功", 201)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to create template: {e}")
        return error_response(f"创建模板失败: {str(e)}", 500)


@hyperparameter_optimization_bp.route('/templates/<template_id>/apply', methods=['POST'])
@jwt_required()
def apply_template(template_id):
    """应用模板创建优化任务
    
    Request Body:
        {
            "name": "string",
            "description": "string" (optional),
            "training_config": {} (optional)
        }
    
    Returns:
        {
            "optimization": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
        
        if not data.get('name'):
            return error_response("优化任务名称不能为空", 400)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        optimization = service.apply_template(template_id, tenant_id, user_id, data)
        
        logger.info(f"Created optimization from template: {optimization.get('id')}")
        
        return success_response({
            "optimization": optimization
        }, "从模板创建优化任务成功", 201)
        
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to apply template: {e}")
        return error_response(f"应用模板失败: {str(e)}", 500)


# ============================================================================
# 统计与导出
# ============================================================================

@hyperparameter_optimization_bp.route('/statistics', methods=['GET'])
@jwt_required()
def get_statistics():
    """获取超参数优化统计信息
    
    Returns:
        {
            "statistics": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        stats = service.get_statistics(tenant_id, user_id)
        
        return success_response({
            "statistics": stats
        }, "获取统计信息成功")
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return error_response(f"获取统计信息失败: {str(e)}", 500)


@hyperparameter_optimization_bp.route('/optimizations/<optimization_id>/export', methods=['GET'])
@jwt_required()
def export_optimization(optimization_id):
    """导出优化结果
    
    Query Parameters:
        - format: 导出格式 (json, csv)
    
    Returns:
        优化结果数据
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        optimization = service.get_optimization(optimization_id, tenant_id)
        if not optimization:
            return error_response("优化任务不存在", 404)
        
        # 验证权限
        if optimization.get('user_id') != user_id:
            return error_response("无权限访问此优化任务", 403)
        
        # 获取试验记录
        trials_result = service.get_trials(optimization_id, tenant_id)
        trials_data = trials_result.get('trials', [])
        
        export_format = request.args.get('format', 'json')
        
        if export_format == 'csv':
            # CSV格式
            import io
            import csv
            
            output = io.StringIO()
            
            if trials_data:
                # 获取所有参数名
                all_params = set()
                for t in trials_data:
                    if t.get('params'):
                        all_params.update(t['params'].keys())
                
                fieldnames = ['trial_number', 'status', 'score'] + list(all_params) + ['completed_at']
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                
                for t in trials_data:
                    row = {
                        'trial_number': t.get('trial_number'),
                        'status': t.get('status'),
                        'score': t.get('score'),
                        'completed_at': t.get('completed_at')
                    }
                    if t.get('params'):
                        row.update(t['params'])
                    writer.writerow(row)
            
            return success_response({
                "format": "csv",
                "data": output.getvalue(),
                "filename": f"optimization_{optimization_id}.csv"
            }, "导出成功")
        
        else:
            # JSON格式
            export_data = {
                "optimization": optimization,
                "trials": trials_data,
                "best_trial": trials_result.get('best_trial'),
                "exported_at": datetime.utcnow().isoformat()
            }
            
            return success_response({
                "format": "json",
                "data": export_data,
                "filename": f"optimization_{optimization_id}.json"
            }, "导出成功")
        
    except Exception as e:
        logger.error(f"Failed to export optimization: {e}")
        return error_response(f"导出优化结果失败: {str(e)}", 500)


# ============================================================================
# 快速优化（兼容旧接口）
# ============================================================================

@hyperparameter_optimization_bp.route('/optimize', methods=['POST'])
@jwt_required()
def quick_optimize():
    """快速执行超参数优化（同步，兼容旧接口）
    
    Request Body:
        {
            "search_space": [...],
            "scenario_type": "string",
            "training_config": {},
            "max_trials": number,
            "method": "string"
        }
    
    Returns:
        {
            "best_params": {...},
            "best_score": number,
            "trials": [...]
        }
    """
    try:
        user_id = get_jwt_identity()
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
            
        search_space_data = data.get('search_space', [])
        scenario_type_str = data.get('scenario_type')
        training_config = data.get('training_config', {})
        max_trials = data.get('max_trials', 10)
        method = data.get('method', 'random')
        
        if not search_space_data:
            return error_response("搜索空间不能为空", 400)
            
        if not scenario_type_str:
            return error_response("场景类型不能为空", 400)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        # 转换搜索空间
        search_space = service.define_search_space(search_space_data)
            
        # 转换场景类型
        try:
            scenario_type = TrainingScenario(scenario_type_str)
        except ValueError:
            return error_response(f"不支持的场景类型: {scenario_type_str}", 400)
            
        # 执行优化
        result = service.optimize_hyperparameters(
            user_id=user_id,
            search_space=search_space,
            scenario_type=scenario_type,
            training_config=training_config,
            max_trials=max_trials,
            method=method
        )
        
        # 转换结果
        result_data = {
            'best_params': result.best_params,
            'best_score': result.best_score,
            'completed_at': result.completed_at.isoformat(),
            'trials': []
        }
        
        for trial in result.trials:
            trial_data = {
                'trial': trial['trial'],
                'params': trial['params'],
                'score': trial['score'],
                'evaluated_at': trial['result']['evaluated_at']
            }
            result_data['trials'].append(trial_data)
            
        return success_response(result_data, "超参数优化完成")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        logger.error(f"Failed to optimize: {e}")
        return error_response(f"超参数优化失败: {str(e)}", 500)


@hyperparameter_optimization_bp.route('/history', methods=['GET'])
@jwt_required()
def get_history():
    """获取超参数优化历史（兼容旧接口）
    
    Returns:
        {
            "history": [...]
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        result = service.list_optimizations(
            tenant_id=tenant_id,
            user_id=user_id,
            status='completed',
            page=1,
            page_size=50
        )
        
        history = []
        for opt in result.get('items', []):
            history.append({
                'id': opt.get('id'),
                'user_id': opt.get('user_id'),
                'scenario_type': opt.get('scenario_type'),
                'best_params': opt.get('best_params'),
                'best_score': opt.get('best_score'),
                'max_trials': opt.get('max_trials'),
                'method': opt.get('optimization_method'),
                'created_at': opt.get('created_at'),
                'completed_at': opt.get('completed_at')
            })
        
        return success_response({
            "history": history
        }, "获取优化历史成功")
        
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return error_response(f"获取优化历史失败: {str(e)}", 500)
