# -*- coding: utf-8 -*-
"""训练统计API接口

提供训练统计信息相关的API接口，支持多维度的统计分析。

功能特性:
- 基础统计：任务数量、成功率、运行状态
- 详细统计：时间分析、资源使用、模型分布
- 趋势分析：每日/每周/每月趋势
- 实时监控：运行中任务状态
- 资源统计：GPU/CPU使用情况
- 导出功能：统计数据导出
"""

import sys
import os
from flask import Blueprint, request, jsonify, Response
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime, timedelta
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.exceptions import ValidationError, BusinessLogicError
from backend.utils.response import success_response, error_response
from backend.services.training_statistics_service import (
    get_training_statistics_service,
    StatisticsTimeRange,
    StatisticsGroupBy
)

# 创建蓝图
training_statistics_bp = Blueprint('training_statistics', __name__, url_prefix='/api/v1/training/statistics')


def _get_current_user() -> str:
    """获取当前用户ID"""
    try:
        return get_jwt_identity()
    except Exception:
        return None


def _get_tenant_id() -> str:
    """获取租户ID
    
    从请求头或JWT中获取租户ID
    """
    # 优先从请求头获取
    tenant_id = request.headers.get('X-Tenant-ID')
    if tenant_id:
        return tenant_id
    
    # 尝试从JWT获取
    try:
        identity = get_jwt_identity()
        if isinstance(identity, dict):
            return identity.get('tenant_id')
    except Exception:
        pass
    
    # 返回默认租户
    return 'default'


# ==================== 基础统计 ====================

@training_statistics_bp.route('/basic', methods=['GET'])
@jwt_required()
def get_basic_statistics():
    """获取基础训练统计信息
    
    提供核心的训练任务统计数据，包括任务数量、成功率等。
    
    Query Parameters:
        user_id: 可选，过滤特定用户的统计
    
    Returns:
        {
            "success": true,
            "data": {
                "total_jobs": "integer - 总任务数",
                "running_jobs": "integer - 运行中任务数", 
                "completed_jobs": "integer - 已完成任务数",
                "failed_jobs": "integer - 失败任务数",
                "pending_jobs": "integer - 待处理任务数",
                "paused_jobs": "integer - 暂停任务数",
                "cancelled_jobs": "integer - 取消任务数",
                "success_rate": "float - 成功率(0-1)",
                "average_duration": "float - 平均时长(秒)",
                "total_training_hours": "float - 总训练时长(小时)",
                "timestamp": "string - 时间戳"
            },
            "message": "获取基础统计信息成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        # 可选的用户过滤
        filter_user_id = request.args.get('user_id', user_id)
        
        service = get_training_statistics_service()
        stats = service.get_basic_statistics(
            tenant_id=tenant_id,
            user_id=filter_user_id
        )
        
        return success_response(stats, "获取基础统计信息成功")
        
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"获取基础统计信息失败: {str(e)}", 500)


@training_statistics_bp.route('/detailed', methods=['GET'])
@jwt_required()
def get_detailed_statistics():
    """获取详细训练统计信息
    
    提供全面的训练统计数据，包括资源使用、模型分布、每日趋势等。
    
    Query Parameters:
        days: 统计天数，默认30天
        user_id: 可选，过滤特定用户的统计
    
    Returns:
        {
            "success": true,
            "data": {
                "total_jobs": "integer - 总任务数",
                "running_jobs": "integer - 运行中任务数",
                "completed_jobs": "integer - 已完成任务数", 
                "failed_jobs": "integer - 失败任务数",
                "cancelled_jobs": "integer - 取消任务数",
                "success_rate": "float - 成功率",
                "average_training_time": "float - 平均训练时间(秒)",
                "total_training_hours": "float - 总训练时长(小时)",
                "most_used_model": "string - 最常用模型",
                "resource_usage": {
                    "cpu_avg": "float - CPU平均使用率(%)",
                    "memory_avg": "float - 内存平均使用(GB)",
                    "gpu_avg": "float - GPU平均使用率(%)"
                },
                "daily_stats": [
                    {
                        "date": "string - 日期",
                        "jobs_count": "integer - 任务数",
                        "success_count": "integer - 成功数"
                    }
                ],
                "top_models": [...],
                "scenario_breakdown": [...]
            },
            "message": "获取详细统计信息成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        days = request.args.get('days', 30, type=int)
        filter_user_id = request.args.get('user_id', user_id)
        
        # 参数验证
        if days < 1 or days > 365:
            return error_response("统计天数必须在1-365之间", 400)
        
        service = get_training_statistics_service()
        stats = service.get_detailed_statistics(
            tenant_id=tenant_id,
            user_id=filter_user_id,
            days=days
        )
        
        return success_response(stats, "获取详细统计信息成功")
        
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"获取详细统计信息失败: {str(e)}", 500)


@training_statistics_bp.route('/overview', methods=['GET'])
@jwt_required()
def get_statistics_overview():
    """获取统计概览信息
    
    提供统计信息的综合概览，包含整体统计和近期趋势。
    
    Returns:
        {
            "success": true,
            "data": {
                "overall_stats": {...},
                "recent_trends": [...],
                "performance_metrics": {...},
                "top_models": [...],
                "scenario_breakdown": [...]
            },
            "message": "获取统计概览成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_statistics_service()
        stats = service.get_statistics_overview(
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        return success_response(stats, "获取统计概览成功")
        
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"获取统计概览失败: {str(e)}", 500)


# ==================== 趋势统计 ====================

@training_statistics_bp.route('/trends', methods=['GET'])
@jwt_required()
def get_trend_statistics():
    """获取趋势统计信息
    
    提供指定时间范围内的统计趋势数据。
    
    Query Parameters:
        days: 统计天数，默认7天
        group_by: 分组方式 (hour/day/week/month)，默认day
    
    Returns:
        {
            "success": true,
            "data": {
                "period": "string - 时间范围描述",
                "group_by": "string - 分组方式",
                "total_jobs": "integer - 总任务数",
                "completed_jobs": "integer - 完成任务数",
                "failed_jobs": "integer - 失败任务数",
                "success_rate": "float - 成功率",
                "trend_data": [
                    {
                        "date": "string - 日期",
                        "jobs": "integer - 任务数",
                        "completed": "integer - 完成数",
                        "failed": "integer - 失败数",
                        "success_rate": "float - 成功率"
                    }
                ]
            },
            "message": "获取趋势统计成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        days = request.args.get('days', 7, type=int)
        group_by_str = request.args.get('group_by', 'day')
        
        # 参数验证
        if days < 1 or days > 365:
            return error_response("统计天数必须在1-365之间", 400)
        
        # 解析分组方式
        try:
            group_by = StatisticsGroupBy(group_by_str)
        except ValueError:
            group_by = StatisticsGroupBy.DAY
        
        service = get_training_statistics_service()
        stats = service.get_trend_statistics(
            tenant_id=tenant_id,
            user_id=user_id,
            days=days,
            group_by=group_by
        )
        
        return success_response(stats, "获取趋势统计成功")
        
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"获取趋势统计失败: {str(e)}", 500)


@training_statistics_bp.route('/daily', methods=['GET'])
@jwt_required()
def get_daily_statistics():
    """获取每日统计信息
    
    Query Parameters:
        days: 统计天数，默认30天
    
    Returns:
        每日统计数据列表
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        days = request.args.get('days', 30, type=int)
        
        if days < 1 or days > 365:
            return error_response("统计天数必须在1-365之间", 400)
        
        service = get_training_statistics_service()
        stats = service.get_trend_statistics(
            tenant_id=tenant_id,
            user_id=user_id,
            days=days,
            group_by=StatisticsGroupBy.DAY
        )
        
        return success_response(stats.get('trend_data', []), "获取每日统计成功")
        
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"获取每日统计失败: {str(e)}", 500)


# ==================== 资源统计 ====================

@training_statistics_bp.route('/resources', methods=['GET'])
@jwt_required()
def get_resource_statistics():
    """获取资源使用统计
    
    提供GPU、CPU等资源的使用统计信息。
    
    Query Parameters:
        days: 统计天数，默认7天
    
    Returns:
        {
            "success": true,
            "data": {
                "period": "string - 时间范围",
                "gpu": {
                    "average_utilization": "float - 平均GPU使用率(%)",
                    "max_utilization": "float - 最大GPU使用率(%)",
                    "average_memory_used_gb": "float - 平均显存使用(GB)"
                },
                "cpu": {
                    "average_utilization": "float - 平均CPU使用率(%)",
                    "max_utilization": "float - 最大CPU使用率(%)",
                    "average_memory_used_gb": "float - 平均内存使用(GB)"
                }
            },
            "message": "获取资源统计成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        days = request.args.get('days', 7, type=int)
        
        if days < 1 or days > 365:
            return error_response("统计天数必须在1-365之间", 400)
        
        service = get_training_statistics_service()
        stats = service.get_resource_statistics(
            tenant_id=tenant_id,
            user_id=user_id,
            days=days
        )
        
        return success_response(stats, "获取资源统计成功")
        
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"获取资源统计失败: {str(e)}", 500)


# ==================== 模型统计 ====================

@training_statistics_bp.route('/models', methods=['GET'])
@jwt_required()
def get_model_statistics():
    """获取模型使用统计
    
    提供各模型的使用频率和成功率统计。
    
    Query Parameters:
        limit: 返回数量限制，默认10
    
    Returns:
        {
            "success": true,
            "data": {
                "top_models": [
                    {
                        "model_name": "string - 模型名称",
                        "usage_count": "integer - 使用次数",
                        "success_count": "integer - 成功次数",
                        "success_rate": "float - 成功率(%)"
                    }
                ],
                "total_models": "integer - 模型总数"
            },
            "message": "获取模型统计成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        limit = request.args.get('limit', 10, type=int)
        
        if limit < 1 or limit > 100:
            return error_response("返回数量必须在1-100之间", 400)
        
        service = get_training_statistics_service()
        stats = service.get_model_statistics(
            tenant_id=tenant_id,
            user_id=user_id,
            limit=limit
        )
        
        return success_response(stats, "获取模型统计成功")
        
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"获取模型统计失败: {str(e)}", 500)


# ==================== 场景统计 ====================

@training_statistics_bp.route('/scenarios', methods=['GET'])
@jwt_required()
def get_scenario_statistics():
    """获取训练场景统计
    
    提供各训练场景的使用情况统计。
    
    Returns:
        {
            "success": true,
            "data": {
                "scenarios": [
                    {
                        "scenario_type": "string - 场景类型",
                        "total_count": "integer - 总数",
                        "completed_count": "integer - 完成数",
                        "failed_count": "integer - 失败数",
                        "success_rate": "float - 成功率(%)"
                    }
                ],
                "total_scenarios": "integer - 场景类型总数"
            },
            "message": "获取场景统计成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_statistics_service()
        stats = service.get_scenario_statistics(
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        return success_response(stats, "获取场景统计成功")
        
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"获取场景统计失败: {str(e)}", 500)


# ==================== 性能指标统计 ====================

@training_statistics_bp.route('/performance', methods=['GET'])
@jwt_required()
def get_performance_statistics():
    """获取性能指标统计
    
    提供训练性能相关指标的统计信息。
    
    Returns:
        {
            "success": true,
            "data": {
                "loss": {
                    "average": "float - 平均损失值",
                    "minimum": "float - 最小损失值"
                },
                "accuracy": {
                    "average": "float - 平均准确率",
                    "maximum": "float - 最高准确率"
                },
                "throughput": {
                    "average_samples_per_second": "float - 平均样本处理速度",
                    "max_samples_per_second": "float - 最大样本处理速度"
                }
            },
            "message": "获取性能指标统计成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_statistics_service()
        stats = service.get_performance_statistics(
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        return success_response(stats, "获取性能指标统计成功")
        
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"获取性能指标统计失败: {str(e)}", 500)


# ==================== 任务统计 ====================

@training_statistics_bp.route('/jobs/<job_id>', methods=['GET'])
@jwt_required()
def get_job_statistics(job_id: str):
    """获取特定任务的统计信息
    
    Args:
        job_id: 任务ID
    
    Returns:
        {
            "success": true,
            "data": {
                "job_id": "string - 任务ID",
                "status": "string - 任务状态",
                "created_at": "string - 创建时间",
                "result": "object - 任务结果",
                "error": "string - 错误信息",
                "duration": "string - 持续时间"
            },
            "message": "获取任务统计成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        if not job_id:
            return error_response("任务ID不能为空", 400)
        
        service = get_training_statistics_service()
        stats = service.get_job_statistics(
            tenant_id=tenant_id,
            user_id=user_id,
            job_id=job_id
        )
        
        return success_response(stats, "获取任务统计成功")
        
    except ValidationError as e:
        return error_response(str(e), 404)
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"获取任务统计失败: {str(e)}", 500)


# ==================== 整体统计 ====================

@training_statistics_bp.route('/overall', methods=['GET'])
@jwt_required()
def get_overall_statistics():
    """获取整体训练统计信息
    
    提供全面的训练系统统计概览。
    
    Returns:
        {
            "success": true,
            "data": {
                "period": {...},
                "summary": {...},
                "time_statistics": {...},
                "resource_usage": {...},
                "top_models": [...],
                "scenario_breakdown": [...],
                "performance_metrics": {...},
                "uptime": "string - 服务运行时间"
            },
            "message": "获取整体统计成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_statistics_service()
        stats = service.get_overall_statistics(
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        return success_response(stats, "获取整体统计成功")
        
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"获取整体统计失败: {str(e)}", 500)


# ==================== 实时统计 ====================

@training_statistics_bp.route('/realtime', methods=['GET'])
@jwt_required()
def get_realtime_statistics():
    """获取实时统计信息
    
    提供当日的实时统计数据。
    
    Returns:
        {
            "success": true,
            "data": {
                "today": {
                    "total_jobs": "integer - 今日任务总数",
                    "running_jobs": "integer - 运行中任务",
                    "completed_jobs": "integer - 完成任务",
                    "failed_jobs": "integer - 失败任务"
                },
                "current_time": "string - 当前时间",
                "uptime": "string - 服务运行时间"
            },
            "message": "获取实时统计成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_statistics_service()
        stats = service.get_realtime_statistics(
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        return success_response(stats, "获取实时统计成功")
        
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"获取实时统计失败: {str(e)}", 500)


# ==================== 导出功能 ====================

@training_statistics_bp.route('/export', methods=['GET'])
@jwt_required()
def export_statistics():
    """导出统计数据
    
    导出指定时间范围的统计数据。
    
    Query Parameters:
        format: 导出格式 (json/csv)，默认json
        days: 统计天数，默认30天
    
    Returns:
        导出的统计数据（JSON格式或CSV文件下载）
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        export_format = request.args.get('format', 'json')
        days = request.args.get('days', 30, type=int)
        
        if export_format not in ['json', 'csv']:
            return error_response("导出格式必须是 json 或 csv", 400)
        
        if days < 1 or days > 365:
            return error_response("统计天数必须在1-365之间", 400)
        
        service = get_training_statistics_service()
        export_data = service.export_statistics(
            tenant_id=tenant_id,
            user_id=user_id,
            format=export_format,
            days=days
        )
        
        if export_format == 'csv':
            # CSV 格式导出
            csv_content = _convert_to_csv(export_data)
            return Response(
                csv_content,
                mimetype='text/csv',
                headers={
                    'Content-Disposition': f'attachment; filename=training_statistics_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv'
                }
            )
        else:
            # JSON 格式
            return success_response(export_data, "导出统计数据成功")
        
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"导出统计数据失败: {str(e)}", 500)


def _convert_to_csv(data: dict) -> str:
    """将统计数据转换为CSV格式"""
    lines = []
    
    # 元数据
    metadata = data.get('metadata', {})
    lines.append("# 训练统计数据导出")
    lines.append(f"# 租户ID: {metadata.get('tenant_id', 'N/A')}")
    lines.append(f"# 导出时间: {metadata.get('export_time', 'N/A')}")
    lines.append(f"# 统计周期: {metadata.get('period_days', 'N/A')} 天")
    lines.append("")
    
    # 整体统计
    overall = data.get('overall', {}).get('summary', {})
    lines.append("## 整体统计")
    lines.append("指标,数值")
    lines.append(f"总任务数,{overall.get('total_jobs', 0)}")
    lines.append(f"完成任务,{overall.get('completed_jobs', 0)}")
    lines.append(f"失败任务,{overall.get('failed_jobs', 0)}")
    lines.append(f"运行中任务,{overall.get('running_jobs', 0)}")
    lines.append(f"成功率,{overall.get('success_rate', 0)}%")
    lines.append("")
    
    # 趋势数据
    trends = data.get('trends', {}).get('trend_data', [])
    if trends:
        lines.append("## 每日趋势")
        lines.append("日期,任务数,完成数,失败数,成功率")
        for item in trends:
            lines.append(f"{item.get('date', '')},{item.get('jobs', 0)},{item.get('completed', 0)},{item.get('failed', 0)},{item.get('success_rate', 0)}")
    
    return "\n".join(lines)


# ==================== 健康检查 ====================

@training_statistics_bp.route('/health', methods=['GET'])
def health_check():
    """API健康检查
    
    Returns:
        API健康状态
    """
    try:
        service = get_training_statistics_service()
        
        return jsonify({
            'status': 'healthy',
            'service': 'training_statistics',
            'timestamp': datetime.utcnow().isoformat(),
            'repository_available': service._repository is not None,
            'scenario_manager_available': service._scenario_manager is not None
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'training_statistics',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500


# ==================== 比较分析 ====================

@training_statistics_bp.route('/compare', methods=['POST'])
@jwt_required()
def compare_statistics():
    """比较两个时间段的统计数据
    
    Request Body:
        {
            "period1": {
                "start_date": "YYYY-MM-DD",
                "end_date": "YYYY-MM-DD"
            },
            "period2": {
                "start_date": "YYYY-MM-DD",
                "end_date": "YYYY-MM-DD"
            }
        }
    
    Returns:
        两个时间段的统计对比
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        data = request.get_json()
        if not data:
            return error_response("请求体不能为空", 400)
        
        period1 = data.get('period1', {})
        period2 = data.get('period2', {})
        
        if not period1 or not period2:
            return error_response("必须提供两个时间段进行比较", 400)
        
        # 解析日期
        try:
            p1_start = datetime.strptime(period1.get('start_date'), '%Y-%m-%d')
            p1_end = datetime.strptime(period1.get('end_date'), '%Y-%m-%d')
            p2_start = datetime.strptime(period2.get('start_date'), '%Y-%m-%d')
            p2_end = datetime.strptime(period2.get('end_date'), '%Y-%m-%d')
        except (ValueError, TypeError) as e:
            return error_response(f"日期格式错误: {e}", 400)
        
        service = get_training_statistics_service()
        
        # 获取两个时间段的统计
        if service._repository:
            stats1 = service._repository.get_job_count_by_status(
                tenant_id=tenant_id,
                user_id=user_id,
                start_date=p1_start,
                end_date=p1_end
            )
            stats2 = service._repository.get_job_count_by_status(
                tenant_id=tenant_id,
                user_id=user_id,
                start_date=p2_start,
                end_date=p2_end
            )
            
            # 计算变化
            comparison = {
                'period1': {
                    'date_range': f"{period1['start_date']} - {period1['end_date']}",
                    'statistics': stats1
                },
                'period2': {
                    'date_range': f"{period2['start_date']} - {period2['end_date']}",
                    'statistics': stats2
                },
                'changes': {
                    'total_jobs_change': stats2.get('total', 0) - stats1.get('total', 0),
                    'completed_jobs_change': stats2.get('completed', 0) - stats1.get('completed', 0),
                    'failed_jobs_change': stats2.get('failed', 0) - stats1.get('failed', 0)
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return success_response(comparison, "统计对比成功")
        else:
            return error_response("统计服务暂不可用", 503)
        
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"统计对比失败: {str(e)}", 500)


# ==================== 聚合查询 ====================

@training_statistics_bp.route('/aggregate', methods=['POST'])
@jwt_required()
def aggregate_statistics():
    """聚合查询统计数据
    
    支持自定义维度和指标的聚合查询。
    
    Request Body:
        {
            "dimensions": ["scenario_type", "model_name"],
            "metrics": ["count", "success_rate", "avg_duration"],
            "filters": {
                "status": ["completed", "failed"],
                "start_date": "YYYY-MM-DD",
                "end_date": "YYYY-MM-DD"
            },
            "limit": 100
        }
    
    Returns:
        聚合统计结果
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        data = request.get_json()
        if not data:
            return error_response("请求体不能为空", 400)
        
        dimensions = data.get('dimensions', [])
        metrics = data.get('metrics', ['count'])
        filters = data.get('filters', {})
        limit = data.get('limit', 100)
        
        # 验证维度
        valid_dimensions = ['scenario_type', 'model_name', 'status', 'training_mode']
        for dim in dimensions:
            if dim not in valid_dimensions:
                return error_response(f"不支持的维度: {dim}", 400)
        
        # 验证指标
        valid_metrics = ['count', 'success_rate', 'avg_duration', 'total_duration']
        for metric in metrics:
            if metric not in valid_metrics:
                return error_response(f"不支持的指标: {metric}", 400)
        
        service = get_training_statistics_service()
        
        # 获取基础统计作为聚合基础
        if service._repository:
            # 按场景聚合
            if 'scenario_type' in dimensions:
                scenario_stats = service._repository.get_scenario_statistics(
                    tenant_id=tenant_id,
                    user_id=user_id
                )
                
                return success_response({
                    'dimensions': dimensions,
                    'metrics': metrics,
                    'data': scenario_stats,
                    'total_records': len(scenario_stats),
                    'timestamp': datetime.utcnow().isoformat()
                }, "聚合查询成功")
            
            # 按模型聚合
            elif 'model_name' in dimensions:
                model_stats = service._repository.get_model_usage_statistics(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    limit=limit
                )
                
                return success_response({
                    'dimensions': dimensions,
                    'metrics': metrics,
                    'data': model_stats,
                    'total_records': len(model_stats),
                    'timestamp': datetime.utcnow().isoformat()
                }, "聚合查询成功")
            
            else:
                # 默认按状态聚合
                status_stats = service._repository.get_job_count_by_status(
                    tenant_id=tenant_id,
                    user_id=user_id
                )
                
                return success_response({
                    'dimensions': ['status'],
                    'metrics': metrics,
                    'data': status_stats,
                    'timestamp': datetime.utcnow().isoformat()
                }, "聚合查询成功")
        else:
            return error_response("统计服务暂不可用", 503)
        
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        return error_response(f"聚合查询失败: {str(e)}", 500)
