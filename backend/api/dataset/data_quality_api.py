"""数据质量管理API

提供数据质量管理相关的API接口。
"""

import sys
import os
from typing import Dict, Any
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.core.exceptions import ValidationError
from backend.services.data_quality_service import DataQualityService
from backend.repositories.dataset_repository import DatasetRepository
from backend.repositories.data_quality_repository import get_quality_repository_manager
from backend.modules.dataset.data_quality_module import DataQualityModule
from backend.modules.dataset.dataset_exceptions import (
    DatasetNotFoundError,
    QualityAssessmentNotFoundError,
    QualityAssessmentFailedError,
    DataCleaningError,
    CleaningRollbackError,
    QualityRuleError,
    QualityReportGenerationError,
    QualityMonitoringError,
)
from backend.utils.response import success_response, error_response

# 创建蓝图
data_quality_bp = Blueprint('data_quality', __name__, url_prefix='/api/v1/datasets')

# 初始化服务
repo_manager = get_quality_repository_manager()
data_quality_service = DataQualityService(
    dataset_repository=DatasetRepository(),
    assessment_repository=repo_manager.assessment_repo,
    issue_repository=repo_manager.issue_repo,
    cleaning_repository=repo_manager.cleaning_repo,
    rule_repository=repo_manager.rule_repo,
    validation_repository=repo_manager.validation_repo,
    report_repository=repo_manager.report_repo,
    monitoring_config_repository=repo_manager.monitoring_config_repo,
    alert_repository=repo_manager.alert_repo,
    quality_module=DataQualityModule()
)


# ============================================================================
# 质量评估相关API
# ============================================================================

@data_quality_bp.route('/<dataset_id>/quality/assess', methods=['POST'])
@jwt_required()
def assess_data_quality(dataset_id: str):
    """评估数据质量
    
    Args:
        dataset_id: 数据集ID
        
    Request Body:
        dimensions: 要评估的质量维度列表（可选）
        include_column_metrics: 是否包含列级指标（默认True）
        sample_size: 采样大小（可选）
        
    Returns:
        数据质量评估结果
    """
    try:
        data = request.get_json() or {}
        
        dimensions = data.get('dimensions')
        include_column_metrics = data.get('include_column_metrics', True)
        sample_size = data.get('sample_size')
        
        result = data_quality_service.assess_data_quality(
            dataset_id=dataset_id,
            dimensions=dimensions,
            include_column_metrics=include_column_metrics,
            sample_size=sample_size
        )
        
        return success_response(
            data=result,
            message="数据质量评估完成"
        )
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except QualityAssessmentFailedError as e:
        return error_response(
            message=str(e),
            code=500,
            error_type="QUALITY_ASSESSMENT_FAILED"
        ), 500
    except Exception as e:
        return error_response(
            message=f"评估数据质量时发生错误: {str(e)}",
            code=500,
            error_type="QUALITY_ASSESSMENT_ERROR"
        ), 500


@data_quality_bp.route('/<dataset_id>/quality/assessments', methods=['GET'])
@jwt_required()
def get_assessment_history(dataset_id: str):
    """获取评估历史
    
    Args:
        dataset_id: 数据集ID
        
    Query Parameters:
        limit: 返回数量限制（默认50）
        offset: 偏移量（默认0）
        
    Returns:
        评估历史列表
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        result = data_quality_service.get_assessment_history(
            dataset_id=dataset_id,
            limit=limit,
            offset=offset
        )
        
        return success_response(
            data=result,
            message="获取评估历史成功"
        )
    except Exception as e:
        return error_response(
            message=f"获取评估历史时发生错误: {str(e)}",
            code=500,
            error_type="GET_ASSESSMENT_HISTORY_ERROR"
        ), 500


@data_quality_bp.route('/quality/assessments/<assessment_id>', methods=['GET'])
@jwt_required()
def get_assessment_by_id(assessment_id: str):
    """根据ID获取评估记录
    
    Args:
        assessment_id: 评估记录ID
        
    Returns:
        评估记录详情
    """
    try:
        result = data_quality_service.get_assessment_by_id(assessment_id)
        
        if not result:
            return error_response(
                message=f"评估记录 {assessment_id} 不存在",
                code=404,
                error_type="ASSESSMENT_NOT_FOUND"
            ), 404
        
        return success_response(
            data=result,
            message="获取评估记录成功"
        )
    except Exception as e:
        return error_response(
            message=f"获取评估记录时发生错误: {str(e)}",
            code=500,
            error_type="GET_ASSESSMENT_ERROR"
        ), 500


# ============================================================================
# 问题检测相关API
# ============================================================================

@data_quality_bp.route('/<dataset_id>/quality/issues/detect', methods=['POST'])
@jwt_required()
def detect_data_issues(dataset_id: str):
    """检测数据问题
    
    Args:
        dataset_id: 数据集ID
        
    Request Body:
        issue_types: 要检测的问题类型列表（可选）
        severity_threshold: 严重程度阈值（默认"low"）
        max_issues: 最大返回问题数（默认100）
        include_samples: 是否包含示例值（默认True）
        sample_count: 示例数量（默认5）
        
    Returns:
        检测到的数据问题列表
    """
    try:
        data = request.get_json() or {}
        
        result = data_quality_service.detect_data_issues(
            dataset_id=dataset_id,
            issue_types=data.get('issue_types'),
            severity_threshold=data.get('severity_threshold', 'low'),
            max_issues=data.get('max_issues', 100),
            include_samples=data.get('include_samples', True),
            sample_count=data.get('sample_count', 5)
        )
        
        return success_response(
            data=result,
            message="数据问题检测完成"
        )
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except Exception as e:
        return error_response(
            message=f"检测数据问题时发生错误: {str(e)}",
            code=500,
            error_type="ISSUE_DETECTION_ERROR"
        ), 500


@data_quality_bp.route('/<dataset_id>/quality/issues', methods=['GET'])
@jwt_required()
def get_issue_history(dataset_id: str):
    """获取问题历史
    
    Args:
        dataset_id: 数据集ID
        
    Query Parameters:
        status: 状态过滤（可多选，逗号分隔）
        severity: 严重程度过滤（可多选，逗号分隔）
        issue_type: 问题类型过滤（可多选，逗号分隔）
        limit: 返回数量限制（默认100）
        offset: 偏移量（默认0）
        
    Returns:
        问题历史列表
    """
    try:
        status_filter = request.args.get('status')
        severity_filter = request.args.get('severity')
        issue_type_filter = request.args.get('issue_type')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        result = data_quality_service.get_issue_history(
            dataset_id=dataset_id,
            status_filter=status_filter.split(',') if status_filter else None,
            severity_filter=severity_filter.split(',') if severity_filter else None,
            issue_type_filter=issue_type_filter.split(',') if issue_type_filter else None,
            limit=limit,
            offset=offset
        )
        
        return success_response(
            data=result,
            message="获取问题历史成功"
        )
    except Exception as e:
        return error_response(
            message=f"获取问题历史时发生错误: {str(e)}",
            code=500,
            error_type="GET_ISSUE_HISTORY_ERROR"
        ), 500


@data_quality_bp.route('/quality/issues/<issue_id>/resolve', methods=['POST'])
@jwt_required()
def resolve_issue(issue_id: str):
    """解决问题
    
    Args:
        issue_id: 问题ID
        
    Request Body:
        resolution_notes: 解决备注（可选）
        
    Returns:
        更新后的问题记录
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        result = data_quality_service.resolve_issue(
            issue_id=issue_id,
            user_id=user_id,
            resolution_notes=data.get('resolution_notes')
        )
        
        if not result.get('success'):
            return error_response(
                message=result.get('error', '解决问题失败'),
                code=404,
                error_type="RESOLVE_ISSUE_FAILED"
            ), 404
        
        return success_response(
            data=result,
            message="问题已解决"
        )
    except Exception as e:
        return error_response(
            message=f"解决问题时发生错误: {str(e)}",
            code=500,
            error_type="RESOLVE_ISSUE_ERROR"
        ), 500


@data_quality_bp.route('/quality/issues/<issue_id>/ignore', methods=['POST'])
@jwt_required()
def ignore_issue(issue_id: str):
    """忽略问题
    
    Args:
        issue_id: 问题ID
        
    Request Body:
        ignore_reason: 忽略原因（可选）
        
    Returns:
        更新后的问题记录
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        result = data_quality_service.ignore_issue(
            issue_id=issue_id,
            user_id=user_id,
            ignore_reason=data.get('ignore_reason')
        )
        
        if not result.get('success'):
            return error_response(
                message=result.get('error', '忽略问题失败'),
                code=404,
                error_type="IGNORE_ISSUE_FAILED"
            ), 404
        
        return success_response(
            data=result,
            message="问题已忽略"
        )
    except Exception as e:
        return error_response(
            message=f"忽略问题时发生错误: {str(e)}",
            code=500,
            error_type="IGNORE_ISSUE_ERROR"
        ), 500


# ============================================================================
# 数据清理相关API
# ============================================================================

@data_quality_bp.route('/<dataset_id>/quality/clean', methods=['POST'])
@jwt_required()
def clean_data(dataset_id: str):
    """清理数据
    
    Args:
        dataset_id: 数据集ID
        
    Request Body:
        config: 清理配置
            - operations: 清理操作列表
            - remove_duplicates: 是否删除重复记录
            - handle_missing_values: 是否处理缺失值
            - handle_outliers: 是否处理异常值
            - missing_value_strategy: 缺失值处理策略
            - outlier_strategy: 异常值处理策略
            - preserve_original: 是否保留原始数据
        create_new_dataset: 是否创建新数据集（默认True）
        new_dataset_name: 新数据集名称（可选）
        
    Returns:
        清理结果
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        config = data.get('config', {})
        create_new_dataset = data.get('create_new_dataset', True)
        new_dataset_name = data.get('new_dataset_name')
        
        result = data_quality_service.clean_data(
            dataset_id=dataset_id,
            config=config,
            user_id=user_id,
            create_new_dataset=create_new_dataset,
            new_dataset_name=new_dataset_name
        )
        
        return success_response(
            data=result,
            message="数据清理完成"
        )
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except DataCleaningError as e:
        return error_response(
            message=str(e),
            code=500,
            error_type="DATA_CLEANING_ERROR"
        ), 500
    except Exception as e:
        return error_response(
            message=f"清理数据时发生错误: {str(e)}",
            code=500,
            error_type="DATA_CLEANING_ERROR"
        ), 500


@data_quality_bp.route('/<dataset_id>/quality/cleaning-history', methods=['GET'])
@jwt_required()
def get_cleaning_history(dataset_id: str):
    """获取清理历史
    
    Args:
        dataset_id: 数据集ID
        
    Query Parameters:
        status: 状态过滤（可多选，逗号分隔）
        limit: 返回数量限制（默认50）
        offset: 偏移量（默认0）
        
    Returns:
        清理历史列表
    """
    try:
        status_filter = request.args.get('status')
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        result = data_quality_service.get_cleaning_history(
            dataset_id=dataset_id,
            status_filter=status_filter.split(',') if status_filter else None,
            limit=limit,
            offset=offset
        )
        
        return success_response(
            data=result,
            message="获取清理历史成功"
        )
    except Exception as e:
        return error_response(
            message=f"获取清理历史时发生错误: {str(e)}",
            code=500,
            error_type="GET_CLEANING_HISTORY_ERROR"
        ), 500


@data_quality_bp.route('/quality/cleaning/<cleaning_id>', methods=['GET'])
@jwt_required()
def get_cleaning_record(cleaning_id: str):
    """获取清理记录
    
    Args:
        cleaning_id: 清理记录ID
        
    Returns:
        清理记录详情
    """
    try:
        result = data_quality_service.get_cleaning_record(cleaning_id)
        
        if not result:
            return error_response(
                message=f"清理记录 {cleaning_id} 不存在",
                code=404,
                error_type="CLEANING_RECORD_NOT_FOUND"
            ), 404
        
        return success_response(
            data=result,
            message="获取清理记录成功"
        )
    except Exception as e:
        return error_response(
            message=f"获取清理记录时发生错误: {str(e)}",
            code=500,
            error_type="GET_CLEANING_RECORD_ERROR"
        ), 500


@data_quality_bp.route('/quality/cleaning/<cleaning_id>/rollback', methods=['POST'])
@jwt_required()
def rollback_cleaning(cleaning_id: str):
    """回滚清理操作
    
    Args:
        cleaning_id: 清理记录ID
        
    Returns:
        回滚结果
    """
    try:
        user_id = get_jwt_identity()
        
        result = data_quality_service.rollback_cleaning(
            cleaning_id=cleaning_id,
            user_id=user_id
        )
        
        return success_response(
            data=result,
            message="清理操作已回滚"
        )
    except CleaningRollbackError as e:
        return error_response(
            message=str(e),
            code=400,
            error_type="ROLLBACK_ERROR"
        ), 400
    except Exception as e:
        return error_response(
            message=f"回滚清理操作时发生错误: {str(e)}",
            code=500,
            error_type="ROLLBACK_CLEANING_ERROR"
        ), 500


# ============================================================================
# 质量规则相关API
# ============================================================================

@data_quality_bp.route('/quality/rules', methods=['POST'])
@jwt_required()
def create_quality_rule():
    """创建质量规则
    
    Request Body:
        name: 规则名称
        description: 规则描述
        rule_type: 规则类型
        target_column: 目标列
        condition: 规则条件
        parameters: 规则参数
        severity: 违反规则的严重程度
        enabled: 是否启用
        dataset_ids: 应用此规则的数据集ID列表
        
    Returns:
        创建的规则
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        # 从请求中获取租户ID（如果有）
        tenant_id = data.get('tenant_id')
        
        result = data_quality_service.create_quality_rule(
            rule=data,
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        return success_response(
            data=result,
            message="质量规则创建成功"
        ), 201
    except Exception as e:
        return error_response(
            message=f"创建质量规则时发生错误: {str(e)}",
            code=500,
            error_type="CREATE_RULE_ERROR"
        ), 500


@data_quality_bp.route('/quality/rules', methods=['GET'])
@jwt_required()
def get_quality_rules():
    """获取质量规则列表
    
    Query Parameters:
        enabled_only: 是否只返回启用的规则（默认False）
        limit: 返回数量限制（默认100）
        offset: 偏移量（默认0）
        
    Returns:
        规则列表
    """
    try:
        user_id = get_jwt_identity()
        enabled_only = request.args.get('enabled_only', 'false').lower() == 'true'
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        tenant_id = request.args.get('tenant_id')
        
        result = data_quality_service.get_quality_rules(
            user_id=user_id,
            tenant_id=tenant_id,
            enabled_only=enabled_only,
            limit=limit,
            offset=offset
        )
        
        return success_response(
            data=result,
            message="获取质量规则成功"
        )
    except Exception as e:
        return error_response(
            message=f"获取质量规则时发生错误: {str(e)}",
            code=500,
            error_type="GET_RULES_ERROR"
        ), 500


@data_quality_bp.route('/quality/rules/<rule_id>', methods=['PUT'])
@jwt_required()
def update_quality_rule(rule_id: str):
    """更新质量规则
    
    Args:
        rule_id: 规则ID
        
    Request Body:
        name: 规则名称
        description: 规则描述
        condition: 规则条件
        parameters: 规则参数
        severity: 严重程度
        enabled: 是否启用
        
    Returns:
        更新后的规则
    """
    try:
        data = request.get_json() or {}
        
        result = data_quality_service.update_quality_rule(
            rule_id=rule_id,
            updates=data
        )
        
        return success_response(
            data=result,
            message="质量规则更新成功"
        )
    except QualityRuleError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="RULE_NOT_FOUND"
        ), 404
    except Exception as e:
        return error_response(
            message=f"更新质量规则时发生错误: {str(e)}",
            code=500,
            error_type="UPDATE_RULE_ERROR"
        ), 500


@data_quality_bp.route('/quality/rules/<rule_id>', methods=['DELETE'])
@jwt_required()
def delete_quality_rule(rule_id: str):
    """删除质量规则
    
    Args:
        rule_id: 规则ID
        
    Returns:
        删除结果
    """
    try:
        result = data_quality_service.delete_quality_rule(rule_id)
        
        if not result:
            return error_response(
                message=f"规则 {rule_id} 不存在",
                code=404,
                error_type="RULE_NOT_FOUND"
            ), 404
        
        return success_response(
            data={'deleted': True, 'rule_id': rule_id},
            message="质量规则删除成功"
        )
    except Exception as e:
        return error_response(
            message=f"删除质量规则时发生错误: {str(e)}",
            code=500,
            error_type="DELETE_RULE_ERROR"
        ), 500


@data_quality_bp.route('/<dataset_id>/quality/validate', methods=['POST'])
@jwt_required()
def validate_rules(dataset_id: str):
    """验证质量规则
    
    Args:
        dataset_id: 数据集ID
        
    Request Body:
        rules: 规则列表
        stop_on_failure: 遇到失败是否停止（默认False）
        
    Returns:
        验证结果
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        rules = data.get('rules', [])
        stop_on_failure = data.get('stop_on_failure', False)
        
        if not rules:
            return error_response(
                message="规则列表不能为空",
                code=400,
                error_type="VALIDATION_ERROR"
            ), 400
        
        result = data_quality_service.validate_rules(
            dataset_id=dataset_id,
            rules=rules,
            user_id=user_id,
            stop_on_failure=stop_on_failure
        )
        
        return success_response(
            data=result,
            message="规则验证完成"
        )
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except Exception as e:
        return error_response(
            message=f"验证规则时发生错误: {str(e)}",
            code=500,
            error_type="VALIDATE_RULES_ERROR"
        ), 500


@data_quality_bp.route('/<dataset_id>/quality/validation-history', methods=['GET'])
@jwt_required()
def get_validation_history(dataset_id: str):
    """获取验证历史
    
    Args:
        dataset_id: 数据集ID
        
    Query Parameters:
        limit: 返回数量限制（默认50）
        offset: 偏移量（默认0）
        
    Returns:
        验证历史列表
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        result = data_quality_service.get_validation_history(
            dataset_id=dataset_id,
            limit=limit,
            offset=offset
        )
        
        return success_response(
            data=result,
            message="获取验证历史成功"
        )
    except Exception as e:
        return error_response(
            message=f"获取验证历史时发生错误: {str(e)}",
            code=500,
            error_type="GET_VALIDATION_HISTORY_ERROR"
        ), 500


# ============================================================================
# 质量报告相关API
# ============================================================================

@data_quality_bp.route('/<dataset_id>/quality/report', methods=['POST'])
@jwt_required()
def generate_quality_report(dataset_id: str):
    """生成数据质量报告
    
    Args:
        dataset_id: 数据集ID
        
    Request Body:
        include_trends: 是否包含趋势分析（默认True）
        trend_period_days: 趋势分析周期（默认30天）
        include_recommendations: 是否包含改进建议（默认True）
        
    Returns:
        数据质量报告
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        result = data_quality_service.generate_quality_report(
            dataset_id=dataset_id,
            user_id=user_id,
            include_trends=data.get('include_trends', True),
            trend_period_days=data.get('trend_period_days', 30),
            include_recommendations=data.get('include_recommendations', True)
        )
        
        return success_response(
            data=result,
            message="数据质量报告生成完成"
        )
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except QualityReportGenerationError as e:
        return error_response(
            message=str(e),
            code=500,
            error_type="REPORT_GENERATION_ERROR"
        ), 500
    except Exception as e:
        return error_response(
            message=f"生成数据质量报告时发生错误: {str(e)}",
            code=500,
            error_type="REPORT_GENERATION_ERROR"
        ), 500


@data_quality_bp.route('/<dataset_id>/quality/reports', methods=['GET'])
@jwt_required()
def get_report_history(dataset_id: str):
    """获取报告历史
    
    Args:
        dataset_id: 数据集ID
        
    Query Parameters:
        limit: 返回数量限制（默认50）
        offset: 偏移量（默认0）
        
    Returns:
        报告历史列表
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        result = data_quality_service.get_report_history(
            dataset_id=dataset_id,
            limit=limit,
            offset=offset
        )
        
        return success_response(
            data=result,
            message="获取报告历史成功"
        )
    except Exception as e:
        return error_response(
            message=f"获取报告历史时发生错误: {str(e)}",
            code=500,
            error_type="GET_REPORT_HISTORY_ERROR"
        ), 500


@data_quality_bp.route('/quality/reports/<report_id>', methods=['GET'])
@jwt_required()
def get_report_by_id(report_id: str):
    """根据ID获取报告
    
    Args:
        report_id: 报告ID
        
    Returns:
        报告详情
    """
    try:
        result = data_quality_service.get_report_by_id(report_id)
        
        if not result:
            return error_response(
                message=f"报告 {report_id} 不存在",
                code=404,
                error_type="REPORT_NOT_FOUND"
            ), 404
        
        return success_response(
            data=result,
            message="获取报告成功"
        )
    except Exception as e:
        return error_response(
            message=f"获取报告时发生错误: {str(e)}",
            code=500,
            error_type="GET_REPORT_ERROR"
        ), 500


# ============================================================================
# 质量监控相关API
# ============================================================================

@data_quality_bp.route('/<dataset_id>/quality/monitoring', methods=['POST'])
@jwt_required()
def setup_quality_monitoring(dataset_id: str):
    """设置质量监控
    
    Args:
        dataset_id: 数据集ID
        
    Request Body:
        enabled: 是否启用监控
        thresholds: 质量阈值列表
        check_interval_minutes: 检查间隔（分钟）
        alert_channels: 告警渠道列表
        
    Returns:
        监控配置
    """
    try:
        user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        result = data_quality_service.setup_quality_monitoring(
            dataset_id=dataset_id,
            config=data,
            user_id=user_id
        )
        
        return success_response(
            data=result,
            message="质量监控设置成功"
        )
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except Exception as e:
        return error_response(
            message=f"设置质量监控时发生错误: {str(e)}",
            code=500,
            error_type="SETUP_MONITORING_ERROR"
        ), 500


@data_quality_bp.route('/<dataset_id>/quality/monitoring', methods=['GET'])
@jwt_required()
def get_monitoring_status(dataset_id: str):
    """获取监控状态
    
    Args:
        dataset_id: 数据集ID
        
    Returns:
        监控状态
    """
    try:
        result = data_quality_service.get_monitoring_status(dataset_id)
        
        return success_response(
            data=result,
            message="获取监控状态成功"
        )
    except Exception as e:
        return error_response(
            message=f"获取监控状态时发生错误: {str(e)}",
            code=500,
            error_type="GET_MONITORING_STATUS_ERROR"
        ), 500


@data_quality_bp.route('/<dataset_id>/quality/monitoring', methods=['PUT'])
@jwt_required()
def update_monitoring_config(dataset_id: str):
    """更新监控配置
    
    Args:
        dataset_id: 数据集ID
        
    Request Body:
        enabled: 是否启用监控
        thresholds: 质量阈值列表
        check_interval_minutes: 检查间隔（分钟）
        alert_channels: 告警渠道列表
        
    Returns:
        更新后的配置
    """
    try:
        data = request.get_json() or {}
        
        result = data_quality_service.update_monitoring_config(
            dataset_id=dataset_id,
            updates=data
        )
        
        return success_response(
            data=result,
            message="监控配置更新成功"
        )
    except QualityMonitoringError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="MONITORING_CONFIG_NOT_FOUND"
        ), 404
    except Exception as e:
        return error_response(
            message=f"更新监控配置时发生错误: {str(e)}",
            code=500,
            error_type="UPDATE_MONITORING_ERROR"
        ), 500


@data_quality_bp.route('/<dataset_id>/quality/monitoring', methods=['DELETE'])
@jwt_required()
def disable_monitoring(dataset_id: str):
    """禁用监控
    
    Args:
        dataset_id: 数据集ID
        
    Returns:
        禁用结果
    """
    try:
        result = data_quality_service.disable_monitoring(dataset_id)
        
        return success_response(
            data={'disabled': result, 'dataset_id': dataset_id},
            message="监控已禁用" if result else "监控配置不存在"
        )
    except Exception as e:
        return error_response(
            message=f"禁用监控时发生错误: {str(e)}",
            code=500,
            error_type="DISABLE_MONITORING_ERROR"
        ), 500


@data_quality_bp.route('/<dataset_id>/quality/alerts', methods=['GET'])
@jwt_required()
def get_quality_alerts(dataset_id: str):
    """获取质量告警
    
    Args:
        dataset_id: 数据集ID
        
    Query Parameters:
        acknowledged: 是否已确认过滤（true/false）
        limit: 返回数量限制（默认100）
        offset: 偏移量（默认0）
        
    Returns:
        告警列表
    """
    try:
        acknowledged_param = request.args.get('acknowledged')
        if acknowledged_param is not None:
            acknowledged = acknowledged_param.lower() == 'true'
        else:
            acknowledged = None
        
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        result = data_quality_service.get_quality_alerts(
            dataset_id=dataset_id,
            acknowledged=acknowledged,
            limit=limit,
            offset=offset
        )
        
        return success_response(
            data=result,
            message="获取质量告警成功"
        )
    except Exception as e:
        return error_response(
            message=f"获取质量告警时发生错误: {str(e)}",
            code=500,
            error_type="GET_ALERTS_ERROR"
        ), 500


@data_quality_bp.route('/quality/alerts/<alert_id>/acknowledge', methods=['POST'])
@jwt_required()
def acknowledge_alert(alert_id: str):
    """确认告警
    
    Args:
        alert_id: 告警ID
        
    Returns:
        更新后的告警
    """
    try:
        user_id = get_jwt_identity()
        
        result = data_quality_service.acknowledge_alert(
            alert_id=alert_id,
            user_id=user_id
        )
        
        if not result.get('success'):
            return error_response(
                message=result.get('error', '确认告警失败'),
                code=404,
                error_type="ALERT_NOT_FOUND"
            ), 404
        
        return success_response(
            data=result,
            message="告警已确认"
        )
    except Exception as e:
        return error_response(
            message=f"确认告警时发生错误: {str(e)}",
            code=500,
            error_type="ACKNOWLEDGE_ALERT_ERROR"
        ), 500


# ============================================================================
# 统计相关API
# ============================================================================

@data_quality_bp.route('/<dataset_id>/quality/stats', methods=['GET'])
@jwt_required()
def get_quality_stats(dataset_id: str):
    """获取质量统计
    
    Args:
        dataset_id: 数据集ID
        
    Query Parameters:
        period_days: 统计周期（默认30天）
        
    Returns:
        质量统计信息
    """
    try:
        period_days = request.args.get('period_days', 30, type=int)
        
        result = data_quality_service.get_quality_stats(
            dataset_id=dataset_id,
            period_days=period_days
        )
        
        return success_response(
            data=result,
            message="获取质量统计成功"
        )
    except Exception as e:
        return error_response(
            message=f"获取质量统计时发生错误: {str(e)}",
            code=500,
            error_type="GET_STATS_ERROR"
        ), 500


@data_quality_bp.route('/<dataset_id>/quality/trends', methods=['GET'])
@jwt_required()
def get_quality_trends(dataset_id: str):
    """获取质量趋势
    
    Args:
        dataset_id: 数据集ID
        
    Query Parameters:
        start_date: 开始日期（可选）
        end_date: 结束日期（可选）
        
    Returns:
        质量趋势数据
    """
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        result = data_quality_service.get_quality_trends(
            dataset_id=dataset_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return success_response(
            data=result,
            message="获取质量趋势成功"
        )
    except Exception as e:
        return error_response(
            message=f"获取质量趋势时发生错误: {str(e)}",
            code=500,
            error_type="GET_TRENDS_ERROR"
        ), 500
