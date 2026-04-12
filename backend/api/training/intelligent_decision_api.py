"""
智能化决策API
提供AI驱动的自动化和知识图谱驱动的REST API接口。
支持租户维度的数据隔离和持久化，通过Service层访问Repository层。
"""
import sys
import os
import logging
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.utils.response import success_response, error_response

logger = logging.getLogger(__name__)

# 创建蓝图
intelligent_decision_bp = Blueprint('intelligent_decision', __name__, url_prefix='/api/v1/training/intelligent')


def _get_tenant_id(user_id: str) -> str:
    """获取用户的租户ID"""
    tenant_id = request.headers.get('X-Tenant-ID')
    if not tenant_id:
        tenant_id = f"tenant_{user_id}"
    return tenant_id


def _get_service():
    """获取智能决策服务（延迟加载）"""
    try:
        from backend.services.intelligent_decision_service import (
            get_intelligent_decision_service,
            DecisionScenario,
            DecisionContext,
            AdaptiveConfiguration
        )
        return get_intelligent_decision_service(use_memory_storage=True)
    except ImportError as e:
        logger.error(f"Failed to import IntelligentDecisionService: {e}")
        return None


def _get_decision_scenario(scenario_str: str):
    """获取决策场景枚举"""
    try:
        from backend.services.intelligent_decision_service import DecisionScenario
        return DecisionScenario(scenario_str)
    except (ValueError, ImportError):
        return None


# ============================================================================
# 智能决策接口
# ============================================================================

@intelligent_decision_bp.route('/decisions', methods=['POST'])
@jwt_required()
def make_intelligent_decision():
    """
    智能决策
    
    Request Body:
        {
            "scenario": "string" (data_preprocessing, model_architecture, etc.),
            "inputs": {},
            "constraints": {} (optional),
            "history": [] (optional)
        }
        
    Returns:
        {
            "decision": {
                "decision_id": "string",
                "scenario": "string",
                "recommended_action": {},
                "confidence": float,
                "reasoning": "string",
                "alternatives": [],
                "execution_plan": {}
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
        
        scenario_str = data.get('scenario')
        if not scenario_str:
            return error_response("决策场景不能为空", 400)
        
        scenario = _get_decision_scenario(scenario_str)
        if not scenario:
            return error_response(f"不支持的决策场景: {scenario_str}", 400)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        from backend.services.intelligent_decision_service import DecisionContext
        
        context = DecisionContext(
            scenario=scenario,
            inputs=data.get('inputs', {}),
            constraints=data.get('constraints', {}),
            history=data.get('history', [])
        )
        
        result = service.make_intelligent_decision(context, tenant_id, user_id)
        
        logger.info(f"Made intelligent decision: {result.decision_id} for user {user_id}")
        
        return success_response({
            "decision": {
                "decision_id": result.decision_id,
                "scenario": result.scenario,
                "recommended_action": result.recommended_action,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "alternatives": result.alternatives,
                "execution_plan": result.execution_plan,
                "metadata": result.metadata
            }
        }, "智能决策成功")
        
    except Exception as e:
        logger.error(f"Failed to make intelligent decision: {e}")
        return error_response(f"智能决策失败: {str(e)}", 500)


@intelligent_decision_bp.route('/decisions', methods=['GET'])
@jwt_required()
def list_decisions():
    """
    获取决策历史列表
    
    Query Parameters:
        - scenario: 场景过滤
        - page: 页码 (默认: 1)
        - limit: 每页数量 (默认: 20)
    
    Returns:
        {
            "decisions": [...],
            "pagination": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        scenario = request.args.get('scenario')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        result = service.list_decisions(
            tenant_id=tenant_id,
            user_id=user_id,
            scenario=scenario,
            page=page,
            page_size=limit
        )
        
        return success_response({
            "decisions": result.get('items', []),
            "pagination": {
                "page": result.get('page', page),
                "limit": result.get('page_size', limit),
                "total": result.get('total', 0),
                "pages": result.get('total_pages', 0)
            }
        }, "获取决策历史成功")
        
    except Exception as e:
        logger.error(f"Failed to list decisions: {e}")
        return error_response(f"获取决策历史失败: {str(e)}", 500)


@intelligent_decision_bp.route('/decisions/<decision_id>', methods=['GET'])
@jwt_required()
def get_decision(decision_id):
    """
    获取决策详情
    
    Returns:
        {
            "decision": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        decision = service.get_decision(decision_id, tenant_id)
        if not decision:
            return error_response("决策不存在", 404)
        
        return success_response({
            "decision": decision
        }, "获取决策详情成功")
        
    except Exception as e:
        logger.error(f"Failed to get decision: {e}")
        return error_response(f"获取决策详情失败: {str(e)}", 500)


@intelligent_decision_bp.route('/decisions/<decision_id>/feedback', methods=['POST'])
@jwt_required()
def provide_decision_feedback(decision_id):
    """
    提供决策反馈
    
    Request Body:
        {
            "score": float (0-1),
            "comment": "string" (optional)
        }
    
    Returns:
        {
            "success": boolean
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
        
        score = data.get('score')
        if score is None or not (0 <= score <= 1):
            return error_response("评分必须在0-1之间", 400)
        
        comment = data.get('comment')
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        success = service.provide_feedback(decision_id, tenant_id, user_id, score, comment)
        
        if not success:
            return error_response("决策不存在或无权操作", 404)
        
        return success_response({
            "success": True
        }, "反馈提交成功")
        
    except Exception as e:
        logger.error(f"Failed to provide feedback: {e}")
        return error_response(f"提交反馈失败: {str(e)}", 500)


@intelligent_decision_bp.route('/decisions/statistics', methods=['GET'])
@jwt_required()
def get_decision_statistics():
    """
    获取决策统计信息
    
    Returns:
        {
            "statistics": {
                "total": int,
                "by_scenario": {},
                "by_algorithm": {},
                "avg_confidence": float,
                "avg_feedback_score": float
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        stats = service.get_decision_statistics(tenant_id, user_id)
        
        return success_response({
            "statistics": stats
        }, "获取统计信息成功")
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return error_response(f"获取统计信息失败: {str(e)}", 500)


# ============================================================================
# 自适应优化接口
# ============================================================================

@intelligent_decision_bp.route('/optimization/adaptive', methods=['POST'])
@jwt_required()
def adaptive_optimization():
    """
    自适应优化
    
    Request Body:
        {
            "parameter_name": "string",
            "current_value": any,
            "adjustment_strategy": "string",
            "adjustment_range": {} (optional),
            "monitoring_metrics": [] (optional)
        }
        
    Returns:
        {
            "optimization": {
                "optimization_id": "string",
                "parameter_name": "string",
                "original_value": any,
                "optimized_value": any,
                "improvement_metric": "string",
                "improvement_value": float,
                "adjustment_reason": "string"
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
        
        parameter_name = data.get('parameter_name')
        current_value = data.get('current_value')
        adjustment_strategy = data.get('adjustment_strategy')
        
        if not parameter_name or current_value is None or not adjustment_strategy:
            return error_response("缺少必要参数", 400)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        from backend.services.intelligent_decision_service import AdaptiveConfiguration
        
        config = AdaptiveConfiguration(
            parameter_name=parameter_name,
            current_value=current_value,
            adjustment_strategy=adjustment_strategy,
            adjustment_range=data.get('adjustment_range'),
            monitoring_metrics=data.get('monitoring_metrics', [])
        )
        
        result = service.adaptive_optimization(config, tenant_id, user_id)
        
        logger.info(f"Adaptive optimization completed: {result.optimization_id}")
        
        return success_response({
            "optimization": {
                "optimization_id": result.optimization_id,
                "parameter_name": result.parameter_name,
                "original_value": result.original_value,
                "optimized_value": result.optimized_value,
                "improvement_metric": result.improvement_metric,
                "improvement_value": result.improvement_value,
                "adjustment_reason": result.adjustment_reason,
                "timestamp": result.timestamp.isoformat()
            }
        }, "自适应优化成功")
        
    except Exception as e:
        logger.error(f"Failed to perform adaptive optimization: {e}")
        return error_response(f"自适应优化失败: {str(e)}", 500)


@intelligent_decision_bp.route('/optimization/history', methods=['GET'])
@jwt_required()
def get_optimization_history():
    """
    获取优化历史
    
    Query Parameters:
        - parameter_name: 参数名过滤
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
        
        parameter_name = request.args.get('parameter_name')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        result = service.list_adaptive_optimizations(
            tenant_id=tenant_id,
            user_id=user_id,
            parameter_name=parameter_name,
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
        }, "获取优化历史成功")
        
    except Exception as e:
        logger.error(f"Failed to get optimization history: {e}")
        return error_response(f"获取优化历史失败: {str(e)}", 500)


@intelligent_decision_bp.route('/optimization/<optimization_id>/apply', methods=['POST'])
@jwt_required()
def apply_optimization(optimization_id):
    """
    应用优化结果
    
    Returns:
        {
            "success": boolean
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        success = service.apply_optimization(optimization_id, tenant_id)
        
        if not success:
            return error_response("优化记录不存在", 404)
        
        return success_response({
            "success": True
        }, "优化已应用")
        
    except Exception as e:
        logger.error(f"Failed to apply optimization: {e}")
        return error_response(f"应用优化失败: {str(e)}", 500)


# ============================================================================
# 知识库接口
# ============================================================================

@intelligent_decision_bp.route('/knowledge/graph', methods=['POST'])
@jwt_required()
def get_knowledge_graph():
    """
    获取知识图谱
    
    Request Body:
        {
            "query": "string"
        }
        
    Returns:
        {
            "knowledge_graph": {
                "query": "string",
                "entities": [],
                "relationships": [],
                "recommendations": []
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
        
        query = data.get('query')
        if not query:
            return error_response("查询语句不能为空", 400)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        knowledge_graph = service.get_knowledge_graph(query, tenant_id)
        
        return success_response({
            "knowledge_graph": knowledge_graph
        }, "获取知识图谱成功")
        
    except Exception as e:
        logger.error(f"Failed to get knowledge graph: {e}")
        return error_response(f"获取知识图谱失败: {str(e)}", 500)


@intelligent_decision_bp.route('/knowledge', methods=['GET'])
@jwt_required()
def list_knowledge():
    """
    获取知识列表
    
    Query Parameters:
        - knowledge_type: 类型过滤
        - category: 类别过滤
        - page: 页码 (默认: 1)
        - limit: 每页数量 (默认: 20)
    
    Returns:
        {
            "knowledge_list": [...],
            "pagination": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        knowledge_type = request.args.get('knowledge_type')
        category = request.args.get('category')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        result = service.list_knowledge(
            tenant_id=tenant_id,
            knowledge_type=knowledge_type,
            category=category,
            page=page,
            page_size=limit
        )
        
        return success_response({
            "knowledge_list": result.get('items', []),
            "pagination": {
                "page": result.get('page', page),
                "limit": result.get('page_size', limit),
                "total": result.get('total', 0),
                "pages": result.get('total_pages', 0)
            }
        }, "获取知识列表成功")
        
    except Exception as e:
        logger.error(f"Failed to list knowledge: {e}")
        return error_response(f"获取知识列表失败: {str(e)}", 500)


@intelligent_decision_bp.route('/knowledge', methods=['POST'])
@jwt_required()
def create_knowledge():
    """
    创建知识
    
    Request Body:
        {
            "knowledge_type": "string",
            "category": "string" (optional),
            "title": "string",
            "content": {},
            "related_entities": [] (optional),
            "relationships": [] (optional),
            "is_public": boolean (optional, default: false)
        }
    
    Returns:
        {
            "knowledge": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
        
        if not data.get('title'):
            return error_response("知识标题不能为空", 400)
        if not data.get('content'):
            return error_response("知识内容不能为空", 400)
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        knowledge = service.create_knowledge(tenant_id, user_id, data)
        
        logger.info(f"Created knowledge: {knowledge.get('id')}")
        
        return success_response({
            "knowledge": knowledge
        }, "创建知识成功", 201)
        
    except Exception as e:
        logger.error(f"Failed to create knowledge: {e}")
        return error_response(f"创建知识失败: {str(e)}", 500)


@intelligent_decision_bp.route('/knowledge/base', methods=['POST'])
@jwt_required()
def update_knowledge_base():
    """
    更新知识库（兼容旧接口）
    
    Request Body:
        {
            "knowledge": {...}
        }
        
    Returns:
        {
            "success": boolean
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
        
        knowledge = data.get('knowledge', {})
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        success = service.update_knowledge_base(knowledge, tenant_id, user_id)
        
        if success:
            return success_response({
                "success": True
            }, "知识库更新成功")
        else:
            return error_response("知识库更新失败", 500)
        
    except Exception as e:
        logger.error(f"Failed to update knowledge base: {e}")
        return error_response(f"更新知识库失败: {str(e)}", 500)


# ============================================================================
# 经验积累接口
# ============================================================================

@intelligent_decision_bp.route('/experience/accumulate', methods=['POST'])
@jwt_required()
def accumulate_experience():
    """
    积累经验
    
    Request Body:
        {
            "type": "string",
            "scenario": "string" (optional),
            "context": {},
            "action": {},
            "result": {},
            "reward": float (optional),
            "decision_id": "string" (optional),
            "lessons_learned": "string" (optional)
        }
        
    Returns:
        {
            "success": boolean
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
        
        experience_data = data.get('experience_data') or data
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        success = service.experience_accumulation(experience_data, tenant_id, user_id)
        
        if success:
            return success_response({
                "success": True
            }, "经验积累成功")
        else:
            return error_response("经验积累失败", 500)
        
    except Exception as e:
        logger.error(f"Failed to accumulate experience: {e}")
        return error_response(f"积累经验失败: {str(e)}", 500)


@intelligent_decision_bp.route('/experience', methods=['GET'])
@jwt_required()
def list_experiences():
    """
    获取经验列表
    
    Query Parameters:
        - scenario: 场景过滤
        - is_positive: 是否正面经验
        - page: 页码 (默认: 1)
        - limit: 每页数量 (默认: 20)
    
    Returns:
        {
            "experiences": [...],
            "pagination": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        scenario = request.args.get('scenario')
        is_positive_str = request.args.get('is_positive')
        is_positive = None
        if is_positive_str is not None:
            is_positive = is_positive_str.lower() == 'true'
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        result = service.list_experiences(
            tenant_id=tenant_id,
            user_id=user_id,
            scenario=scenario,
            is_positive=is_positive,
            page=page,
            page_size=limit
        )
        
        return success_response({
            "experiences": result.get('items', []),
            "pagination": {
                "page": result.get('page', page),
                "limit": result.get('page_size', limit),
                "total": result.get('total', 0),
                "pages": result.get('total_pages', 0)
            }
        }, "获取经验列表成功")
        
    except Exception as e:
        logger.error(f"Failed to list experiences: {e}")
        return error_response(f"获取经验列表失败: {str(e)}", 500)


# ============================================================================
# 辅助接口
# ============================================================================

@intelligent_decision_bp.route('/scenarios', methods=['GET'])
@jwt_required()
def get_decision_scenarios():
    """
    获取支持的决策场景和算法
    
    Returns:
        {
            "scenarios": [...],
            "algorithms": [...]
        }
    """
    try:
        from backend.services.intelligent_decision_service import DecisionScenario, DecisionAlgorithm
        
        scenarios = [scenario.value for scenario in DecisionScenario]
        algorithms = [algo.value for algo in DecisionAlgorithm]
        
        return success_response({
            "scenarios": scenarios,
            "algorithms": algorithms
        }, "获取决策场景成功")
        
    except Exception as e:
        logger.error(f"Failed to get scenarios: {e}")
        return error_response(f"获取决策场景失败: {str(e)}", 500)


@intelligent_decision_bp.route('/decisions/history', methods=['GET'])
@jwt_required()
def get_decision_history():
    """
    获取决策历史（兼容旧接口）
    
    Query Parameters:
        - limit: 限制返回记录数
        - scenario: 场景过滤
        
    Returns:
        {
            "history": [...],
            "count": int
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        limit = request.args.get('limit', 50, type=int)
        scenario = request.args.get('scenario')
        
        service = _get_service()
        if not service:
            return error_response("服务不可用", 503)
        
        # 优先使用数据库中的决策历史
        result = service.list_decisions(
            tenant_id=tenant_id,
            user_id=user_id,
            scenario=scenario,
            page=1,
            page_size=limit
        )
        
        history = result.get('items', [])
        
        # 如果数据库没有数据，使用内存中的历史
        if not history and hasattr(service, 'decision_history'):
            memory_history = service.decision_history or []
            if scenario:
                history = [h for h in memory_history if h.get('scenario') == scenario]
            else:
                history = memory_history
            history = history[-limit:] if limit and limit > 0 else history
        
        return success_response({
            "history": history,
            "count": len(history)
        }, "获取决策历史成功")
        
    except Exception as e:
        logger.error(f"Failed to get decision history: {e}")
        return error_response(f"获取决策历史失败: {str(e)}", 500)
