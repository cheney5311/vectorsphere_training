"""
模型部署API
提供模型部署策略和服务化封装的REST API接口
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from typing import Dict, List, Any
import logging

# 添加缺失的导入
from backend.core.validation import validate_json_schema

from backend.services.model_deployment_service import (
    ModelDeploymentService,
    DeploymentMode,
    ReleaseStrategy,
    DeploymentConfig,
    ServiceConfig
)

# 创建蓝图
model_deployment_bp = Blueprint('model_deployment', __name__, url_prefix='/api/v1/training/model')
logger = logging.getLogger(__name__)

# 初始化服务
deployment_service = ModelDeploymentService()


@model_deployment_bp.route('/deploy', methods=['POST'])
@jwt_required()
@validate_json_schema('backend/api/schemas/model_deploy_request.json')
def deploy_model():
    """
    部署模型
    
    Args:
        model_id: 模型ID
        
    Request Body:
        mode: 部署模式
        release_strategy: 发布策略
        replicas: 副本数
        resources: 资源配置
        environment: 环境变量
        autoscaling: 是否自动扩缩容
        health_check: 是否启用健康检查
        monitoring: 是否启用监控
        
    Returns:
        部署结果
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        
        # 获取请求数据
        data = request.get_json()
        model_id = data.get('model_id')
        mode_str = data.get('mode', 'online')
        release_strategy_str = data.get('release_strategy', 'rolling')
        replicas = data.get('replicas', 1)
        resources = data.get('resources', {})
        environment = data.get('environment', {})
        autoscaling = data.get('autoscaling', False)
        canary_percent = int(data.get('canary_percent', 10))
        rolling_step = int(data.get('rolling_step', 1))
        
        logger.info(f"用户 {user_id} 请求部署模型 {model_id}")
        
        if not model_id:
            return jsonify({
                'success': False,
                'error': '缺少模型ID'
            }), 400
        health_check = data.get('health_check', True)
        monitoring = data.get('monitoring', True)
        ab_percent = data.get('ab_percent', 50)  # 默认值为50
        blue_green_switch_policy = data.get('blue_green_switch_policy', 'manual')  # 默认值为manual
        
        # 验证部署模式
        try:
            mode = DeploymentMode(mode_str)
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'不支持的部署模式: {mode_str}'
            }), 400
        
        # 验证发布策略
        try:
            release_strategy = ReleaseStrategy(release_strategy_str)
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'不支持的发布策略: {release_strategy_str}'
            }), 400
        
        # 创建配置
        config = DeploymentConfig(
            mode=mode,
            release_strategy=release_strategy,
            replicas=replicas,
            resources=resources,
            environment=environment,
            autoscaling=autoscaling,
            health_check=health_check,
            monitoring=monitoring,
            canary_percent=canary_percent,
            rolling_step=rolling_step,
            ab_percent=ab_percent,
            blue_green_switch_policy=blue_green_switch_policy
        )
        
        # 获取租户ID
        tenant_id = request.headers.get('X-Tenant-ID') or data.get('tenant_id')
        
        # 执行模型部署（传递租户和用户信息）
        result = deployment_service.deploy_model(
            model_id=model_id,
            config=config,
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        # 返回结果
        return jsonify({
            'success': True,
            'data': {
                'model_id': result.model_id,
                'deployment_id': result.deployment_id,
                'mode': result.mode,
                'endpoint_url': result.endpoint_url,
                'status': result.status,
                'replicas': result.replicas,
                'resources': result.resources,
                'deployment_time': result.deployment_time,
                'metadata': result.metadata
            }
        }), 200
        
    except Exception as e:
        logger.error(f"模型部署失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'模型部署失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/deployments', methods=['GET'])
@jwt_required()
def get_deployments():
    """
    获取部署状态列表（支持租户隔离）
    
    Query Parameters:
        model_id: 模型ID (可选)
        status: 部署状态 (可选，pending/deploying/running/stopped/failed/rolled_back)
        mode: 部署模式 (可选，online/batch/edge/hybrid)
        limit: 返回数量限制，默认100
        offset: 偏移量，默认0
        
    Returns:
        部署状态列表
    """
    try:
        # 获取当前用户信息
        user_id = get_jwt_identity()
        
        # 从请求上下文获取租户ID（假设JWT中包含或从请求头获取）
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        # 获取查询参数
        model_id = request.args.get('model_id')
        status = request.args.get('status')
        mode = request.args.get('mode')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        logger.info(f"用户 {user_id} 请求获取部署状态，tenant_id: {tenant_id}, model_id: {model_id}, status: {status}")
        
        # 调用服务层获取部署列表
        result = deployment_service.list_deployments(
            model_id=model_id,
            tenant_id=tenant_id,
            status=status,
            mode=mode,
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"获取部署状态失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取部署状态失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/models/<model_id>/service', methods=['POST'])
@jwt_required()
@validate_json_schema('backend/api/schemas/model_service_request.json')
def service_model(model_id: str):
    """
    服务化封装模型
    
    Args:
        model_id: 模型ID
        
    Request Body:
        api_type: API类型
        service_mesh: 是否使用服务网格
        load_balancing: 是否启用负载均衡
        circuit_breaker: 是否启用熔断降级
        rate_limiting: 是否启用限流控制
        timeout: 超时时间
        max_concurrent_requests: 最大并发请求数
        
    Returns:
        服务化结果
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        logger.info(f"用户 {user_id} 请求服务化封装模型 {model_id}")
        
        # 获取请求数据
        data = request.get_json()
        config = ServiceConfig(
            api_type=data.get('api_type', 'rest'),
            service_mesh=data.get('service_mesh', False),
            load_balancing=data.get('load_balancing', True),
            circuit_breaker=data.get('circuit_breaker', True),
            rate_limiting=data.get('rate_limiting', True),
            timeout=data.get('timeout', 30),
            max_concurrent_requests=data.get('max_concurrent_requests', 100)
        )
        
        # 获取租户ID
        tenant_id = request.headers.get('X-Tenant-ID') or data.get('tenant_id')
        
        # 执行服务化封装（传递租户和用户信息）
        result = deployment_service.service_model(
            model_id=model_id,
            config=config,
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        # 返回结果
        return jsonify({
            'success': True,
            'data': {
                'model_id': result.model_id,
                'service_id': result.service_id,
                'api_endpoints': result.api_endpoints,
                'service_type': result.service_type,
                'status': result.status,
                'configuration': {
                    'api_type': result.configuration.api_type,
                    'service_mesh': result.configuration.service_mesh,
                    'load_balancing': result.configuration.load_balancing,
                    'circuit_breaker': result.configuration.circuit_breaker,
                    'rate_limiting': result.configuration.rate_limiting,
                    'timeout': result.configuration.timeout,
                    'max_concurrent_requests': result.configuration.max_concurrent_requests
                },
                'deployment_time': result.deployment_time,
                'metadata': result.metadata
            }
        }), 200
        
    except Exception as e:
        logger.error(f"服务化封装失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'服务化封装失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/deployments/<deployment_id>/undeploy', methods=['POST'])
@jwt_required()
def undeploy_model(deployment_id: str):
    """
    取消部署模型
    
    Args:
        deployment_id: 部署ID
        
    Returns:
        操作结果
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        logger.info(f"用户 {user_id} 请求取消部署 {deployment_id}")
        
        # 执行取消部署
        success = deployment_service.undeploy_model(deployment_id)
        
        # 返回结果
        if success:
            return jsonify({
                'success': True,
                'message': f'部署 {deployment_id} 已成功取消'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': f'取消部署 {deployment_id} 失败'
            }), 500
        
    except Exception as e:
        logger.error(f"取消部署失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'取消部署失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/deployments/<deployment_id>/status', methods=['GET'])
@jwt_required()
def get_deployment_status(deployment_id: str):
    """
    获取部署状态
    
    Args:
        deployment_id: 部署ID
        
    Returns:
        部署状态信息
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        logger.info(f"用户 {user_id} 请求获取部署 {deployment_id} 状态")
        
        # 获取部署状态
        status = deployment_service.get_deployment_status(deployment_id)
        
        # 返回结果
        return jsonify({
            'success': True,
            'data': status
        }), 200
        
    except Exception as e:
        logger.error(f"获取部署状态失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取部署状态失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/deployments/<deployment_id>/scale', methods=['POST'])
@jwt_required()
def scale_deployment(deployment_id: str):
    """
    扩缩容部署
    
    Args:
        deployment_id: 部署ID
        
    Request Body:
        replicas: 目标副本数
        
    Returns:
        操作结果
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        logger.info(f"用户 {user_id} 请求扩缩容部署 {deployment_id}")
        
        # 获取请求数据
        data = request.get_json()
        replicas = data.get('replicas', 1)
        
        if replicas < 0:
            return jsonify({
                'success': False,
                'error': '副本数不能为负数'
            }), 400
        
        # 执行扩缩容
        success = deployment_service.scale_deployment(deployment_id, replicas)
        
        # 返回结果
        if success:
            return jsonify({
                'success': True,
                'message': f'部署 {deployment_id} 副本数已调整为 {replicas}'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': f'扩缩容部署 {deployment_id} 失败'
            }), 500
        
    except Exception as e:
        logger.error(f"扩缩容失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'扩缩容失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/deployments/<deployment_id>', methods=['GET'])
@jwt_required()
def get_deployment_detail(deployment_id: str):
    """
    获取单个部署详情
    
    Args:
        deployment_id: 部署ID
        
    Returns:
        部署详情
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        logger.info(f"用户 {user_id} 请求获取部署详情 {deployment_id}")
        
        deployment = deployment_service.get_deployment_by_id(
            deployment_id=deployment_id,
            tenant_id=tenant_id
        )
        
        if not deployment:
            return jsonify({
                'success': False,
                'error': f'部署 {deployment_id} 不存在或无权访问'
            }), 404
        
        return jsonify({
            'success': True,
            'data': deployment
        }), 200
        
    except Exception as e:
        logger.error(f"获取部署详情失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取部署详情失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/statistics', methods=['GET'])
@jwt_required()
def get_deployment_statistics():
    """
    获取部署统计信息
    
    Returns:
        部署统计数据
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        if not tenant_id:
            return jsonify({
                'success': False,
                'error': '缺少租户ID'
            }), 400
        
        logger.info(f"用户 {user_id} 请求获取部署统计, tenant_id: {tenant_id}")
        
        statistics = deployment_service.get_deployment_statistics(tenant_id)
        
        return jsonify({
            'success': True,
            'data': statistics
        }), 200
        
    except Exception as e:
        logger.error(f"获取部署统计失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取部署统计失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/modes', methods=['GET'])
@jwt_required()
def get_deployment_modes():
    """
    获取支持的部署模式和发布策略（支持租户隔离）
    
    从数据库或配置中获取所有可用的部署模式和发布策略，
    支持租户级别的自定义配置。
    
    Query Parameters:
        include_disabled: 是否包含已禁用的配置，默认false
    
    Returns:
        部署模式和发布策略列表，包含详细配置信息：
        {
            'deployment_modes': [...],       # 部署模式列表
            'release_strategies': [...],     # 发布策略列表
            'default_mode': str,             # 默认部署模式
            'default_strategy': str,         # 默认发布策略
            'statistics': {...}              # 使用统计
        }
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        
        # 获取租户ID
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        logger.info(f"用户 {user_id} 请求获取部署模式配置, tenant_id: {tenant_id}")
        
        # 获取查询参数
        include_disabled = request.args.get('include_disabled', 'false').lower() == 'true'
        
        # 调用服务层获取部署模式和发布策略
        result = deployment_service.get_deployment_modes(
            tenant_id=tenant_id,
            include_disabled=include_disabled
        )
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"获取部署模式失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取部署模式失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/modes/<mode_code>', methods=['GET'])
@jwt_required()
def get_mode_details(mode_code: str):
    """
    获取特定部署模式的详细信息
    
    Args:
        mode_code: 部署模式代码 (online, batch, edge, hybrid)
    
    Returns:
        部署模式的详细配置信息
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        logger.info(f"用户 {user_id} 请求获取部署模式详情: {mode_code}, tenant_id: {tenant_id}")
        
        result = deployment_service.get_mode_details(
            mode_code=mode_code,
            tenant_id=tenant_id
        )
        
        if result:
            return jsonify({
                'success': True,
                'data': result
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': f'部署模式 {mode_code} 不存在'
            }), 404
            
    except Exception as e:
        logger.error(f"获取部署模式详情失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取部署模式详情失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/strategies/<strategy_code>', methods=['GET'])
@jwt_required()
def get_strategy_details(strategy_code: str):
    """
    获取特定发布策略的详细信息
    
    Args:
        strategy_code: 发布策略代码 (rolling, canary, blue_green, ab_testing)
    
    Returns:
        发布策略的详细配置信息
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        logger.info(f"用户 {user_id} 请求获取发布策略详情: {strategy_code}, tenant_id: {tenant_id}")
        
        result = deployment_service.get_strategy_details(
            strategy_code=strategy_code,
            tenant_id=tenant_id
        )
        
        if result:
            return jsonify({
                'success': True,
                'data': result
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': f'发布策略 {strategy_code} 不存在'
            }), 404
            
    except Exception as e:
        logger.error(f"获取发布策略详情失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取发布策略详情失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/modes/custom', methods=['POST'])
@jwt_required()
def create_custom_mode():
    """
    创建租户自定义部署模式
    
    Request Body:
        code: 配置代码（租户内唯一）
        name: 显示名称
        description: 描述
        default_config: 默认配置
        icon: 图标标识（可选）
        category: 分类（可选）
        required_resources: 所需资源配置（可选）
        supported_features: 支持的特性列表（可选）
        min_replicas: 最小副本数（可选）
        max_replicas: 最大副本数（可选）
        recommended_scenarios: 推荐使用场景（可选）
    
    Returns:
        创建的配置信息
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        if not tenant_id:
            return jsonify({
                'success': False,
                'error': '创建自定义部署模式需要提供租户ID'
            }), 400
        
        data = request.get_json()
        code = data.get('code')
        name = data.get('name')
        description = data.get('description')
        default_config = data.get('default_config', {})
        
        if not all([code, name, description]):
            return jsonify({
                'success': False,
                'error': '缺少必要参数: code, name, description'
            }), 400
        
        logger.info(f"用户 {user_id} 创建自定义部署模式: {code}, tenant_id: {tenant_id}")
        
        # 提取可选参数
        optional_params = {
            k: v for k, v in data.items()
            if k in ['icon', 'category', 'required_resources', 'supported_features',
                     'limitations', 'min_replicas', 'max_replicas', 'requires_gpu',
                     'recommended_scenarios', 'sort_order', 'tags']
        }
        
        result = deployment_service.create_custom_deployment_mode(
            tenant_id=tenant_id,
            code=code,
            name=name,
            description=description,
            default_config=default_config,
            user_id=user_id,
            **optional_params
        )
        
        if result:
            return jsonify({
                'success': True,
                'data': result,
                'message': f'自定义部署模式 {code} 创建成功'
            }), 201
        else:
            return jsonify({
                'success': False,
                'error': f'创建自定义部署模式失败，可能已存在同名配置'
            }), 400
            
    except Exception as e:
        logger.error(f"创建自定义部署模式失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'创建自定义部署模式失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/config/validate', methods=['POST'])
@jwt_required()
def validate_deployment_config():
    """
    验证部署配置
    
    检查部署配置是否符合所选部署模式和发布策略的要求。
    
    Request Body:
        mode: 部署模式代码
        release_strategy: 发布策略代码
        config: 用户提供的配置
    
    Returns:
        验证结果：
        {
            'valid': bool,
            'errors': [...],
            'warnings': [...],
            'suggestions': [...]
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        data = request.get_json()
        mode_code = data.get('mode', 'online')
        strategy_code = data.get('release_strategy', 'rolling')
        config = data.get('config', {})
        
        logger.info(
            f"用户 {user_id} 验证部署配置: mode={mode_code}, strategy={strategy_code}"
        )
        
        result = deployment_service.validate_deployment_config(
            mode_code=mode_code,
            strategy_code=strategy_code,
            config=config,
            tenant_id=tenant_id
        )
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"验证部署配置失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'验证部署配置失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/models/<model_id>/deployment-history', methods=['GET'])
@jwt_required()
def get_deployment_history(model_id: str):
    """
    获取模型部署历史（支持租户隔离）
    
    Args:
        model_id: 模型ID
        
    Query Parameters:
        limit: 限制返回记录数，默认10
        
    Returns:
        部署历史记录
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        
        # 获取租户ID
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        logger.info(f"用户 {user_id} 请求获取模型 {model_id} 的部署历史, tenant_id: {tenant_id}")
        
        # 获取查询参数
        limit = request.args.get('limit', 10, type=int)
        
        # 调用服务获取部署历史
        history = deployment_service.get_deployment_history(
            model_id=model_id,
            tenant_id=tenant_id,
            limit=limit
        )
        
        return jsonify({
            'success': True, 
            'data': {
                'history': history,
                'total': len(history)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"获取部署历史失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取部署历史失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/deployments/<deployment_id>/rollback', methods=['POST'])
@jwt_required()
def rollback_deployment(deployment_id: str):
    """
    手动触发部署回滚（支持租户隔离和版本选择）
    
    执行部署回滚操作，可以回滚到指定版本或上一版本。
    
    Args:
        deployment_id: 部署ID
        
    Request Body (JSON, optional):
        target_version: 目标版本号，为空则回滚到上一版本
        reason: 回滚原因
        force: 是否强制回滚（忽略某些检查），默认false
        
    Returns:
        回滚操作结果：
        {
            'success': bool,
            'deployment_id': str,
            'old_version': str,
            'new_version': str,
            'status': str,
            'message': str,
            'rollback_details': {...}
        }
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        
        # 获取租户ID
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        logger.info(
            f"用户 {user_id} 请求手动回滚部署 {deployment_id}, tenant_id: {tenant_id}"
        )
        
        # 获取请求参数
        data = request.get_json() or {}
        target_version = data.get('target_version')
        rollback_reason = data.get('reason')
        force = data.get('force', False)
        
        # 调用服务层执行回滚
        result = deployment_service.rollback_deployment(
            deployment_id=deployment_id,
            target_version=target_version,
            rollback_reason=rollback_reason,
            tenant_id=tenant_id,
            user_id=user_id,
            force=force
        )
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'message': result.get('message', f'部署 {deployment_id} 已回滚'),
                'data': result
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('message', f'部署 {deployment_id} 回滚失败'),
                'data': result
            }), 400
            
    except Exception as e:
        logger.error(f"回滚失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'回滚失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/deployments/<deployment_id>/rollback/versions', methods=['GET'])
@jwt_required()
def get_rollback_versions(deployment_id: str):
    """
    获取可用的回滚版本列表
    
    Args:
        deployment_id: 部署ID
        
    Returns:
        可用回滚版本列表：
        {
            'current_version': str,
            'previous_version': str,
            'available_versions': [...],
            'recommended_version': str
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        logger.info(
            f"用户 {user_id} 获取部署 {deployment_id} 的可回滚版本, tenant_id: {tenant_id}"
        )
        
        result = deployment_service.get_available_rollback_versions(
            deployment_id=deployment_id,
            tenant_id=tenant_id
        )
        
        if result.get('error'):
            return jsonify({
                'success': False,
                'error': result.get('error')
            }), 404
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"获取可回滚版本失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取可回滚版本失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/deployments/<deployment_id>/rollback/history', methods=['GET'])
@jwt_required()
def get_rollback_history(deployment_id: str):
    """
    获取部署的回滚历史
    
    Args:
        deployment_id: 部署ID
        
    Returns:
        回滚历史列表
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        logger.info(
            f"用户 {user_id} 获取部署 {deployment_id} 的回滚历史, tenant_id: {tenant_id}"
        )
        
        history = deployment_service.get_rollback_history(
            deployment_id=deployment_id,
            tenant_id=tenant_id
        )
        
        return jsonify({
            'success': True,
            'data': {
                'history': history,
                'total': len(history)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"获取回滚历史失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取回滚历史失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/deployments/<deployment_id>/rollback/preview', methods=['POST'])
@jwt_required()
def preview_rollback(deployment_id: str):
    """
    预览回滚操作
    
    不执行实际回滚，只返回回滚将产生的变更预览。
    
    Args:
        deployment_id: 部署ID
        
    Request Body (JSON, optional):
        target_version: 目标版本号，为空则预览回滚到上一版本
        
    Returns:
        回滚预览信息：
        {
            'can_rollback': bool,
            'current_version': str,
            'target_version': str,
            'config_changes': [...],
            'warnings': [...]
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        data = request.get_json() or {}
        target_version = data.get('target_version')
        
        logger.info(
            f"用户 {user_id} 预览部署 {deployment_id} 的回滚, "
            f"target_version={target_version}, tenant_id={tenant_id}"
        )
        
        result = deployment_service.preview_rollback(
            deployment_id=deployment_id,
            target_version=target_version,
            tenant_id=tenant_id
        )
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"预览回滚失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'预览回滚失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/deployments/<deployment_id>/audit', methods=['GET'])
@jwt_required()
def get_deployment_audit(deployment_id: str):
    """
    获取部署审计事件列表（支持租户隔离和多种过滤）
    
    Args:
        deployment_id: 部署ID
        
    Query Parameters:
        event_type: 事件类型过滤（deploy, undeploy, scale, rollback, update, health_check, error, warning）
        action: 动作过滤
        status: 状态过滤（success, failure, pending）
        start_time: 开始时间（ISO格式）
        end_time: 结束时间（ISO格式）
        limit: 返回数量限制，默认100
        offset: 偏移量，默认0
        
    Returns:
        审计事件列表：
        {
            'events': [...],
            'total': int,
            'has_more': bool,
            'source': str
        }
    """
    try:
        # 获取当前用户ID
        user_id = get_jwt_identity()
        
        # 获取租户ID
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        # 获取查询参数
        event_type = request.args.get('event_type')
        action = request.args.get('action')
        status = request.args.get('status')
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        logger.info(
            f"用户 {user_id} 获取部署 {deployment_id} 审计事件, "
            f"tenant_id={tenant_id}, event_type={event_type}, limit={limit}"
        )
        
        # 调用服务层获取审计事件
        result = deployment_service.get_deployment_audit(
            deployment_id=deployment_id,
            tenant_id=tenant_id,
            event_type=event_type,
            action=action,
            status=status,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset
        )
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"获取审计事件失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取审计事件失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/deployments/<deployment_id>/audit', methods=['POST'])
@jwt_required()
def record_audit_event(deployment_id: str):
    """
    记录部署审计事件
    
    Args:
        deployment_id: 部署ID
        
    Request Body (JSON):
        event_type: 事件类型（必需）
        action: 具体动作（必需）
        status: 操作状态，默认'success'
        message: 事件消息
        description: 详细描述
        old_value: 变更前的值
        new_value: 变更后的值
        trigger: 触发方式（manual, automatic, scheduled）
        from_version: 原版本
        to_version: 目标版本
        resource_changes: 资源配置变更
        correlation_id: 关联ID
        tags: 标签列表
        metadata: 额外元数据
        
    Returns:
        创建的审计事件
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        data = request.get_json() or {}
        
        # 验证必需参数
        event_type = data.get('event_type')
        action = data.get('action')
        
        if not event_type or not action:
            return jsonify({
                'success': False,
                'error': '缺少必要参数: event_type, action'
            }), 400
        
        # 获取客户端信息
        ip_address = request.remote_addr
        user_agent = request.headers.get('User-Agent', '')[:500]
        
        logger.info(f"用户 {user_id} 记录审计事件: {event_type}/{action} for {deployment_id}")
        
        # 调用服务层记录审计事件
        result = deployment_service.record_audit_event(
            deployment_id=deployment_id,
            event_type=event_type,
            action=action,
            tenant_id=tenant_id,
            user_id=user_id,
            model_id=data.get('model_id'),
            status=data.get('status', 'success'),
            message=data.get('message'),
            description=data.get('description'),
            old_value=data.get('old_value'),
            new_value=data.get('new_value'),
            source='api',
            trigger=data.get('trigger', 'manual'),
            ip_address=ip_address,
            user_agent=user_agent,
            duration_ms=data.get('duration_ms'),
            from_version=data.get('from_version'),
            to_version=data.get('to_version'),
            resource_changes=data.get('resource_changes'),
            error_code=data.get('error_code'),
            error_message=data.get('error_message'),
            correlation_id=data.get('correlation_id'),
            parent_event_id=data.get('parent_event_id'),
            tags=data.get('tags'),
            metadata=data.get('metadata')
        )
        
        if result:
            return jsonify({
                'success': True,
                'data': result,
                'message': '审计事件记录成功'
            }), 201
        else:
            return jsonify({
                'success': False,
                'error': '记录审计事件失败'
            }), 500
            
    except Exception as e:
        logger.error(f"记录审计事件失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'记录审计事件失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/audit/statistics', methods=['GET'])
@jwt_required()
def get_audit_statistics():
    """
    获取审计事件统计信息
    
    Query Parameters:
        deployment_id: 部署ID（可选，不提供则统计租户下所有部署）
        start_time: 开始时间（ISO格式）
        end_time: 结束时间（ISO格式）
        
    Returns:
        统计信息：
        {
            'total_events': int,
            'success_count': int,
            'failure_count': int,
            'success_rate': float,
            'by_event_type': {...},
            'by_status': {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        if not tenant_id:
            return jsonify({
                'success': False,
                'error': '获取审计统计需要提供租户ID'
            }), 400
        
        deployment_id = request.args.get('deployment_id')
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        
        logger.info(
            f"用户 {user_id} 获取审计统计, tenant_id={tenant_id}, "
            f"deployment_id={deployment_id}"
        )
        
        result = deployment_service.get_audit_statistics(
            tenant_id=tenant_id,
            deployment_id=deployment_id,
            start_time=start_time,
            end_time=end_time
        )
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"获取审计统计失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取审计统计失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/audit/correlated/<correlation_id>', methods=['GET'])
@jwt_required()
def get_correlated_audit_events(correlation_id: str):
    """
    获取关联的审计事件
    
    通过关联ID追踪一系列相关的审计事件。
    
    Args:
        correlation_id: 关联ID
        
    Returns:
        关联的事件列表（按时间顺序）
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        logger.info(
            f"用户 {user_id} 获取关联审计事件, correlation_id={correlation_id}"
        )
        
        events = deployment_service.get_correlated_events(
            correlation_id=correlation_id,
            tenant_id=tenant_id
        )
        
        return jsonify({
            'success': True,
            'data': {
                'events': events,
                'total': len(events),
                'correlation_id': correlation_id
            }
        }), 200
        
    except Exception as e:
        logger.error(f"获取关联审计事件失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'获取关联审计事件失败: {str(e)}'
        }), 500


@model_deployment_bp.route('/audit/cleanup', methods=['POST'])
@jwt_required()
def cleanup_audit_events():
    """
    清理过期的审计事件
    
    Request Body (JSON):
        retention_days: 保留天数，默认90天
        
    Returns:
        清理结果
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        if not tenant_id:
            return jsonify({
                'success': False,
                'error': '清理审计事件需要提供租户ID'
            }), 400
        
        data = request.get_json() or {}
        retention_days = data.get('retention_days', 90)
        
        logger.info(
            f"用户 {user_id} 清理审计事件, tenant_id={tenant_id}, "
            f"retention_days={retention_days}"
        )
        
        result = deployment_service.cleanup_audit_events(
            tenant_id=tenant_id,
            retention_days=retention_days
        )
        
        return jsonify({
            'success': result.get('success', False),
            'data': result
        }), 200 if result.get('success') else 500
        
    except Exception as e:
        logger.error(f"清理审计事件失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'清理审计事件失败: {str(e)}'
        }), 500
