"""认证API接口

提供生产级的认证API，集成Agent长记忆推理能力。

功能模块:
- 用户认证: 注册、登录、登出、令牌管理
- 用户管理: 资料获取、更新、密码修改
- 安全分析: 风险评估、异常检测、行为分析
- Agent推理: 智能风险检测、安全建议、记忆管理
- 会话管理: 多设备会话、会话安全监控
- 统计分析: 登录统计、安全统计
"""

from flask import Blueprint, request
import logging
from typing import Dict, Any, Optional
from functools import wraps

from backend.utils.response import success_response, error_response
from backend.utils.validation import (
    validate_json, validate_required_fields, validate_string_length,
    validate_phone_format, validate_email_format
)
from backend.services.auth_service import AuthService, get_auth_service
from backend.modules.auth.auth_exceptions import (
    AuthException, AuthenticationError, AuthorizationError, UserNotFoundError,
    InvalidTokenError, ExpiredTokenError, InvalidCredentialsError,
    UserAlreadyExistsError, PermissionDeniedError
)

logger = logging.getLogger(__name__)

# 创建蓝图
auth_api_bp = Blueprint('auth_api', __name__, url_prefix='/api/v2/auth')


def get_auth_service_instance() -> AuthService:
    """获取认证服务实例
    
    Returns:
        AuthService: 认证服务实例
    """
    return get_auth_service()


def require_auth(f):
    """认证装饰器
    
    验证请求中的访问令牌。
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return error_response("缺少访问令牌", 401, "MissingToken")
        
        token = auth_header.split(' ', 1)[1]
        auth_service = get_auth_service_instance()
        
        try:
            payload = auth_service.verify_token(token)
            request.current_user_id = payload.get('user_id')
            request.current_session_id = payload.get('session_id')
            request.current_tenant_id = payload.get('tenant_id')
        except (InvalidTokenError, ExpiredTokenError) as e:
            return error_response(str(e), 401, "InvalidToken", e.details)
        except Exception as e:
            logger.error(f"Token verification failed: {str(e)}")
            return error_response("令牌验证失败", 401, "TokenVerificationError")
        
        return f(*args, **kwargs)
    return decorated


def get_client_info() -> Dict[str, Any]:
    """获取客户端信息
    
    Returns:
        Dict: 客户端信息
    """
    return {
        'ip_address': request.headers.get('X-Forwarded-For', request.remote_addr),
        'user_agent': request.headers.get('User-Agent'),
        'device_fingerprint': request.headers.get('X-Device-Fingerprint'),
        'location': request.headers.get('X-Location')
    }


# ============================================================================
# 用户认证端点
# ============================================================================

@auth_api_bp.route('/register', methods=['POST'])
def register():
    """用户注册
    
    创建新用户账户。
    
    Request Body:
        - username (str, required): 用户名 (3-50字符)
        - email (str, required): 邮箱地址
        - password (str, required): 密码 (至少8字符，包含大小写字母和数字)
        - full_name (str, optional): 全名
        - phone (str, optional): 手机号码
        
    Returns:
        - user_id: 用户ID
        - username: 用户名
        - email: 邮箱
        - full_name: 全名
        - created_at: 创建时间
        
    Raises:
        - 400: 验证失败
        - 409: 用户已存在
        - 500: 注册失败
    """
    try:
        # 验证请求数据
        data = validate_json(request)
        validate_required_fields(data, ['username', 'email', 'password'])

        username = data['username'].strip()
        email = data['email'].strip().lower()
        password = data['password']
        full_name = data.get('full_name', '').strip()
        phone = data.get('phone', '').strip() if data.get('phone') else None

        # 验证字段长度
        validate_string_length(username, 'username', 3, 50)
        validate_string_length(email, 'email', 5, 200)
        validate_string_length(password, 'password', 6, 128)

        # 委托给AuthService处理业务逻辑
        auth_service = get_auth_service_instance()
        user = auth_service.register_user(
            username=username,
            email=email,
            password=password,
            full_name=full_name,
            phone=phone
        )

        result = {
            "user_id": str(user.id),
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "created_at": user.created_at.isoformat()
        }

        return success_response(result, "用户注册成功")

    except UserAlreadyExistsError as e:
        return error_response(str(e), 409, "UserAlreadyExists", e.details)
    except ValueError as e:
        return error_response(str(e), 400, "ValidationError", {"error": str(e)})
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        return error_response("注册失败", 500, "RegistrationError", {"error": str(e)})


@auth_api_bp.route('/login', methods=['POST'])
def login():
    """用户登录
    
    统一用户登录接口，支持用户名/邮箱+密码登录。
    包含智能风险评估和行为分析。
    
    Request Body:
        - identifier (str, required): 用户名或邮箱
        - password (str, required): 密码
        - mfa_code (str, optional): MFA验证码
        
    Headers:
        - X-Forwarded-For: 客户端IP
        - User-Agent: 用户代理
        - X-Device-Fingerprint: 设备指纹
        - X-Location: 地理位置
        
    Returns:
        - tokens: 令牌信息
            - access_token: 访问令牌
            - refresh_token: 刷新令牌
            - expires_in: 过期时间（秒）
        - user: 用户信息
        - session_id: 会话ID
        - risk_assessment: 风险评估结果
        
    Raises:
        - 400: 验证失败
        - 401: 认证失败
        - 500: 登录失败
    """
    try:
        # 验证请求数据
        data = validate_json(request)
        
        # 支持新格式（identifier + password）和旧格式（username + password）
        if 'identifier' in data:
            identifier = data['identifier'].strip()
        elif 'username' in data:
            identifier = data['username'].strip()
        else:
            return error_response("需要提供用户名或邮箱", 400, "ValidationError")
        
        password = data.get('password')
        if not password:
            return error_response("需要提供密码", 400, "ValidationError")
        
        mfa_code = data.get('mfa_code')
        
        # 获取客户端信息
        client_info = get_client_info()

        # 委托给AuthService处理业务逻辑
        auth_service = get_auth_service_instance()
        auth_result = auth_service.authenticate_user(
            identifier=identifier,
            password=password,
            ip_address=client_info['ip_address'],
            user_agent=client_info['user_agent'],
            device_fingerprint=client_info['device_fingerprint'],
            location=client_info['location']
        )
        
        result = {
            "tokens": auth_result['tokens'],
            "user": auth_result['user'],
            "session_id": auth_result['session_id'],
            "risk_assessment": auth_result.get('risk_assessment')
        }

        return success_response(result, "登录成功")

    except (UserNotFoundError, InvalidCredentialsError) as e:
        return error_response(str(e), 401, "AuthenticationFailed", e.details)
    except AuthenticationError as e:
        return error_response(str(e), 401, "AuthenticationError", e.details)
    except ValueError as e:
        return error_response(str(e), 400, "ValidationError", {"error": str(e)})
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")
        return error_response("登录失败", 500, "LoginError", {"error": str(e)})


@auth_api_bp.route('/logout', methods=['POST'])
@require_auth
def logout():
    """用户登出
    
    撤销当前会话的访问令牌。
    
    Headers:
        - Authorization: Bearer <access_token>
        
    Returns:
        - message: 登出成功消息
        
    Raises:
        - 401: 未认证
        - 500: 登出失败
    """
    try:
        auth_header = request.headers.get('Authorization')
        token = auth_header.split(' ', 1)[1]
        
        auth_service = get_auth_service_instance()
        auth_service.revoke_token(token)
        
        logger.info(f"User {request.current_user_id} logged out")
        
        return success_response({
            'message': '登出成功'
        })

    except Exception as e:
        logger.error(f"Logout failed: {str(e)}")
        return error_response("登出失败", 500, "LogoutError", {"error": str(e)})


@auth_api_bp.route('/logout/all', methods=['POST'])
@require_auth
def logout_all_devices():
    """登出所有设备
    
    撤销用户所有会话，可选保留当前会话。
    
    Headers:
        - Authorization: Bearer <access_token>
        
    Request Body:
        - keep_current (bool, optional): 是否保留当前会话，默认false
        
    Returns:
        - message: 成功消息
        - sessions_revoked: 撤销的会话数量
        
    Raises:
        - 401: 未认证
        - 500: 操作失败
    """
    try:
        data = request.get_json() or {}
        keep_current = data.get('keep_current', False)
        
        exclude_session = request.current_session_id if keep_current else None
        
        auth_service = get_auth_service_instance()
        count = auth_service.logout_all_devices(request.current_user_id, exclude_session)
        
        return success_response({
            'message': f'已登出 {count} 个设备',
            'sessions_revoked': count
        })

    except Exception as e:
        logger.error(f"Logout all devices failed: {str(e)}")
        return error_response("操作失败", 500, "LogoutError", {"error": str(e)})


@auth_api_bp.route('/refresh', methods=['POST'])
def refresh_token():
    """刷新访问令牌
    
    使用刷新令牌获取新的访问令牌。
    
    Request Body:
        - refresh_token (str, optional): 刷新令牌
        
    Headers:
        - Authorization: Bearer <refresh_token> (可选)
        
    Returns:
        - access_token: 新的访问令牌
        - refresh_token: 新的刷新令牌
        
    Raises:
        - 400: 缺少刷新令牌
        - 401: 刷新令牌无效或已过期
        - 500: 刷新失败
    """
    try:
        refresh_token_value = None
        
        # 首先尝试从Authorization头获取
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            refresh_token_value = auth_header.split(' ', 1)[1]
        
        # 如果头部没有，尝试从请求体获取
        if not refresh_token_value:
            try:
                data = validate_json(request)
                refresh_token_value = data.get('refresh_token')
            except:
                pass
        
        if not refresh_token_value:
            return error_response("缺少刷新令牌", 400, "MissingRefreshToken")

        auth_service = get_auth_service_instance()
        access_token, new_refresh_token = auth_service.refresh_token(refresh_token_value)
        
        result = {
            "access_token": access_token,
            "refresh_token": new_refresh_token
        }

        return success_response(result, "令牌刷新成功")

    except (InvalidTokenError, ExpiredTokenError) as e:
        return error_response(str(e), 401, "InvalidToken", e.details)
    except Exception as e:
        logger.error(f"Token refresh failed: {str(e)}")
        return error_response("令牌刷新失败", 500, "TokenRefreshError", {"error": str(e)})


# ============================================================================
# 用户资料端点
# ============================================================================

@auth_api_bp.route('/profile', methods=['GET'])
@require_auth
def get_profile():
    """获取用户资料
    
    获取当前登录用户的详细资料。
    
    Headers:
        - Authorization: Bearer <access_token>
        
    Returns:
        - user: 用户信息
            - id: 用户ID
            - username: 用户名
            - email: 邮箱
            - full_name: 全名
            - phone: 手机号码
            - is_active: 是否激活
            - is_superuser: 是否超级用户
            - mfa_enabled: 是否启用MFA
            - trust_score: 信任分数
            - created_at: 创建时间
            - last_login: 最后登录时间
            
    Raises:
        - 401: 未认证
        - 404: 用户不存在
        - 500: 获取失败
    """
    try:
        auth_service = get_auth_service_instance()
        user = auth_service.get_user_by_id(request.current_user_id)
        
        if not user:
            return error_response("用户不存在", 404, "UserNotFound")
        
        result = {
            "user": user.to_dict()
        }

        return success_response(result, "获取用户资料成功")

    except Exception as e:
        logger.error(f"Get profile failed: {str(e)}")
        return error_response("获取用户资料失败", 500, "GetProfileError", {"error": str(e)})


@auth_api_bp.route('/profile', methods=['PUT'])
@require_auth
def update_profile():
    """更新用户资料
    
    更新当前登录用户的资料信息。
    
    Headers:
        - Authorization: Bearer <access_token>
        
    Request Body:
        - full_name (str, optional): 全名
        - phone (str, optional): 手机号码
        - avatar_url (str, optional): 头像URL
        - preferences (dict, optional): 用户偏好设置
        
    Returns:
        - user: 更新后的用户信息
        
    Raises:
        - 401: 未认证
        - 400: 验证失败
        - 500: 更新失败
    """
    try:
        data = validate_json(request)
        
        # 允许更新的字段
        allowed_fields = ['full_name', 'phone', 'avatar_url', 'preferences', 'first_name', 'last_name']
        update_data = {k: v for k, v in data.items() if k in allowed_fields}
        
        if not update_data:
            return error_response("没有可更新的字段", 400, "ValidationError")
        
        auth_service = get_auth_service_instance()
        user = auth_service.update_user(request.current_user_id, update_data)
        
        if not user:
            return error_response("用户不存在", 404, "UserNotFound")
        
        return success_response({
            "user": user.to_dict()
        }, "用户资料更新成功")

    except ValueError as e:
        return error_response(str(e), 400, "ValidationError", {"error": str(e)})
    except Exception as e:
        logger.error(f"Update profile failed: {str(e)}")
        return error_response("更新用户资料失败", 500, "UpdateProfileError", {"error": str(e)})


@auth_api_bp.route('/password', methods=['PUT'])
@require_auth
def change_password():
    """修改密码
    
    修改当前登录用户的密码。
    
    Headers:
        - Authorization: Bearer <access_token>
        
    Request Body:
        - old_password (str, required): 当前密码
        - new_password (str, required): 新密码 (至少8字符，包含大小写字母和数字)
        
    Returns:
        - message: 成功消息
        
    Raises:
        - 401: 未认证或当前密码错误
        - 400: 验证失败
        - 500: 修改失败
    """
    try:
        data = validate_json(request)
        validate_required_fields(data, ['old_password', 'new_password'])
        
        old_password = data['old_password']
        new_password = data['new_password']
        
        auth_service = get_auth_service_instance()
        auth_service.change_password(request.current_user_id, old_password, new_password)
        
        return success_response({
            'message': '密码修改成功，请重新登录'
        })

    except InvalidCredentialsError as e:
        return error_response(str(e), 401, "InvalidPassword", e.details)
    except ValueError as e:
        return error_response(str(e), 400, "ValidationError", {"error": str(e)})
    except Exception as e:
        logger.error(f"Change password failed: {str(e)}")
        return error_response("密码修改失败", 500, "ChangePasswordError", {"error": str(e)})


# ============================================================================
# 会话管理端点
# ============================================================================

@auth_api_bp.route('/sessions', methods=['GET'])
@require_auth
def get_sessions():
    """获取用户会话
    
    获取当前用户的所有活跃会话。
    
    Headers:
        - Authorization: Bearer <access_token>
        
    Returns:
        - sessions: 会话列表
        - current_session_id: 当前会话ID
        
    Raises:
        - 401: 未认证
        - 500: 获取失败
    """
    try:
        auth_service = get_auth_service_instance()
        sessions = auth_service.get_user_sessions(request.current_user_id)
        
        return success_response({
            'sessions': sessions,
            'current_session_id': request.current_session_id
        })

    except Exception as e:
        logger.error(f"Get sessions failed: {str(e)}")
        return error_response("获取会话列表失败", 500, "GetSessionsError", {"error": str(e)})


@auth_api_bp.route('/sessions/<session_id>', methods=['DELETE'])
@require_auth
def revoke_session(session_id):
    """撤销指定会话
    
    撤销用户的指定会话。
    
    Headers:
        - Authorization: Bearer <access_token>
        
    Path Parameters:
        - session_id: 要撤销的会话ID
        
    Returns:
        - message: 成功消息
        
    Raises:
        - 401: 未认证
        - 404: 会话不存在
        - 500: 撤销失败
    """
    try:
        auth_service = get_auth_service_instance()
        
        # 验证会话属于当前用户
        if not auth_service.validate_session(session_id, request.current_user_id):
            return error_response("会话不存在或无权限", 404, "SessionNotFound")
        
        auth_service.logout_user(request.current_user_id, session_id)
        
        return success_response({
            'message': '会话已撤销'
        })

    except Exception as e:
        logger.error(f"Revoke session failed: {str(e)}")
        return error_response("撤销会话失败", 500, "RevokeSessionError", {"error": str(e)})


# ============================================================================
# 安全分析端点
# ============================================================================

@auth_api_bp.route('/security/analysis', methods=['GET'])
@require_auth
def get_security_analysis():
    """获取安全分析
    
    使用Agent进行综合安全分析，包括风险评估、异常检测和行为分析。
    
    Headers:
        - Authorization: Bearer <access_token>
        
    Returns:
        - trust_score: 信任分数 (0.0-1.0)
        - risk_level: 风险等级 (low, medium, high, critical)
        - anomalies: 检测到的异常列表
        - recommendations: 安全建议
        - agent_insights: Agent智能洞察
        
    Raises:
        - 401: 未认证
        - 404: 用户不存在
        - 500: 分析失败
    """
    try:
        auth_service = get_auth_service_instance()
        analysis = auth_service.analyze_user_security(request.current_user_id)
        
        return success_response({
            'trust_score': analysis.trust_score,
            'risk_level': analysis.risk_level,
            'anomalies': analysis.anomalies,
            'recommendations': analysis.recommendations,
            'agent_insights': analysis.agent_insights
        }, "安全分析完成")

    except UserNotFoundError as e:
        return error_response(str(e), 404, "UserNotFound", e.details)
    except Exception as e:
        logger.error(f"Security analysis failed: {str(e)}")
        return error_response("安全分析失败", 500, "SecurityAnalysisError", {"error": str(e)})


@auth_api_bp.route('/security/events', methods=['GET'])
@require_auth
def get_security_events():
    """获取安全事件
    
    获取用户的安全事件列表。
    
    Headers:
        - Authorization: Bearer <access_token>
        
    Query Parameters:
        - event_type (str, optional): 事件类型过滤
        - severity (str, optional): 严重程度过滤 (low, medium, high, critical)
        - is_resolved (bool, optional): 是否已解决过滤
        - days (int, optional): 时间范围（天），默认30
        - offset (int, optional): 偏移量，默认0
        - limit (int, optional): 限制数量，默认100
        
    Returns:
        - events: 安全事件列表
        - total: 总数
        
    Raises:
        - 401: 未认证
        - 500: 获取失败
    """
    try:
        event_type = request.args.get('event_type')
        severity = request.args.get('severity')
        is_resolved = request.args.get('is_resolved')
        days = int(request.args.get('days', 30))
        offset = int(request.args.get('offset', 0))
        limit = min(int(request.args.get('limit', 100)), 500)
        
        if is_resolved is not None:
            is_resolved = is_resolved.lower() == 'true'
        
        auth_service = get_auth_service_instance()
        events, total = auth_service.get_security_events(
            user_id=request.current_user_id,
            event_type=event_type,
            severity=severity,
            days=days,
            offset=offset,
            limit=limit
        )
        
        return success_response({
            'events': events,
            'total': total,
            'offset': offset,
            'limit': limit
        })

    except Exception as e:
        logger.error(f"Get security events failed: {str(e)}")
        return error_response("获取安全事件失败", 500, "GetSecurityEventsError", {"error": str(e)})


@auth_api_bp.route('/security/login-statistics', methods=['GET'])
@require_auth
def get_login_statistics():
    """获取登录统计
    
    获取用户的登录统计数据。
    
    Headers:
        - Authorization: Bearer <access_token>
        
    Query Parameters:
        - days (int, optional): 统计天数，默认30
        
    Returns:
        - total_attempts: 总尝试次数
        - successful: 成功次数
        - failed: 失败次数
        - success_rate: 成功率
        - top_ips: 最常用IP列表
        - period_days: 统计周期
        
    Raises:
        - 401: 未认证
        - 500: 获取失败
    """
    try:
        days = int(request.args.get('days', 30))
        
        auth_service = get_auth_service_instance()
        statistics = auth_service.get_login_statistics(
            user_id=request.current_user_id,
            days=days
        )
        
        return success_response(statistics)

    except Exception as e:
        logger.error(f"Get login statistics failed: {str(e)}")
        return error_response("获取登录统计失败", 500, "GetLoginStatisticsError", {"error": str(e)})


# ============================================================================
# Agent 记忆端点
# ============================================================================

@auth_api_bp.route('/security/memories', methods=['GET'])
@require_auth
def get_security_memories():
    """获取安全记忆
    
    获取用户的Agent安全记忆。
    
    Headers:
        - Authorization: Bearer <access_token>
        
    Query Parameters:
        - memory_type (str, optional): 记忆类型过滤
        - limit (int, optional): 限制数量，默认50
        
    Returns:
        - memories: 记忆列表
        
    Raises:
        - 401: 未认证
        - 500: 获取失败
    """
    try:
        memory_type = request.args.get('memory_type')
        limit = min(int(request.args.get('limit', 50)), 200)
        
        auth_service = get_auth_service_instance()
        memories = auth_service.get_user_memories(
            user_id=request.current_user_id,
            memory_type=memory_type,
            limit=limit
        )
        
        return success_response({
            'memories': memories
        })

    except Exception as e:
        logger.error(f"Get security memories failed: {str(e)}")
        return error_response("获取安全记忆失败", 500, "GetMemoriesError", {"error": str(e)})


@auth_api_bp.route('/security/memories', methods=['POST'])
@require_auth
def add_security_memory():
    """添加安全记忆
    
    添加用户的Agent安全记忆。
    
    Headers:
        - Authorization: Bearer <access_token>
        
    Request Body:
        - memory_type (str, required): 记忆类型
        - content (str, required): 记忆内容
        - importance (float, optional): 重要性 (0.0-1.0)，默认0.5
        - metadata (dict, optional): 元数据
        
    Returns:
        - memory: 创建的记忆
        
    Raises:
        - 401: 未认证
        - 400: 验证失败
        - 500: 添加失败
    """
    try:
        data = validate_json(request)
        validate_required_fields(data, ['memory_type', 'content'])
        
        memory_type = data['memory_type']
        content = data['content']
        importance = float(data.get('importance', 0.5))
        metadata = data.get('metadata')
        
        auth_service = get_auth_service_instance()
        memory = auth_service.add_security_memory(
            user_id=request.current_user_id,
            memory_type=memory_type,
            content=content,
            importance=importance,
            metadata=metadata
        )
        
        return success_response({
            'memory': memory
        }, "安全记忆添加成功")

    except ValueError as e:
        return error_response(str(e), 400, "ValidationError", {"error": str(e)})
    except Exception as e:
        logger.error(f"Add security memory failed: {str(e)}")
        return error_response("添加安全记忆失败", 500, "AddMemoryError", {"error": str(e)})


@auth_api_bp.route('/security/memories/search', methods=['POST'])
@require_auth
def search_security_memories():
    """搜索安全记忆
    
    搜索用户的Agent安全记忆。
    
    Headers:
        - Authorization: Bearer <access_token>
        
    Request Body:
        - query (str, required): 搜索查询
        - limit (int, optional): 限制数量，默认10
        
    Returns:
        - memories: 匹配的记忆列表
        
    Raises:
        - 401: 未认证
        - 400: 验证失败
        - 500: 搜索失败
    """
    try:
        data = validate_json(request)
        validate_required_fields(data, ['query'])
        
        query = data['query']
        limit = min(int(data.get('limit', 10)), 50)
        
        auth_service = get_auth_service_instance()
        memories = auth_service.search_security_memories(
            query=query,
            user_id=request.current_user_id,
            limit=limit
        )
        
        return success_response({
            'memories': memories,
            'query': query
        })

    except ValueError as e:
        return error_response(str(e), 400, "ValidationError", {"error": str(e)})
    except Exception as e:
        logger.error(f"Search security memories failed: {str(e)}")
        return error_response("搜索安全记忆失败", 500, "SearchMemoriesError", {"error": str(e)})


# ============================================================================
# 管理员端点
# ============================================================================

@auth_api_bp.route('/admin/users', methods=['GET'])
@require_auth
def list_users():
    """列出用户
    
    获取用户列表（需要管理员权限）。
    
    Headers:
        - Authorization: Bearer <access_token>
        
    Query Parameters:
        - status (str, optional): 状态过滤
        - role (str, optional): 角色过滤
        - search (str, optional): 搜索关键词
        - offset (int, optional): 偏移量，默认0
        - limit (int, optional): 限制数量，默认100
        
    Returns:
        - users: 用户列表
        - total: 总数
        
    Raises:
        - 401: 未认证
        - 403: 无权限
        - 500: 获取失败
    """
    try:
        # 验证管理员权限
        auth_service = get_auth_service_instance()
        user = auth_service.get_user_by_id(request.current_user_id)
        
        if not user or not user.is_superuser:
            return error_response("无权限访问", 403, "PermissionDenied")
        
        status = request.args.get('status')
        role = request.args.get('role')
        search = request.args.get('search')
        offset = int(request.args.get('offset', 0))
        limit = min(int(request.args.get('limit', 100)), 500)
        
        users, total = auth_service.repository.list_users(
            offset=offset,
            limit=limit,
            status=status,
            role=role,
            search=search
        )
        
        return success_response({
            'users': [u.to_dict() for u in users],
            'total': total,
            'offset': offset,
            'limit': limit
        })

    except Exception as e:
        logger.error(f"List users failed: {str(e)}")
        return error_response("获取用户列表失败", 500, "ListUsersError", {"error": str(e)})


@auth_api_bp.route('/admin/security-statistics', methods=['GET'])
@require_auth
def get_admin_security_statistics():
    """获取系统安全统计
    
    获取系统级别的安全统计数据（需要管理员权限）。
    
    Headers:
        - Authorization: Bearer <access_token>
        
    Query Parameters:
        - days (int, optional): 统计天数，默认30
        
    Returns:
        - total_events: 总事件数
        - unresolved: 未解决数
        - resolved: 已解决数
        - by_type: 按类型统计
        - by_severity: 按严重程度统计
        
    Raises:
        - 401: 未认证
        - 403: 无权限
        - 500: 获取失败
    """
    try:
        # 验证管理员权限
        auth_service = get_auth_service_instance()
        user = auth_service.get_user_by_id(request.current_user_id)
        
        if not user or not user.is_superuser:
            return error_response("无权限访问", 403, "PermissionDenied")
        
        days = int(request.args.get('days', 30))
        
        statistics = auth_service.get_security_statistics(days=days)
        
        return success_response(statistics)

    except Exception as e:
        logger.error(f"Get security statistics failed: {str(e)}")
        return error_response("获取安全统计失败", 500, "GetStatisticsError", {"error": str(e)})


@auth_api_bp.route('/admin/users/<user_id>/lock', methods=['POST'])
@require_auth
def lock_user(user_id):
    """锁定用户
    
    锁定指定用户账户（需要管理员权限）。
    
    Headers:
        - Authorization: Bearer <access_token>
        
    Path Parameters:
        - user_id: 用户ID
        
    Request Body:
        - lockout_minutes (int, optional): 锁定分钟数，默认30
        
    Returns:
        - message: 成功消息
        
    Raises:
        - 401: 未认证
        - 403: 无权限
        - 404: 用户不存在
        - 500: 操作失败
    """
    try:
        # 验证管理员权限
        auth_service = get_auth_service_instance()
        admin_user = auth_service.get_user_by_id(request.current_user_id)
        
        if not admin_user or not admin_user.is_superuser:
            return error_response("无权限访问", 403, "PermissionDenied")
        
        # 不能锁定自己
        if user_id == request.current_user_id:
            return error_response("不能锁定自己的账户", 400, "ValidationError")
        
        data = request.get_json() or {}
        lockout_minutes = int(data.get('lockout_minutes', 30))
        
        target_user = auth_service.get_user_by_id(user_id)
        if not target_user:
            return error_response("用户不存在", 404, "UserNotFound")
        
        auth_service.repository.lock_user(user_id, lockout_minutes)
        
        return success_response({
            'message': f'用户已锁定 {lockout_minutes} 分钟'
        })

    except Exception as e:
        logger.error(f"Lock user failed: {str(e)}")
        return error_response("锁定用户失败", 500, "LockUserError", {"error": str(e)})


@auth_api_bp.route('/admin/users/<user_id>/unlock', methods=['POST'])
@require_auth
def unlock_user(user_id):
    """解锁用户
    
    解锁指定用户账户（需要管理员权限）。
    
    Headers:
        - Authorization: Bearer <access_token>
        
    Path Parameters:
        - user_id: 用户ID
        
    Returns:
        - message: 成功消息
        
    Raises:
        - 401: 未认证
        - 403: 无权限
        - 404: 用户不存在
        - 500: 操作失败
    """
    try:
        # 验证管理员权限
        auth_service = get_auth_service_instance()
        admin_user = auth_service.get_user_by_id(request.current_user_id)
        
        if not admin_user or not admin_user.is_superuser:
            return error_response("无权限访问", 403, "PermissionDenied")
        
        target_user = auth_service.get_user_by_id(user_id)
        if not target_user:
            return error_response("用户不存在", 404, "UserNotFound")
        
        auth_service.repository.unlock_user(user_id)
        
        return success_response({
            'message': '用户已解锁'
        })

    except Exception as e:
        logger.error(f"Unlock user failed: {str(e)}")
        return error_response("解锁用户失败", 500, "UnlockUserError", {"error": str(e)})


# ============================================================================
# 健康检查端点
# ============================================================================

@auth_api_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查
    
    检查认证服务健康状态。
    
    Returns:
        - status: 服务状态
        - service: 服务名称
        - timestamp: 时间戳
        
    Raises:
        - 500: 服务不健康
    """
    try:
        from datetime import datetime
        
        # 尝试获取服务实例来验证服务可用
        auth_service = get_auth_service_instance()
        
        return success_response({
            'status': 'healthy',
            'service': 'auth_service',
            'timestamp': datetime.utcnow().isoformat(),
            'features': [
                'user_authentication',
                'session_management',
                'risk_assessment',
                'agent_memory',
                'security_analysis'
            ]
        })

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return error_response("服务不健康", 500, "ServiceUnhealthy", {"error": str(e)})


# ============================================================================
# 兼容性端点（保持向后兼容）
# ============================================================================

# 以下为保持向后兼容的别名路由

@auth_api_bp.route('/verify', methods=['POST'])
@require_auth
def verify_token_endpoint():
    """验证令牌
    
    验证访问令牌是否有效。
    
    Headers:
        - Authorization: Bearer <access_token>
        
    Returns:
        - valid: 是否有效
        - user_id: 用户ID
        - session_id: 会话ID
        
    Raises:
        - 401: 令牌无效
    """
    return success_response({
        'valid': True,
        'user_id': request.current_user_id,
        'session_id': request.current_session_id
    })
