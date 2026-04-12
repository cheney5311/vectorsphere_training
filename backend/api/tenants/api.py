"""租户API模块

提供完整的租户管理和计费相关的API接口，包括：
- 租户 CRUD 操作
- 租户用户管理
- 资源配额和使用情况
- 计费和发票管理
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass

from flask import Blueprint, request
from flask_jwt_extended import jwt_required, get_jwt_identity

from backend.core.exceptions import ValidationError, BusinessLogicError, AuthorizationError, TenantError
from backend.utils.response import success_response, error_response, paginated_response

logger = logging.getLogger(__name__)

# 创建租户蓝图
tenants_bp = Blueprint('tenants', __name__, url_prefix='/api/v1/tenants')


# ============================================================================
# 数据类定义（当模块不可用时的回退）
# ============================================================================

@dataclass
class UsageRecord:
    """使用记录数据类"""
    id: str
    user_id: str
    resource_type: str
    resource_id: str
    usage_amount: float
    unit: str
    start_time: datetime
    end_time: datetime
    cost: Decimal = Decimal('0')
    billing_rule_id: str = ""


@dataclass
class Payment:
    """支付记录数据类"""
    id: str
    invoice_id: str
    user_id: str
    amount: Decimal
    currency: str
    payment_method: str
    transaction_id: str = ""
    status: str = "pending"
    payment_date: Optional[datetime] = None


# ============================================================================
# 服务获取器
# ============================================================================

def _get_tenant_service():
    """获取租户服务"""
    try:
        from backend.services.tenant_service import TenantService
        return TenantService()
    except ImportError:
        logger.warning("TenantService not available")
        return None


def _get_billing_service():
    """获取计费服务"""
    try:
        from modules.tenants.services.billing_service import get_billing_service
        return get_billing_service()
    except ImportError:
        logger.warning("BillingService not available")
        return None


def _get_usage_record_class():
    """获取 UsageRecord 类"""
    try:
        from modules.tenants.models.billing import UsageRecord as ModuleUsageRecord
        return ModuleUsageRecord
    except ImportError:
        return UsageRecord


def _get_payment_class():
    """获取 Payment 类"""
    try:
        from modules.tenants.models.billing import Payment as ModulePayment
        return ModulePayment
    except ImportError:
        return Payment


# ============================================================================
# 租户管理 API
# ============================================================================

@tenants_bp.route('', methods=['POST'])
@jwt_required()
def create_tenant():
    """创建租户
    
    Request Body:
        {
            "name": "string (required)",
            "display_name": "string",
            "description": "string",
            "settings": {}
        }
    
    Returns:
        {
            "tenant_id": "string",
            "name": "string",
            "display_name": "string",
            "description": "string",
            "status": "string",
            "created_at": "string"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        if 'name' not in data:
            return error_response("缺少必需字段: name", 400)
        
        tenant_service = _get_tenant_service()
        if not tenant_service:
            return error_response("租户服务不可用", 503, "SERVICE_UNAVAILABLE")
        
        result = tenant_service.create_tenant(data, current_user_id)
        return success_response(result, "租户创建成功", 201)
        
    except ValidationError as e:
        return error_response(str(e), 400, "VALIDATION_ERROR")
    except TenantError as e:
        return error_response(str(e), 500, "TENANT_ERROR")
    except Exception as e:
        logger.exception(f"创建租户失败: {e}")
        return error_response(f"创建租户失败: {str(e)}", 500, "INTERNAL_ERROR")


@tenants_bp.route('', methods=['GET'])
@jwt_required()
def list_tenants():
    """获取租户列表
    
    Query Parameters:
        - page: 页码 (默认 1)
        - page_size: 每页大小 (默认 20)
    
    Returns:
        {
            "tenants": [],
            "total": "integer",
            "page": "integer",
            "page_size": "integer",
            "total_pages": "integer"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 20, type=int)
        
        tenant_service = _get_tenant_service()
        if not tenant_service:
            return error_response("租户服务不可用", 503, "SERVICE_UNAVAILABLE")
        
        result = tenant_service.list_tenants(current_user_id, page, page_size)
        return paginated_response(
            result['tenants'],
            result['total'],
            result['page'],
            result['page_size'],
            "获取租户列表成功"
        )
        
    except TenantError as e:
        return error_response(str(e), 500, "TENANT_ERROR")
    except Exception as e:
        logger.exception(f"获取租户列表失败: {e}")
        return error_response(f"获取租户列表失败: {str(e)}", 500, "INTERNAL_ERROR")


@tenants_bp.route('/<tenant_id>', methods=['GET'])
@jwt_required()
def get_tenant(tenant_id: str):
    """获取租户详情
    
    Path Parameters:
        - tenant_id: 租户ID
    
    Returns:
        {
            "tenant_id": "string",
            "name": "string",
            "display_name": "string",
            "description": "string",
            "status": "string",
            "created_at": "string",
            "updated_at": "string",
            "settings": {},
            "user_role": "string",
            "statistics": {}
        }
    """
    try:
        current_user_id = get_jwt_identity()
        
        tenant_service = _get_tenant_service()
        if not tenant_service:
            return error_response("租户服务不可用", 503, "SERVICE_UNAVAILABLE")
        
        result = tenant_service.get_tenant(tenant_id, current_user_id)
        return success_response(result, "获取租户详情成功")
        
    except AuthorizationError as e:
        return error_response(str(e), 403, "FORBIDDEN")
    except TenantError as e:
        return error_response(str(e), 404, "NOT_FOUND")
    except Exception as e:
        logger.exception(f"获取租户详情失败: {e}")
        return error_response(f"获取租户详情失败: {str(e)}", 500, "INTERNAL_ERROR")


@tenants_bp.route('/<tenant_id>', methods=['PUT'])
@jwt_required()
def update_tenant(tenant_id: str):
    """更新租户信息
    
    Path Parameters:
        - tenant_id: 租户ID
    
    Request Body:
        {
            "display_name": "string",
            "description": "string",
            "settings": {}
        }
    
    Returns:
        {
            "tenant_id": "string",
            "name": "string",
            "display_name": "string",
            "description": "string",
            "status": "string",
            "updated_at": "string"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        tenant_service = _get_tenant_service()
        if not tenant_service:
            return error_response("租户服务不可用", 503, "SERVICE_UNAVAILABLE")
        
        result = tenant_service.update_tenant(tenant_id, data, current_user_id)
        return success_response(result, "租户更新成功")
        
    except ValidationError as e:
        return error_response(str(e), 400, "VALIDATION_ERROR")
    except AuthorizationError as e:
        return error_response(str(e), 403, "FORBIDDEN")
    except TenantError as e:
        return error_response(str(e), 500, "TENANT_ERROR")
    except Exception as e:
        logger.exception(f"更新租户失败: {e}")
        return error_response(f"更新租户失败: {str(e)}", 500, "INTERNAL_ERROR")


@tenants_bp.route('/<tenant_id>', methods=['DELETE'])
@jwt_required()
def delete_tenant(tenant_id: str):
    """删除租户
    
    Path Parameters:
        - tenant_id: 租户ID
    
    Returns:
        {
            "tenant_id": "string",
            "deleted_at": "string"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        
        tenant_service = _get_tenant_service()
        if not tenant_service:
            return error_response("租户服务不可用", 503, "SERVICE_UNAVAILABLE")
        
        result = tenant_service.delete_tenant(tenant_id, current_user_id)
        return success_response(result, "租户删除成功")
        
    except ValidationError as e:
        return error_response(str(e), 400, "VALIDATION_ERROR")
    except AuthorizationError as e:
        return error_response(str(e), 403, "FORBIDDEN")
    except TenantError as e:
        return error_response(str(e), 500, "TENANT_ERROR")
    except Exception as e:
        logger.exception(f"删除租户失败: {e}")
        return error_response(f"删除租户失败: {str(e)}", 500, "INTERNAL_ERROR")


@tenants_bp.route('/<tenant_id>/status', methods=['PUT'])
@jwt_required()
def update_tenant_status(tenant_id: str):
    """更新租户状态
    
    Path Parameters:
        - tenant_id: 租户ID
    
    Request Body:
        {
            "status": "string"  // active, suspended, deleted
        }
    
    Returns:
        {
            "tenant_id": "string",
            "status": "string",
            "updated_at": "string"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data or 'status' not in data:
            return error_response("缺少必需字段: status", 400)
        
        tenant_service = _get_tenant_service()
        if not tenant_service:
            return error_response("租户服务不可用", 503, "SERVICE_UNAVAILABLE")
        
        result = tenant_service.update_tenant_status(tenant_id, data['status'], current_user_id)
        return success_response(result, "租户状态更新成功")
        
    except ValidationError as e:
        return error_response(str(e), 400, "VALIDATION_ERROR")
    except AuthorizationError as e:
        return error_response(str(e), 403, "FORBIDDEN")
    except TenantError as e:
        return error_response(str(e), 500, "TENANT_ERROR")
    except Exception as e:
        logger.exception(f"更新租户状态失败: {e}")
        return error_response(f"更新租户状态失败: {str(e)}", 500, "INTERNAL_ERROR")


# ============================================================================
# 租户用户管理 API
# ============================================================================

@tenants_bp.route('/<tenant_id>/users', methods=['GET'])
@jwt_required()
def list_tenant_users(tenant_id: str):
    """获取租户用户列表
    
    Path Parameters:
        - tenant_id: 租户ID
    
    Query Parameters:
        - page: 页码 (默认 1)
        - page_size: 每页大小 (默认 20)
    
    Returns:
        {
            "users": [],
            "total": "integer",
            "page": "integer",
            "page_size": "integer",
            "total_pages": "integer"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 20, type=int)
        
        tenant_service = _get_tenant_service()
        if not tenant_service:
            return error_response("租户服务不可用", 503, "SERVICE_UNAVAILABLE")
        
        result = tenant_service.list_tenant_users(tenant_id, current_user_id, page, page_size)
        return paginated_response(
            result['users'],
            result['total'],
            result['page'],
            result['page_size'],
            "获取租户用户列表成功"
        )
        
    except AuthorizationError as e:
        return error_response(str(e), 403, "FORBIDDEN")
    except TenantError as e:
        return error_response(str(e), 500, "TENANT_ERROR")
    except Exception as e:
        logger.exception(f"获取租户用户列表失败: {e}")
        return error_response(f"获取租户用户列表失败: {str(e)}", 500, "INTERNAL_ERROR")


@tenants_bp.route('/<tenant_id>/users', methods=['POST'])
@jwt_required()
def add_tenant_user(tenant_id: str):
    """添加租户用户
    
    Path Parameters:
        - tenant_id: 租户ID
    
    Request Body:
        {
            "user_email": "string (required)",
            "role": "string (required)"  // admin, member, owner
        }
    
    Returns:
        {
            "tenant_id": "string",
            "user_id": "string",
            "user_email": "string",
            "role": "string",
            "added_at": "string"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        required_fields = ['user_email', 'role']
        for field in required_fields:
            if field not in data:
                return error_response(f"缺少必需字段: {field}", 400)
        
        tenant_service = _get_tenant_service()
        if not tenant_service:
            return error_response("租户服务不可用", 503, "SERVICE_UNAVAILABLE")
        
        result = tenant_service.add_tenant_user(
            tenant_id, 
            data['user_email'], 
            data['role'], 
            current_user_id
        )
        return success_response(result, "用户添加成功", 201)
        
    except ValidationError as e:
        return error_response(str(e), 400, "VALIDATION_ERROR")
    except AuthorizationError as e:
        return error_response(str(e), 403, "FORBIDDEN")
    except TenantError as e:
        return error_response(str(e), 500, "TENANT_ERROR")
    except Exception as e:
        logger.exception(f"添加租户用户失败: {e}")
        return error_response(f"添加租户用户失败: {str(e)}", 500, "INTERNAL_ERROR")


@tenants_bp.route('/<tenant_id>/users/<user_id>', methods=['DELETE'])
@jwt_required()
def remove_tenant_user(tenant_id: str, user_id: str):
    """移除租户用户
    
    Path Parameters:
        - tenant_id: 租户ID
        - user_id: 用户ID
    
    Returns:
        {
            "tenant_id": "string",
            "user_id": "string",
            "removed_at": "string"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        
        tenant_service = _get_tenant_service()
        if not tenant_service:
            return error_response("租户服务不可用", 503, "SERVICE_UNAVAILABLE")
        
        result = tenant_service.remove_tenant_user(tenant_id, user_id, current_user_id)
        return success_response(result, "用户移除成功")
        
    except ValidationError as e:
        return error_response(str(e), 400, "VALIDATION_ERROR")
    except AuthorizationError as e:
        return error_response(str(e), 403, "FORBIDDEN")
    except TenantError as e:
        return error_response(str(e), 500, "TENANT_ERROR")
    except Exception as e:
        logger.exception(f"移除租户用户失败: {e}")
        return error_response(f"移除租户用户失败: {str(e)}", 500, "INTERNAL_ERROR")


# ============================================================================
# 资源使用情况 API
# ============================================================================

@tenants_bp.route('/<tenant_id>/resources', methods=['GET'])
@jwt_required()
def get_tenant_resource_usage(tenant_id: str):
    """获取租户资源使用情况
    
    Path Parameters:
        - tenant_id: 租户ID
    
    Returns:
        {
            "tenant_id": "string",
            "resource_usage": {
                "max_users": {"current": 5, "limit": 10, "percentage": 50, "remaining": 5},
                ...
            },
            "last_updated": "string"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        
        tenant_service = _get_tenant_service()
        if not tenant_service:
            return error_response("租户服务不可用", 503, "SERVICE_UNAVAILABLE")
        
        result = tenant_service.get_tenant_resource_usage(tenant_id, current_user_id)
        return success_response(result, "获取资源使用情况成功")
        
    except AuthorizationError as e:
        return error_response(str(e), 403, "FORBIDDEN")
    except TenantError as e:
        return error_response(str(e), 500, "TENANT_ERROR")
    except Exception as e:
        logger.exception(f"获取资源使用情况失败: {e}")
        return error_response(f"获取资源使用情况失败: {str(e)}", 500, "INTERNAL_ERROR")


# ============================================================================
# 计费管理 API
# ============================================================================

@tenants_bp.route('/billing/usage', methods=['POST'])
@jwt_required()
def record_resource_usage():
    """记录资源使用
    
    Request Body:
        {
            "user_id": "string",
            "resource_type": "string (required)",
            "resource_id": "string (required)",
            "usage_amount": "number (required)",
            "unit": "string",
            "start_time": "string (required)",
            "end_time": "string (required)"
        }
    
    Returns:
        {
            "usage_id": "string",
            "cost": "number"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        # 验证必需字段
        required_fields = ['resource_type', 'resource_id', 'usage_amount', 'start_time', 'end_time']
        for field in required_fields:
            if field not in data:
                return error_response(f"缺少必需字段: {field}", 400)
        
        billing_service = _get_billing_service()
        if not billing_service:
            return error_response("计费服务不可用", 503, "SERVICE_UNAVAILABLE")
        
        # 获取合适的 UsageRecord 类
        UsageRecordClass = _get_usage_record_class()
        
        # 创建使用记录
        usage_record = UsageRecordClass(
            id=f"usage_{int(datetime.now().timestamp())}",
            user_id=data.get('user_id', current_user_id),
            resource_type=data['resource_type'],
            resource_id=data['resource_id'],
            usage_amount=data['usage_amount'],
            unit=data.get('unit', 'unit'),
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time'])
        )
        
        # 记录使用
        success = billing_service.record_usage(usage_record)
        
        if success:
            cost = float(getattr(usage_record, 'cost', 0))
            return success_response({
                'usage_id': usage_record.id,
                'cost': cost
            }, "资源使用记录成功", 201)
        else:
            return error_response("资源使用记录失败", 500)
            
    except ValidationError as e:
        return error_response(str(e), 400, "VALIDATION_ERROR")
    except ValueError as e:
        return error_response(f"数据格式错误: {str(e)}", 400, "VALUE_ERROR")
    except Exception as e:
        logger.exception(f"记录资源使用失败: {e}")
        return error_response(f"记录资源使用失败: {str(e)}", 500, "INTERNAL_ERROR")


@tenants_bp.route('/billing/invoices', methods=['POST'])
@jwt_required()
def generate_invoice():
    """生成发票
    
    Request Body:
        {
            "user_id": "string",
            "period_start": "string (required)",
            "period_end": "string (required)"
        }
    
    Returns:
        {
            "invoice_id": "string",
            "total_amount": "number",
            "items": []
        }
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        # 验证必需字段
        required_fields = ['period_start', 'period_end']
        for field in required_fields:
            if field not in data:
                return error_response(f"缺少必需字段: {field}", 400)
        
        billing_service = _get_billing_service()
        if not billing_service:
            return error_response("计费服务不可用", 503, "SERVICE_UNAVAILABLE")
        
        # 生成发票
        user_id = data.get('user_id', current_user_id)
        period_start = datetime.fromisoformat(data['period_start'])
        period_end = datetime.fromisoformat(data['period_end'])
        
        invoice = billing_service.generate_invoice(user_id, period_start, period_end)
        
        if invoice:
            items = []
            for item in getattr(invoice, 'items', []):
                items.append({
                    'description': getattr(item, 'description', ''),
                    'quantity': float(getattr(item, 'quantity', 0)),
                    'unit_price': float(getattr(item, 'unit_price', 0)),
                    'total_amount': float(getattr(item, 'total_amount', 0))
                })
            
            return success_response({
                'invoice_id': invoice.id,
                'total_amount': float(getattr(invoice, 'total_amount', 0)),
                'items': items
            }, "发票生成成功", 201)
        else:
            return error_response("发票生成失败", 500)
            
    except ValidationError as e:
        return error_response(str(e), 400, "VALIDATION_ERROR")
    except ValueError as e:
        return error_response(f"数据格式错误: {str(e)}", 400, "VALUE_ERROR")
    except Exception as e:
        logger.exception(f"生成发票失败: {e}")
        return error_response(f"生成发票失败: {str(e)}", 500, "INTERNAL_ERROR")


@tenants_bp.route('/billing/invoices/<invoice_id>', methods=['GET'])
@jwt_required()
def get_invoice(invoice_id: str):
    """获取发票详情
    
    Path Parameters:
        - invoice_id: 发票ID
    
    Returns:
        {
            "invoice_id": "string",
            "user_id": "string",
            "total_amount": "number",
            "status": "string",
            "items": [],
            "created_at": "string"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        
        billing_service = _get_billing_service()
        if not billing_service:
            return error_response("计费服务不可用", 503, "SERVICE_UNAVAILABLE")
        
        # 获取发票
        if hasattr(billing_service, 'get_invoice'):
            invoice = billing_service.get_invoice(invoice_id)
        else:
            return error_response("该功能暂不可用", 501, "NOT_IMPLEMENTED")
        
        if invoice:
            items = []
            for item in getattr(invoice, 'items', []):
                items.append({
                    'description': getattr(item, 'description', ''),
                    'quantity': float(getattr(item, 'quantity', 0)),
                    'unit_price': float(getattr(item, 'unit_price', 0)),
                    'total_amount': float(getattr(item, 'total_amount', 0))
                })
            
            return success_response({
                'invoice_id': invoice.id,
                'user_id': getattr(invoice, 'user_id', ''),
                'total_amount': float(getattr(invoice, 'total_amount', 0)),
                'status': getattr(invoice, 'status', 'unknown'),
                'items': items,
                'created_at': getattr(invoice, 'created_at', datetime.utcnow()).isoformat()
            }, "获取发票详情成功")
        else:
            return error_response("发票不存在", 404, "NOT_FOUND")
            
    except Exception as e:
        logger.exception(f"获取发票详情失败: {e}")
        return error_response(f"获取发票详情失败: {str(e)}", 500, "INTERNAL_ERROR")


@tenants_bp.route('/billing/invoices/<invoice_id>/pay', methods=['POST'])
@jwt_required()
def process_payment(invoice_id: str):
    """处理支付
    
    Path Parameters:
        - invoice_id: 发票ID
    
    Request Body:
        {
            "amount": "number (required)",
            "currency": "string (required)",
            "payment_method": "string (required)",
            "transaction_id": "string"
        }
    
    Returns:
        {
            "payment_id": "string",
            "status": "string"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response("请求数据不能为空", 400)
        
        # 验证必需字段
        required_fields = ['amount', 'currency', 'payment_method']
        for field in required_fields:
            if field not in data:
                return error_response(f"缺少必需字段: {field}", 400)
        
        billing_service = _get_billing_service()
        if not billing_service:
            return error_response("计费服务不可用", 503, "SERVICE_UNAVAILABLE")
        
        # 获取合适的 Payment 类
        PaymentClass = _get_payment_class()
        
        # 创建支付记录
        payment = PaymentClass(
            id=f"payment_{int(datetime.now().timestamp())}",
            invoice_id=invoice_id,
            user_id=current_user_id,
            amount=Decimal(str(data['amount'])),
            currency=data['currency'],
            payment_method=data['payment_method'],
            transaction_id=data.get('transaction_id', '')
        )
        
        # 处理支付
        success = billing_service.process_payment(payment)
        
        if success:
            # 获取状态值
            status = getattr(payment, 'status', 'pending')
            if hasattr(status, 'value'):
                status = status.value
            
            return success_response({
                'payment_id': payment.id,
                'status': status
            }, "支付处理成功", 201)
        else:
            return error_response("支付处理失败", 500)
            
    except ValidationError as e:
        return error_response(str(e), 400, "VALIDATION_ERROR")
    except ValueError as e:
        return error_response(f"数据格式错误: {str(e)}", 400, "VALUE_ERROR")
    except Exception as e:
        logger.exception(f"处理支付失败: {e}")
        return error_response(f"处理支付失败: {str(e)}", 500, "INTERNAL_ERROR")


@tenants_bp.route('/billing/summary', methods=['GET'])
@jwt_required()
def get_billing_summary():
    """获取计费摘要
    
    Query Parameters:
        - user_id: 用户ID
        - period_start: 开始时间 (required)
        - period_end: 结束时间 (required)
    
    Returns:
        {
            "user_id": "string",
            "period_start": "string",
            "period_end": "string",
            "total_cost": "number",
            "resource_summary": {}
        }
    """
    try:
        current_user_id = get_jwt_identity()
        
        # 获取查询参数
        user_id = request.args.get('user_id', current_user_id)
        period_start_str = request.args.get('period_start')
        period_end_str = request.args.get('period_end')
        
        if not period_start_str or not period_end_str:
            return error_response("缺少必需的查询参数: period_start, period_end", 400)
        
        billing_service = _get_billing_service()
        if not billing_service:
            return error_response("计费服务不可用", 503, "SERVICE_UNAVAILABLE")
        
        # 解析时间参数
        period_start = datetime.fromisoformat(period_start_str)
        period_end = datetime.fromisoformat(period_end_str)
        
        # 获取计费摘要
        summary = billing_service.get_user_billing_summary(user_id, period_start, period_end)
        
        return success_response(summary, "获取计费摘要成功")
        
    except ValidationError as e:
        return error_response(str(e), 400, "VALIDATION_ERROR")
    except ValueError as e:
        return error_response(f"数据格式错误: {str(e)}", 400, "VALUE_ERROR")
    except Exception as e:
        logger.exception(f"获取计费摘要失败: {e}")
        return error_response(f"获取计费摘要失败: {str(e)}", 500, "INTERNAL_ERROR")


@tenants_bp.route('/billing/dashboard', methods=['GET'])
@jwt_required()
def get_billing_dashboard():
    """获取计费仪表板数据
    
    Returns:
        {
            "current_month_revenue": "number",
            "total_invoices": "integer",
            "paid_invoices": "integer",
            "pending_invoices": "integer",
            "overdue_invoices": "integer"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        
        billing_service = _get_billing_service()
        if not billing_service:
            return error_response("计费服务不可用", 503, "SERVICE_UNAVAILABLE")
        
        # 获取计费仪表板数据
        dashboard_data = billing_service.get_billing_dashboard_data()
        
        return success_response(dashboard_data, "获取计费仪表板数据成功")
        
    except Exception as e:
        logger.exception(f"获取计费仪表板数据失败: {e}")
        return error_response(f"获取计费仪表板数据失败: {str(e)}", 500, "INTERNAL_ERROR")


@tenants_bp.route('/billing/history', methods=['GET'])
@jwt_required()
def get_billing_history():
    """获取计费历史记录
    
    Query Parameters:
        - user_id: 用户ID
        - page: 页码 (默认 1)
        - page_size: 每页大小 (默认 20)
        - start_date: 开始日期
        - end_date: 结束日期
    
    Returns:
        {
            "records": [],
            "total": "integer",
            "page": "integer",
            "page_size": "integer"
        }
    """
    try:
        current_user_id = get_jwt_identity()
        
        user_id = request.args.get('user_id', current_user_id)
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 20, type=int)
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        
        billing_service = _get_billing_service()
        if not billing_service:
            return error_response("计费服务不可用", 503, "SERVICE_UNAVAILABLE")
        
        # 获取计费历史
        if hasattr(billing_service, 'get_billing_history'):
            start_date = datetime.fromisoformat(start_date_str) if start_date_str else None
            end_date = datetime.fromisoformat(end_date_str) if end_date_str else None
            
            result = billing_service.get_billing_history(
                user_id, 
                page=page, 
                page_size=page_size,
                start_date=start_date,
                end_date=end_date
            )
            return paginated_response(
                result.get('records', []),
                result.get('total', 0),
                page,
                page_size,
                "获取计费历史成功"
            )
        else:
            return error_response("该功能暂不可用", 501, "NOT_IMPLEMENTED")
        
    except ValueError as e:
        return error_response(f"数据格式错误: {str(e)}", 400, "VALUE_ERROR")
    except Exception as e:
        logger.exception(f"获取计费历史失败: {e}")
        return error_response(f"获取计费历史失败: {str(e)}", 500, "INTERNAL_ERROR")
