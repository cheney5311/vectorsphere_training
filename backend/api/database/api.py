"""数据库管理API接口

提供数据库相关的REST API接口，包括健康检查、表管理、连接池监控、
数据迁移、备份恢复、性能监控等功能。

所有接口都需要JWT认证（除健康检查外）。

API端点:
    健康检查:
        - GET /health: 数据库健康检查（公开）
        - GET /health/detailed: 详细健康检查
    
    表管理:
        - GET /tables: 获取所有表信息
        - GET /tables/<table_name>: 获取指定表详情
        - GET /tables/<table_name>/count: 获取表记录数
        - POST /tables/sync: 同步表结构
        - POST /tables/<table_name>/truncate: 清空表数据
    
    连接池管理:
        - GET /pool/stats: 获取连接池统计
        - POST /pool/reset: 重置连接池
    
    数据统计:
        - GET /stats: 获取数据库统计信息
        - GET /stats/tables: 获取各表统计
        - GET /stats/size: 获取数据库大小
    
    查询执行:
        - POST /query: 执行只读查询
        - POST /query/explain: 分析查询计划
    
    维护操作:
        - POST /maintenance/vacuum: 执行清理操作
        - POST /maintenance/analyze: 更新统计信息
        - GET /maintenance/locks: 获取当前锁信息
    
    备份恢复:
        - POST /backup: 创建备份
        - GET /backup/list: 获取备份列表
        - POST /restore: 恢复备份
"""

import sys
import os
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.utils.response import success_response, error_response
from backend.core.exceptions import ValidationError, BusinessLogicError

logger = logging.getLogger(__name__)

# 创建蓝图
database_bp = Blueprint('database', __name__, url_prefix='/api/v1/database')


# ============================================================================
# 健康检查API
# ============================================================================

@database_bp.route('/health', methods=['GET'])
def health_check():
    """数据库健康检查（公开接口）
    
    检查数据库连接是否正常。不需要认证。
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "健康检查成功",
                "data": {
                    "status": "healthy",
                    "database_type": "postgresql",
                    "connection_available": true,
                    "timestamp": "2024-01-01T00:00:00"
                }
            }
        失败 (503):
            {
                "code": 503,
                "message": "数据库连接异常",
                "error_type": "DATABASE_UNHEALTHY"
            }
    
    示例:
        curl -X GET "http://localhost:5000/api/v1/database/health"
    """
    try:
        from backend.services.database_management_service import DatabaseManagementService
        
        service = DatabaseManagementService()
        health_info = service.check_health()
        
        if health_info.get('status') == 'healthy':
            return success_response(
                data=health_info,
                message="健康检查成功"
            )
        else:
            return error_response(
                message="数据库连接异常",
                code=503,
                error_type="DATABASE_UNHEALTHY"
            ), 503
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return error_response(
            message=f"健康检查失败: {str(e)}",
            code=500,
            error_type="HEALTH_CHECK_ERROR"
        ), 500


@database_bp.route('/health/detailed', methods=['GET'])
@jwt_required()
def detailed_health_check():
    """详细健康检查
    
    获取数据库的详细健康状态信息，包括连接池状态、响应时间等。
    需要JWT认证。
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "详细健康检查成功",
                "data": {
                    "status": "healthy",
                    "database_type": "postgresql",
                    "connection": {
                        "available": true,
                        "response_time_ms": 5.2
                    },
                    "pool": {
                        "size": 10,
                        "checked_in": 8,
                        "checked_out": 2,
                        "overflow": 0
                    },
                    "tables": {
                        "total_count": 25,
                        "healthy_count": 25
                    },
                    "timestamp": "2024-01-01T00:00:00"
                }
            }
    
    示例:
        curl -X GET "http://localhost:5000/api/v1/database/health/detailed" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        from backend.services.database_management_service import DatabaseManagementService
        
        service = DatabaseManagementService()
        detailed_health = service.get_detailed_health()
        
        return success_response(
            data=detailed_health,
            message="详细健康检查成功"
        )
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return error_response(
            message=f"详细健康检查失败: {str(e)}",
            code=500,
            error_type="HEALTH_CHECK_ERROR"
        ), 500


# ============================================================================
# 表管理API
# ============================================================================

@database_bp.route('/tables', methods=['GET'])
@jwt_required()
def list_tables():
    """获取所有表信息
    
    获取数据库中所有表的基本信息列表。
    需要JWT认证。
    
    查询参数:
        include_system (bool): 是否包含系统表 (可选，默认false)
        search (str): 表名搜索关键词 (可选)
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "获取表信息成功",
                "data": {
                    "tables": [
                        {
                            "name": "users",
                            "schema": "public",
                            "columns_count": 10,
                            "has_primary_key": true,
                            "indexes_count": 3
                        }
                    ],
                    "total_count": 25
                }
            }
    
    示例:
        curl -X GET "http://localhost:5000/api/v1/database/tables" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        from backend.services.database_management_service import DatabaseManagementService
        
        include_system = request.args.get('include_system', 'false').lower() == 'true'
        search = request.args.get('search')
        
        service = DatabaseManagementService()
        tables_info = service.list_tables(
            include_system=include_system,
            search=search
        )
        
        return success_response(
            data=tables_info,
            message="获取表信息成功"
        )
        
    except Exception as e:
        logger.error(f"List tables failed: {e}")
        return error_response(
            message=f"获取表信息失败: {str(e)}",
            code=500,
            error_type="LIST_TABLES_ERROR"
        ), 500


@database_bp.route('/tables/<table_name>', methods=['GET'])
@jwt_required()
def get_table_details(table_name: str):
    """获取指定表的详细信息
    
    获取指定表的完整结构信息，包括列定义、索引、约束等。
    需要JWT认证。
    
    路径参数:
        table_name (str): 表名
            - 必填
            - 示例: "users"
    
    查询参数:
        include_sample_data (bool): 是否包含样本数据 (可选，默认false)
        sample_limit (int): 样本数据条数 (可选，默认5)
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "获取表详情成功",
                "data": {
                    "name": "users",
                    "schema": "public",
                    "columns": [
                        {
                            "name": "id",
                            "type": "UUID",
                            "nullable": false,
                            "primary_key": true,
                            "default": "uuid_generate_v4()"
                        }
                    ],
                    "indexes": [
                        {
                            "name": "ix_users_email",
                            "columns": ["email"],
                            "unique": true
                        }
                    ],
                    "foreign_keys": [],
                    "constraints": [],
                    "row_count": 1000,
                    "size_bytes": 102400,
                    "sample_data": []
                }
            }
        失败:
            - 404: 表不存在
    
    示例:
        curl -X GET "http://localhost:5000/api/v1/database/tables/users?include_sample_data=true" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        from backend.services.database_management_service import DatabaseManagementService
        
        include_sample = request.args.get('include_sample_data', 'false').lower() == 'true'
        sample_limit = request.args.get('sample_limit', 5, type=int)
        
        service = DatabaseManagementService()
        table_details = service.get_table_details(
            table_name=table_name,
            include_sample_data=include_sample,
            sample_limit=sample_limit
        )
        
        if not table_details:
            return error_response(
                message=f"表 {table_name} 不存在",
                code=404,
                error_type="TABLE_NOT_FOUND"
            ), 404
        
        return success_response(
            data=table_details,
            message="获取表详情成功"
        )
        
    except Exception as e:
        logger.error(f"Get table details failed: {e}")
        return error_response(
            message=f"获取表详情失败: {str(e)}",
            code=500,
            error_type="GET_TABLE_ERROR"
        ), 500


@database_bp.route('/tables/<table_name>/count', methods=['GET'])
@jwt_required()
def get_table_count(table_name: str):
    """获取表记录数
    
    获取指定表的记录总数。
    需要JWT认证。
    
    路径参数:
        table_name (str): 表名
            - 必填
    
    查询参数:
        condition (str): 过滤条件SQL片段 (可选)
            - 示例: "status = 'active'"
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "获取记录数成功",
                "data": {
                    "table_name": "users",
                    "count": 1000,
                    "condition": null,
                    "counted_at": "2024-01-01T00:00:00"
                }
            }
    
    示例:
        curl -X GET "http://localhost:5000/api/v1/database/tables/users/count" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        from backend.services.database_management_service import DatabaseManagementService
        
        condition = request.args.get('condition')
        
        service = DatabaseManagementService()
        count_info = service.get_table_count(table_name, condition)
        
        return success_response(
            data=count_info,
            message="获取记录数成功"
        )
        
    except Exception as e:
        logger.error(f"Get table count failed: {e}")
        return error_response(
            message=f"获取记录数失败: {str(e)}",
            code=500,
            error_type="COUNT_ERROR"
        ), 500


@database_bp.route('/tables/sync', methods=['POST'])
@jwt_required()
def sync_tables():
    """同步表结构
    
    将ORM模型同步到数据库，创建缺失的表和列。
    需要JWT认证。此操作可能需要较长时间。
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
        Content-Type (str): application/json
            - 可选
    
    请求体 (JSON):
        {
            "force": false,           // 是否强制同步 (可选)
            "tables": ["users"]       // 指定要同步的表 (可选，默认全部)
        }
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "表结构同步成功",
                "data": {
                    "created_tables": ["new_table1", "new_table2"],
                    "updated_tables": ["users"],
                    "skipped_tables": [],
                    "errors": [],
                    "sync_duration_ms": 1234,
                    "synced_at": "2024-01-01T00:00:00"
                }
            }
    
    示例:
        curl -X POST "http://localhost:5000/api/v1/database/tables/sync" \\
             -H "Authorization: Bearer <token>" \\
             -H "Content-Type: application/json" \\
             -d '{"force": false}'
    """
    try:
        from backend.services.database_management_service import DatabaseManagementService
        
        data = request.get_json() or {}
        force = data.get('force', False)
        tables = data.get('tables')
        
        service = DatabaseManagementService()
        sync_result = service.sync_tables(force=force, tables=tables)
        
        return success_response(
            data=sync_result,
            message="表结构同步成功"
        )
        
    except Exception as e:
        logger.error(f"Sync tables failed: {e}")
        return error_response(
            message=f"表结构同步失败: {str(e)}",
            code=500,
            error_type="SYNC_ERROR"
        ), 500


@database_bp.route('/tables/<table_name>/truncate', methods=['POST'])
@jwt_required()
def truncate_table(table_name: str):
    """清空表数据
    
    删除指定表的所有数据。此操作不可逆，请谨慎使用。
    需要JWT认证。
    
    路径参数:
        table_name (str): 表名
            - 必填
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
        Content-Type (str): application/json
            - 必填
    
    请求体 (JSON):
        {
            "confirm": true,          // 确认操作 (必填，必须为true)
            "cascade": false          // 是否级联删除 (可选)
        }
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "表数据清空成功",
                "data": {
                    "table_name": "users",
                    "deleted_count": 1000,
                    "truncated_at": "2024-01-01T00:00:00"
                }
            }
        失败:
            - 400: 未确认操作
            - 403: 无权限
            - 404: 表不存在
    
    示例:
        curl -X POST "http://localhost:5000/api/v1/database/tables/users/truncate" \\
             -H "Authorization: Bearer <token>" \\
             -H "Content-Type: application/json" \\
             -d '{"confirm": true}'
    """
    try:
        from backend.services.database_management_service import DatabaseManagementService
        
        data = request.get_json()
        if not data or not data.get('confirm'):
            return error_response(
                message="请确认操作：设置 confirm=true",
                code=400,
                error_type="CONFIRMATION_REQUIRED"
            ), 400
        
        cascade = data.get('cascade', False)
        
        # 检查保护表
        protected_tables = {'users', 'roles', 'permissions', 'alembic_version'}
        if table_name.lower() in protected_tables:
            return error_response(
                message=f"表 {table_name} 是受保护的系统表，不允许清空",
                code=403,
                error_type="PROTECTED_TABLE"
            ), 403
        
        service = DatabaseManagementService()
        result = service.truncate_table(table_name, cascade=cascade)
        
        current_user = get_jwt_identity()
        logger.warning(f"User {current_user} truncated table {table_name}")
        
        return success_response(
            data=result,
            message="表数据清空成功"
        )
        
    except Exception as e:
        logger.error(f"Truncate table failed: {e}")
        return error_response(
            message=f"清空表数据失败: {str(e)}",
            code=500,
            error_type="TRUNCATE_ERROR"
        ), 500


# ============================================================================
# 连接池管理API
# ============================================================================

@database_bp.route('/pool/stats', methods=['GET'])
@jwt_required()
def get_pool_stats():
    """获取连接池统计信息
    
    获取数据库连接池的当前状态和统计信息。
    需要JWT认证。
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "获取连接池统计成功",
                "data": {
                    "pool_size": 10,
                    "max_overflow": 20,
                    "checked_in": 8,
                    "checked_out": 2,
                    "overflow": 0,
                    "invalid": 0,
                    "idle_connections": 6,
                    "utilization_percent": 20.0,
                    "config": {
                        "pool_timeout": 30,
                        "pool_recycle": 3600
                    }
                }
            }
    
    示例:
        curl -X GET "http://localhost:5000/api/v1/database/pool/stats" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        from backend.services.database_management_service import DatabaseManagementService
        
        service = DatabaseManagementService()
        pool_stats = service.get_pool_stats()
        
        return success_response(
            data=pool_stats,
            message="获取连接池统计成功"
        )
        
    except Exception as e:
        logger.error(f"Get pool stats failed: {e}")
        return error_response(
            message=f"获取连接池统计失败: {str(e)}",
            code=500,
            error_type="POOL_STATS_ERROR"
        ), 500


@database_bp.route('/pool/reset', methods=['POST'])
@jwt_required()
def reset_pool():
    """重置连接池
    
    关闭所有连接并重新初始化连接池。用于处理连接泄漏或僵死连接。
    需要JWT认证。此操作会短暂影响数据库访问。
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
        Content-Type (str): application/json
            - 可选
    
    请求体 (JSON):
        {
            "force": false,           // 是否强制重置 (可选)
            "timeout": 30             // 等待超时秒数 (可选)
        }
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "连接池重置成功",
                "data": {
                    "closed_connections": 10,
                    "new_pool_size": 10,
                    "reset_duration_ms": 500,
                    "reset_at": "2024-01-01T00:00:00"
                }
            }
    
    示例:
        curl -X POST "http://localhost:5000/api/v1/database/pool/reset" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        from backend.services.database_management_service import DatabaseManagementService
        
        data = request.get_json() or {}
        force = data.get('force', False)
        timeout = data.get('timeout', 30)
        
        service = DatabaseManagementService()
        result = service.reset_pool(force=force, timeout=timeout)
        
        current_user = get_jwt_identity()
        logger.info(f"User {current_user} reset database connection pool")
        
        return success_response(
            data=result,
            message="连接池重置成功"
        )
        
    except Exception as e:
        logger.error(f"Reset pool failed: {e}")
        return error_response(
            message=f"连接池重置失败: {str(e)}",
            code=500,
            error_type="RESET_POOL_ERROR"
        ), 500


# ============================================================================
# 数据统计API
# ============================================================================

@database_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_database_stats():
    """获取数据库统计信息
    
    获取数据库的整体统计信息概览。
    需要JWT认证。
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "获取数据库统计信息成功",
                "data": {
                    "database_info": {
                        "name": "vectorsphere",
                        "type": "postgresql",
                        "version": "14.0",
                        "encoding": "UTF8"
                    },
                    "tables": {
                        "total_count": 25,
                        "total_rows": 100000,
                        "largest_table": "training_sessions"
                    },
                    "storage": {
                        "total_size_bytes": 1073741824,
                        "total_size_human": "1.00 GB",
                        "data_size_bytes": 858993459,
                        "index_size_bytes": 214748364
                    },
                    "connection_pool": {
                        "pool_size": 10,
                        "active_connections": 2,
                        "utilization_percent": 20.0
                    }
                }
            }
    
    示例:
        curl -X GET "http://localhost:5000/api/v1/database/stats" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        from backend.services.database_management_service import DatabaseManagementService
        
        service = DatabaseManagementService()
        stats = service.get_database_stats()
        
        return success_response(
            data=stats,
            message="获取数据库统计信息成功"
        )
        
    except Exception as e:
        logger.error(f"Get database stats failed: {e}")
        return error_response(
            message=f"获取数据库统计信息失败: {str(e)}",
            code=500,
            error_type="STATS_ERROR"
        ), 500


@database_bp.route('/stats/tables', methods=['GET'])
@jwt_required()
def get_tables_stats():
    """获取各表统计信息
    
    获取所有表的详细统计信息。
    需要JWT认证。
    
    查询参数:
        order_by (str): 排序字段 (可选)
            - 可选值: name, row_count, size
            - 默认: row_count
        order_desc (bool): 是否降序 (可选，默认true)
        limit (int): 返回数量 (可选，默认全部)
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "获取表统计成功",
                "data": {
                    "tables": [
                        {
                            "name": "training_sessions",
                            "row_count": 50000,
                            "size_bytes": 52428800,
                            "size_human": "50.00 MB",
                            "index_size_bytes": 10485760,
                            "last_vacuum": "2024-01-01T00:00:00",
                            "last_analyze": "2024-01-01T00:00:00"
                        }
                    ],
                    "total_tables": 25
                }
            }
    
    示例:
        curl -X GET "http://localhost:5000/api/v1/database/stats/tables?order_by=size&limit=10" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        from backend.services.database_management_service import DatabaseManagementService
        
        order_by = request.args.get('order_by', 'row_count')
        order_desc = request.args.get('order_desc', 'true').lower() == 'true'
        limit = request.args.get('limit', type=int)
        
        service = DatabaseManagementService()
        tables_stats = service.get_tables_stats(
            order_by=order_by,
            order_desc=order_desc,
            limit=limit
        )
        
        return success_response(
            data=tables_stats,
            message="获取表统计成功"
        )
        
    except Exception as e:
        logger.error(f"Get tables stats failed: {e}")
        return error_response(
            message=f"获取表统计失败: {str(e)}",
            code=500,
            error_type="TABLES_STATS_ERROR"
        ), 500


@database_bp.route('/stats/size', methods=['GET'])
@jwt_required()
def get_database_size():
    """获取数据库大小信息
    
    获取数据库的存储空间使用详情。
    需要JWT认证。
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "获取数据库大小成功",
                "data": {
                    "total_size_bytes": 1073741824,
                    "total_size_human": "1.00 GB",
                    "data_size_bytes": 858993459,
                    "data_size_human": "819.20 MB",
                    "index_size_bytes": 214748364,
                    "index_size_human": "204.80 MB",
                    "tables_breakdown": [
                        {
                            "name": "training_sessions",
                            "size_bytes": 52428800,
                            "percentage": 4.88
                        }
                    ]
                }
            }
    
    示例:
        curl -X GET "http://localhost:5000/api/v1/database/stats/size" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        from backend.services.database_management_service import DatabaseManagementService
        
        service = DatabaseManagementService()
        size_info = service.get_database_size()
        
        return success_response(
            data=size_info,
            message="获取数据库大小成功"
        )
        
    except Exception as e:
        logger.error(f"Get database size failed: {e}")
        return error_response(
            message=f"获取数据库大小失败: {str(e)}",
            code=500,
            error_type="SIZE_ERROR"
        ), 500


# ============================================================================
# 查询执行API
# ============================================================================

@database_bp.route('/query', methods=['POST'])
@jwt_required()
def execute_query():
    """执行只读查询
    
    执行只读SQL查询并返回结果。仅支持SELECT语句。
    需要JWT认证。
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
        Content-Type (str): application/json
            - 必填
    
    请求体 (JSON):
        {
            "sql": "SELECT * FROM users LIMIT 10",   // SQL查询 (必填)
            "params": {},                             // 参数 (可选)
            "timeout": 30,                            // 超时秒数 (可选)
            "max_rows": 1000                          // 最大返回行数 (可选)
        }
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "查询执行成功",
                "data": {
                    "columns": ["id", "name", "email"],
                    "rows": [
                        {"id": "1", "name": "Admin", "email": "admin@example.com"}
                    ],
                    "row_count": 1,
                    "execution_time_ms": 5.2,
                    "truncated": false
                }
            }
        失败:
            - 400: SQL语法错误或非SELECT语句
            - 408: 查询超时
    
    示例:
        curl -X POST "http://localhost:5000/api/v1/database/query" \\
             -H "Authorization: Bearer <token>" \\
             -H "Content-Type: application/json" \\
             -d '{"sql": "SELECT id, name FROM users LIMIT 10"}'
    """
    try:
        from backend.services.database_management_service import DatabaseManagementService
        
        data = request.get_json()
        if not data or 'sql' not in data:
            return error_response(
                message="缺少sql参数",
                code=400,
                error_type="MISSING_SQL"
            ), 400
        
        sql = data['sql'].strip()
        
        # 安全检查：只允许SELECT语句
        if not sql.upper().startswith('SELECT'):
            return error_response(
                message="仅允许执行SELECT查询",
                code=400,
                error_type="INVALID_QUERY"
            ), 400
        
        params = data.get('params', {})
        timeout = data.get('timeout', 30)
        max_rows = min(data.get('max_rows', 1000), 10000)  # 最多10000行
        
        service = DatabaseManagementService()
        result = service.execute_query(
            sql=sql,
            params=params,
            timeout=timeout,
            max_rows=max_rows
        )
        
        current_user = get_jwt_identity()
        logger.info(f"User {current_user} executed query: {sql[:100]}...")
        
        return success_response(
            data=result,
            message="查询执行成功"
        )
        
    except Exception as e:
        logger.error(f"Execute query failed: {e}")
        return error_response(
            message=f"查询执行失败: {str(e)}",
            code=500,
            error_type="QUERY_ERROR"
        ), 500


@database_bp.route('/query/explain', methods=['POST'])
@jwt_required()
def explain_query():
    """分析查询计划
    
    获取SQL查询的执行计划，用于性能分析。
    需要JWT认证。
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
        Content-Type (str): application/json
            - 必填
    
    请求体 (JSON):
        {
            "sql": "SELECT * FROM users WHERE status = 'active'",   // SQL查询 (必填)
            "analyze": false,                                         // 是否实际执行 (可选)
            "format": "text"                                          // 输出格式 (可选: text/json)
        }
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "查询计划分析成功",
                "data": {
                    "plan": "Seq Scan on users...",
                    "estimated_cost": 100.0,
                    "estimated_rows": 1000,
                    "suggestions": [
                        "Consider adding an index on 'status' column"
                    ]
                }
            }
    
    示例:
        curl -X POST "http://localhost:5000/api/v1/database/query/explain" \\
             -H "Authorization: Bearer <token>" \\
             -H "Content-Type: application/json" \\
             -d '{"sql": "SELECT * FROM users WHERE email = '\''test@example.com'\''"}'
    """
    try:
        from backend.services.database_management_service import DatabaseManagementService
        
        data = request.get_json()
        if not data or 'sql' not in data:
            return error_response(
                message="缺少sql参数",
                code=400,
                error_type="MISSING_SQL"
            ), 400
        
        sql = data['sql']
        analyze = data.get('analyze', False)
        format_type = data.get('format', 'text')
        
        service = DatabaseManagementService()
        result = service.explain_query(
            sql=sql,
            analyze=analyze,
            format_type=format_type
        )
        
        return success_response(
            data=result,
            message="查询计划分析成功"
        )
        
    except Exception as e:
        logger.error(f"Explain query failed: {e}")
        return error_response(
            message=f"查询计划分析失败: {str(e)}",
            code=500,
            error_type="EXPLAIN_ERROR"
        ), 500


# ============================================================================
# 维护操作API
# ============================================================================

@database_bp.route('/maintenance/vacuum', methods=['POST'])
@jwt_required()
def vacuum_database():
    """执行数据库清理操作
    
    执行VACUUM操作，回收删除行占用的空间。
    需要JWT认证。此操作可能需要较长时间。
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
        Content-Type (str): application/json
            - 可选
    
    请求体 (JSON):
        {
            "table_name": null,       // 指定表名 (可选，默认全部)
            "full": false,            // 是否执行FULL VACUUM (可选)
            "analyze": true           // 是否同时执行ANALYZE (可选)
        }
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "数据库清理完成",
                "data": {
                    "tables_vacuumed": ["users", "training_sessions"],
                    "space_reclaimed_bytes": 10485760,
                    "space_reclaimed_human": "10.00 MB",
                    "duration_ms": 5000,
                    "vacuumed_at": "2024-01-01T00:00:00"
                }
            }
    
    示例:
        curl -X POST "http://localhost:5000/api/v1/database/maintenance/vacuum" \\
             -H "Authorization: Bearer <token>" \\
             -H "Content-Type: application/json" \\
             -d '{"analyze": true}'
    """
    try:
        from backend.services.database_management_service import DatabaseManagementService
        
        data = request.get_json() or {}
        table_name = data.get('table_name')
        full = data.get('full', False)
        analyze = data.get('analyze', True)
        
        service = DatabaseManagementService()
        result = service.vacuum(
            table_name=table_name,
            full=full,
            analyze=analyze
        )
        
        current_user = get_jwt_identity()
        logger.info(f"User {current_user} executed vacuum operation")
        
        return success_response(
            data=result,
            message="数据库清理完成"
        )
        
    except Exception as e:
        logger.error(f"Vacuum failed: {e}")
        return error_response(
            message=f"数据库清理失败: {str(e)}",
            code=500,
            error_type="VACUUM_ERROR"
        ), 500


@database_bp.route('/maintenance/analyze', methods=['POST'])
@jwt_required()
def analyze_database():
    """更新数据库统计信息
    
    执行ANALYZE操作，更新查询优化器使用的统计信息。
    需要JWT认证。
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
        Content-Type (str): application/json
            - 可选
    
    请求体 (JSON):
        {
            "table_name": null        // 指定表名 (可选，默认全部)
        }
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "统计信息更新完成",
                "data": {
                    "tables_analyzed": ["users", "training_sessions"],
                    "duration_ms": 2000,
                    "analyzed_at": "2024-01-01T00:00:00"
                }
            }
    
    示例:
        curl -X POST "http://localhost:5000/api/v1/database/maintenance/analyze" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        from backend.services.database_management_service import DatabaseManagementService
        
        data = request.get_json() or {}
        table_name = data.get('table_name')
        
        service = DatabaseManagementService()
        result = service.analyze(table_name=table_name)
        
        return success_response(
            data=result,
            message="统计信息更新完成"
        )
        
    except Exception as e:
        logger.error(f"Analyze failed: {e}")
        return error_response(
            message=f"统计信息更新失败: {str(e)}",
            code=500,
            error_type="ANALYZE_ERROR"
        ), 500


@database_bp.route('/maintenance/locks', methods=['GET'])
@jwt_required()
def get_locks():
    """获取当前锁信息
    
    获取数据库当前的锁状态和等待情况。
    需要JWT认证。
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "获取锁信息成功",
                "data": {
                    "locks": [
                        {
                            "pid": 1234,
                            "database": "vectorsphere",
                            "relation": "users",
                            "lock_type": "RowExclusiveLock",
                            "granted": true,
                            "wait_start": null,
                            "query": "UPDATE users..."
                        }
                    ],
                    "waiting_queries": [],
                    "total_locks": 5,
                    "total_waiting": 0
                }
            }
    
    示例:
        curl -X GET "http://localhost:5000/api/v1/database/maintenance/locks" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        from backend.services.database_management_service import DatabaseManagementService
        
        service = DatabaseManagementService()
        locks_info = service.get_locks()
        
        return success_response(
            data=locks_info,
            message="获取锁信息成功"
        )
        
    except Exception as e:
        logger.error(f"Get locks failed: {e}")
        return error_response(
            message=f"获取锁信息失败: {str(e)}",
            code=500,
            error_type="LOCKS_ERROR"
        ), 500


# ============================================================================
# 备份恢复API
# ============================================================================

@database_bp.route('/backup', methods=['POST'])
@jwt_required()
def create_backup():
    """创建数据库备份
    
    创建数据库的逻辑备份。
    需要JWT认证。此操作可能需要较长时间。
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
        Content-Type (str): application/json
            - 可选
    
    请求体 (JSON):
        {
            "backup_name": "manual_backup",   // 备份名称 (可选)
            "tables": null,                    // 指定表 (可选，默认全部)
            "compression": true                // 是否压缩 (可选)
        }
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "备份创建成功",
                "data": {
                    "backup_id": "backup_20240101_000000",
                    "backup_name": "manual_backup",
                    "backup_path": "/backups/backup_20240101_000000.sql.gz",
                    "size_bytes": 10485760,
                    "size_human": "10.00 MB",
                    "tables_count": 25,
                    "duration_ms": 30000,
                    "created_at": "2024-01-01T00:00:00"
                }
            }
    
    示例:
        curl -X POST "http://localhost:5000/api/v1/database/backup" \\
             -H "Authorization: Bearer <token>" \\
             -H "Content-Type: application/json" \\
             -d '{"backup_name": "before_migration"}'
    """
    try:
        from backend.services.database_management_service import DatabaseManagementService
        
        data = request.get_json() or {}
        backup_name = data.get('backup_name')
        tables = data.get('tables')
        compression = data.get('compression', True)
        
        service = DatabaseManagementService()
        result = service.create_backup(
            backup_name=backup_name,
            tables=tables,
            compression=compression
        )
        
        current_user = get_jwt_identity()
        logger.info(f"User {current_user} created database backup: {result.get('backup_id')}")
        
        return success_response(
            data=result,
            message="备份创建成功"
        )
        
    except Exception as e:
        logger.error(f"Create backup failed: {e}")
        return error_response(
            message=f"备份创建失败: {str(e)}",
            code=500,
            error_type="BACKUP_ERROR"
        ), 500


@database_bp.route('/backup/list', methods=['GET'])
@jwt_required()
def list_backups():
    """获取备份列表
    
    获取可用的数据库备份列表。
    需要JWT认证。
    
    查询参数:
        limit (int): 返回数量 (可选，默认20)
        offset (int): 偏移量 (可选，默认0)
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "获取备份列表成功",
                "data": {
                    "backups": [
                        {
                            "backup_id": "backup_20240101_000000",
                            "backup_name": "manual_backup",
                            "size_bytes": 10485760,
                            "size_human": "10.00 MB",
                            "created_at": "2024-01-01T00:00:00",
                            "status": "completed"
                        }
                    ],
                    "total_count": 5
                }
            }
    
    示例:
        curl -X GET "http://localhost:5000/api/v1/database/backup/list" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        from backend.services.database_management_service import DatabaseManagementService
        
        limit = request.args.get('limit', 20, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        service = DatabaseManagementService()
        backups = service.list_backups(limit=limit, offset=offset)
        
        return success_response(
            data=backups,
            message="获取备份列表成功"
        )
        
    except Exception as e:
        logger.error(f"List backups failed: {e}")
        return error_response(
            message=f"获取备份列表失败: {str(e)}",
            code=500,
            error_type="LIST_BACKUPS_ERROR"
        ), 500


@database_bp.route('/restore', methods=['POST'])
@jwt_required()
def restore_backup():
    """恢复数据库备份
    
    从备份恢复数据库。此操作会覆盖现有数据。
    需要JWT认证。此操作可能需要较长时间。
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
        Content-Type (str): application/json
            - 必填
    
    请求体 (JSON):
        {
            "backup_id": "backup_20240101_000000",   // 备份ID (必填)
            "confirm": true,                          // 确认操作 (必填)
            "tables": null                            // 指定恢复的表 (可选)
        }
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "数据库恢复成功",
                "data": {
                    "backup_id": "backup_20240101_000000",
                    "tables_restored": 25,
                    "rows_restored": 100000,
                    "duration_ms": 60000,
                    "restored_at": "2024-01-01T00:00:00"
                }
            }
        失败:
            - 400: 未确认或备份ID缺失
            - 404: 备份不存在
    
    示例:
        curl -X POST "http://localhost:5000/api/v1/database/restore" \\
             -H "Authorization: Bearer <token>" \\
             -H "Content-Type: application/json" \\
             -d '{"backup_id": "backup_20240101_000000", "confirm": true}'
    """
    try:
        from backend.services.database_management_service import DatabaseManagementService
        
        data = request.get_json()
        if not data:
            return error_response(
                message="请求数据不能为空",
                code=400,
                error_type="EMPTY_REQUEST"
            ), 400
        
        backup_id = data.get('backup_id')
        if not backup_id:
            return error_response(
                message="缺少backup_id参数",
                code=400,
                error_type="MISSING_BACKUP_ID"
            ), 400
        
        if not data.get('confirm'):
            return error_response(
                message="请确认恢复操作：设置 confirm=true",
                code=400,
                error_type="CONFIRMATION_REQUIRED"
            ), 400
        
        tables = data.get('tables')
        
        service = DatabaseManagementService()
        result = service.restore_backup(
            backup_id=backup_id,
            tables=tables
        )
        
        current_user = get_jwt_identity()
        logger.warning(f"User {current_user} restored database from backup: {backup_id}")
        
        return success_response(
            data=result,
            message="数据库恢复成功"
        )
        
    except Exception as e:
        logger.error(f"Restore backup failed: {e}")
        return error_response(
            message=f"数据库恢复失败: {str(e)}",
            code=500,
            error_type="RESTORE_ERROR"
        ), 500
