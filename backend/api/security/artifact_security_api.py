#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""工件安全策略API

提供工件安全策略的配置和管理接口。
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file, g
from werkzeug.utils import secure_filename
from functools import wraps

logger = logging.getLogger(__name__)

# 创建蓝图
artifact_security_bp = Blueprint('artifact_security', __name__, url_prefix='/api/security/artifacts')


# ==================== 全局服务实例 ====================

_security_service = None
_management_service = None


def get_security_service():
    """获取安全服务实例"""
    global _security_service
    if _security_service is None:
        try:
            from backend.services.artifact_security import get_artifact_security_service
            _security_service = get_artifact_security_service()
        except ImportError:
            from backend.services.artifact_security import ArtifactSecurityService
            _security_service = ArtifactSecurityService()
    return _security_service


def get_management_service():
    """获取管理服务实例"""
    global _management_service
    if _management_service is None:
        try:
            from backend.services.artifact_management import get_artifact_management_service
            _management_service = get_artifact_management_service()
        except ImportError:
            logger.warning("ArtifactManagementService not available, using security service")
            _management_service = None
    return _management_service


# ==================== 辅助函数 ====================

def _get_current_user() -> Optional[Dict]:
    """获取当前用户"""
    return getattr(g, 'current_user', None)


def _get_tenant_id() -> Optional[str]:
    """获取当前租户ID"""
    if hasattr(g, 'tenant_id'):
        return g.tenant_id
    return request.headers.get('X-Tenant-ID')


def _get_client_info() -> Dict[str, str]:
    """获取客户端信息"""
    return {
        'ip_address': request.remote_addr or '',
        'user_agent': request.headers.get('User-Agent', '')
    }


def token_required(f):
    """认证装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        # 尝试从请求头获取令牌
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({
                'success': False,
                'error': 'Missing or invalid authorization header'
            }), 401
        
        token = auth_header[7:]
        
        # 验证令牌
        try:
            from backend.services.security_service import get_security_service
            security_svc = get_security_service()
            result = security_svc.auth.validate_token(token)
            
            if not result.get('valid'):
                return jsonify({
                    'success': False,
                    'error': result.get('error', 'Invalid token')
                }), 401
            
            current_user = {
                'id': result.get('user_id'),
                'user_id': result.get('user_id'),
                'tenant_id': result.get('tenant_id')
            }
            g.current_user = current_user
            g.tenant_id = result.get('tenant_id')
            
        except ImportError:
            # 开发模式：使用模拟用户
            current_user = {
                'id': request.headers.get('X-User-ID', 'dev_user'),
                'user_id': request.headers.get('X-User-ID', 'dev_user'),
                'tenant_id': request.headers.get('X-Tenant-ID')
            }
            g.current_user = current_user
            g.tenant_id = current_user.get('tenant_id')
        
        return f(current_user, *args, **kwargs)
    
    return decorated


# ==================== 安全策略API ====================

@artifact_security_bp.route('/policies', methods=['GET'])
@token_required
def get_security_policies(current_user):
    """获取安全策略列表"""
    try:
        tenant_id = _get_tenant_id()
        security_level = request.args.get('security_level')
        
        service = get_security_service()
        policies = service.get_security_policies(tenant_id, security_level)
        
        return jsonify({
            'success': True,
            'data': [
                {
                    'id': policy.id,
                    'name': policy.name,
                    'description': policy.description,
                    'security_level': policy.security_level.value if hasattr(policy.security_level, 'value') else policy.security_level,
                    'allowed_file_types': policy.allowed_file_types,
                    'max_file_size': policy.max_file_size,
                    'encryption_required': policy.encryption_required,
                    'virus_scan_required': policy.virus_scan_required,
                    'access_control_enabled': policy.access_control_enabled,
                    'audit_enabled': policy.audit_enabled,
                    'retention_days': policy.retention_days,
                    'is_default': policy.is_default,
                    'is_active': policy.is_active,
                    'created_at': policy.created_at.isoformat() if policy.created_at else None,
                    'updated_at': policy.updated_at.isoformat() if policy.updated_at else None
                }
                for policy in policies
            ]
        }), 200
        
    except Exception as e:
        logger.error(f"获取安全策略失败: {e}")
        return jsonify({
            'success': False,
            'error': f'获取安全策略失败: {str(e)}'
        }), 500


@artifact_security_bp.route('/policies/<policy_id>', methods=['GET'])
@token_required
def get_security_policy(current_user, policy_id):
    """获取单个安全策略"""
    try:
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        policy = service.get_security_policy(policy_id, tenant_id)
        
        if not policy:
            return jsonify({
                'success': False,
                'error': '安全策略不存在'
            }), 404
        
        return jsonify({
            'success': True,
            'data': {
                'id': policy.id,
                'name': policy.name,
                'description': policy.description,
                'security_level': policy.security_level.value if hasattr(policy.security_level, 'value') else policy.security_level,
                'allowed_file_types': policy.allowed_file_types,
                'max_file_size': policy.max_file_size,
                'encryption_required': policy.encryption_required,
                'virus_scan_required': policy.virus_scan_required,
                'access_control_enabled': policy.access_control_enabled,
                'audit_enabled': policy.audit_enabled,
                'retention_days': policy.retention_days,
                'is_default': policy.is_default,
                'is_active': policy.is_active,
                'created_at': policy.created_at.isoformat() if policy.created_at else None,
                'updated_at': policy.updated_at.isoformat() if policy.updated_at else None
            }
        }), 200
        
    except Exception as e:
        logger.error(f"获取安全策略失败: {e}")
        return jsonify({
            'success': False,
            'error': f'获取安全策略失败: {str(e)}'
        }), 500


@artifact_security_bp.route('/policies', methods=['POST'])
@token_required
def create_security_policy(current_user):
    """创建安全策略"""
    try:
        data = request.get_json() or {}
        tenant_id = _get_tenant_id()
        
        # 验证必填字段
        required_fields = ['name', 'security_level', 'allowed_file_types', 'max_file_size']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'缺少必填字段: {field}'
                }), 400
        
        # 导入枚举类型
        from backend.services.artifact_security import SecurityPolicy, SecurityLevel
        
        # 创建策略对象
        policy = SecurityPolicy(
            id=f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(data['name']) % 10000}",
            name=data['name'],
            description=data.get('description', ''),
            security_level=SecurityLevel(data['security_level']),
            allowed_file_types=data['allowed_file_types'],
            max_file_size=data['max_file_size'],
            encryption_required=data.get('encryption_required', False),
            virus_scan_required=data.get('virus_scan_required', True),
            access_control_enabled=data.get('access_control_enabled', True),
            audit_enabled=data.get('audit_enabled', True),
            retention_days=data.get('retention_days', 365),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            is_default=False,
            is_active=True,
            tenant_id=tenant_id,
            created_by=current_user.get('id')
        )
        
        service = get_security_service()
        success = service.create_security_policy(policy)
        
        if success:
            return jsonify({
                'success': True,
                'data': {
                    'id': policy.id,
                    'name': policy.name,
                    'security_level': policy.security_level.value
                },
                'message': '安全策略创建成功'
            }), 201
        else:
            return jsonify({
                'success': False,
                'error': '安全策略创建失败'
            }), 400
            
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'无效的参数值: {str(e)}'
        }), 400
    except Exception as e:
        logger.error(f"创建安全策略失败: {e}")
        return jsonify({
            'success': False,
            'error': f'创建安全策略失败: {str(e)}'
        }), 500


@artifact_security_bp.route('/policies/<policy_id>', methods=['PUT'])
@token_required
def update_security_policy(current_user, policy_id):
    """更新安全策略"""
    try:
        data = request.get_json() or {}
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        success = service.update_security_policy(policy_id, data, tenant_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': '安全策略更新成功'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': '安全策略不存在或更新失败'
            }), 404
            
    except Exception as e:
        logger.error(f"更新安全策略失败: {e}")
        return jsonify({
            'success': False,
            'error': f'更新安全策略失败: {str(e)}'
        }), 500


@artifact_security_bp.route('/policies/<policy_id>', methods=['DELETE'])
@token_required
def delete_security_policy(current_user, policy_id):
    """删除安全策略"""
    try:
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        success = service.delete_security_policy(policy_id, tenant_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': '安全策略删除成功'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': '安全策略不存在或删除失败'
            }), 404
            
    except Exception as e:
        logger.error(f"删除安全策略失败: {e}")
        return jsonify({
            'success': False,
            'error': f'删除安全策略失败: {str(e)}'
        }), 500


# ==================== 工件API ====================

@artifact_security_bp.route('/artifacts', methods=['GET'])
@token_required
def list_artifacts(current_user):
    """列出工件"""
    try:
        tenant_id = _get_tenant_id()
        
        # 获取查询参数
        artifact_type = request.args.get('artifact_type')
        security_level = request.args.get('security_level')
        status = request.args.get('status')
        owner_id = request.args.get('owner_id')
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        
        # 构建过滤条件
        filters = {}
        if artifact_type:
            filters['artifact_type'] = artifact_type
        if security_level:
            filters['security_level'] = security_level
        if status:
            filters['status'] = status
        if owner_id:
            filters['owner_id'] = owner_id
        
        service = get_security_service()
        artifacts = service.list_artifacts(
            user_id=current_user.get('id'),
            filters=filters,
            tenant_id=tenant_id,
            limit=limit,
            offset=offset
        )
        
        return jsonify({
            'success': True,
            'data': [
                {
                    'id': artifact.id,
                    'name': artifact.name,
                    'description': artifact.description,
                    'artifact_type': artifact.artifact_type.value if hasattr(artifact.artifact_type, 'value') else artifact.artifact_type,
                    'security_level': artifact.security_level.value if hasattr(artifact.security_level, 'value') else artifact.security_level,
                    'status': artifact.status.value if hasattr(artifact.status, 'value') else artifact.status,
                    'owner_id': artifact.owner_id,
                    'current_version': artifact.current_version,
                    'version_count': artifact.version_count,
                    'total_size': artifact.total_size,
                    'tags': artifact.tags,
                    'created_at': artifact.created_at.isoformat() if artifact.created_at else None,
                    'updated_at': artifact.updated_at.isoformat() if artifact.updated_at else None
                }
                for artifact in artifacts
            ],
            'count': len(artifacts)
        }), 200
        
    except Exception as e:
        logger.error(f"列出工件失败: {e}")
        return jsonify({
            'success': False,
            'error': f'列出工件失败: {str(e)}'
        }), 500


@artifact_security_bp.route('/artifacts', methods=['POST'])
@token_required
def create_artifact(current_user):
    """创建工件"""
    try:
        data = request.get_json() or {}
        tenant_id = _get_tenant_id()
        
        # 验证必填字段
        required_fields = ['name', 'artifact_type', 'security_level']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'缺少必填字段: {field}'
                }), 400
        
        from backend.services.artifact_security import ArtifactType, SecurityLevel
        
        service = get_security_service()
        success, error_msg, artifact = service.create_artifact(
            name=data['name'],
            description=data.get('description', ''),
            artifact_type=ArtifactType(data['artifact_type']),
            security_level=SecurityLevel(data['security_level']),
            owner_id=current_user.get('id'),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {}),
            tenant_id=tenant_id,
            policy_id=data.get('policy_id')
        )
        
        if success and artifact:
            return jsonify({
                'success': True,
                'data': {
                    'id': artifact.id,
                    'name': artifact.name,
                    'artifact_type': artifact.artifact_type.value if hasattr(artifact.artifact_type, 'value') else artifact.artifact_type,
                    'security_level': artifact.security_level.value if hasattr(artifact.security_level, 'value') else artifact.security_level
                },
                'message': '工件创建成功'
            }), 201
        else:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
            
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'无效的参数值: {str(e)}'
        }), 400
    except Exception as e:
        logger.error(f"创建工件失败: {e}")
        return jsonify({
            'success': False,
            'error': f'创建工件失败: {str(e)}'
        }), 500


@artifact_security_bp.route('/artifacts/<artifact_id>', methods=['GET'])
@token_required
def get_artifact(current_user, artifact_id):
    """获取工件详情"""
    try:
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        artifact = service.get_artifact(artifact_id, current_user.get('id'), tenant_id)
        
        if not artifact:
            return jsonify({
                'success': False,
                'error': '工件不存在或无权限访问'
            }), 404
        
        # 获取版本列表
        versions = service.get_artifact_versions(artifact_id, current_user.get('id'), tenant_id)
        
        # 获取依赖关系
        dependencies = service.get_artifact_dependencies(artifact_id)
        dependents = service.get_artifact_dependents(artifact_id)
        
        return jsonify({
            'success': True,
            'data': {
                'id': artifact.id,
                'name': artifact.name,
                'description': artifact.description,
                'artifact_type': artifact.artifact_type.value if hasattr(artifact.artifact_type, 'value') else artifact.artifact_type,
                'security_level': artifact.security_level.value if hasattr(artifact.security_level, 'value') else artifact.security_level,
                'status': artifact.status.value if hasattr(artifact.status, 'value') else artifact.status,
                'owner_id': artifact.owner_id,
                'current_version': artifact.current_version,
                'version_count': artifact.version_count,
                'total_size': artifact.total_size,
                'tags': artifact.tags,
                'metadata': artifact.metadata,
                'created_at': artifact.created_at.isoformat() if artifact.created_at else None,
                'updated_at': artifact.updated_at.isoformat() if artifact.updated_at else None,
                'versions': [
                    {
                        'id': version.id,
                        'version': version.version,
                        'status': version.status,
                        'file_size': version.file_size,
                        'changelog': version.changelog,
                        'created_by': version.created_by,
                        'created_at': version.created_at.isoformat() if version.created_at else None,
                        'tags': version.tags
                    }
                    for version in versions
                ],
                'dependencies': [
                    {
                        'target_artifact_id': dep.target_artifact_id,
                        'dependency_type': dep.dependency_type,
                        'version_constraint': dep.version_constraint
                    }
                    for dep in dependencies
                ],
                'dependents': [
                    {
                        'source_artifact_id': dep.source_artifact_id,
                        'dependency_type': dep.dependency_type,
                        'version_constraint': dep.version_constraint
                    }
                    for dep in dependents
                ]
            }
        }), 200
        
    except Exception as e:
        logger.error(f"获取工件详情失败: {e}")
        return jsonify({
            'success': False,
            'error': f'获取工件详情失败: {str(e)}'
        }), 500


@artifact_security_bp.route('/artifacts/<artifact_id>/upload', methods=['POST'])
@token_required
def upload_artifact_version(current_user, artifact_id):
    """上传工件版本"""
    try:
        tenant_id = _get_tenant_id()
        
        # 检查文件是否存在
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': '未选择文件'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '未选择文件'
            }), 400
        
        # 获取其他参数
        version = request.form.get('version')
        changelog = request.form.get('changelog', '')
        tags = request.form.getlist('tags')
        
        if not version:
            return jsonify({
                'success': False,
                'error': '版本号不能为空'
            }), 400
        
        # 保存临时文件
        filename = secure_filename(file.filename)
        temp_path = os.path.join('/tmp', f"{current_user.get('id')}_{filename}")
        file.save(temp_path)
        
        try:
            service = get_security_service()
            success, error_msg, artifact_version = service.upload_artifact_version(
                artifact_id=artifact_id,
                file_path=temp_path,
                version=version,
                changelog=changelog,
                user_id=current_user.get('id'),
                tags=tags,
                tenant_id=tenant_id
            )
            
            if success and artifact_version:
                return jsonify({
                    'success': True,
                    'data': {
                        'id': artifact_version.id,
                        'version': artifact_version.version,
                        'artifact_id': artifact_version.artifact_id,
                        'file_size': artifact_version.file_size
                    },
                    'message': '版本上传成功'
                }), 201
            else:
                return jsonify({
                    'success': False,
                    'error': error_msg
                }), 400
                
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
    except Exception as e:
        logger.error(f"上传工件版本失败: {e}")
        return jsonify({
            'success': False,
            'error': f'上传工件版本失败: {str(e)}'
        }), 500


@artifact_security_bp.route('/artifacts/<artifact_id>/versions/<version>/download', methods=['GET'])
@token_required
def download_artifact_version(current_user, artifact_id, version):
    """下载工件版本"""
    try:
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        success, error_msg, file_path = service.download_artifact_version(
            artifact_id, version, current_user.get('id'), tenant_id
        )
        
        if success and file_path:
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
            
    except Exception as e:
        logger.error(f"下载工件版本失败: {e}")
        return jsonify({
            'success': False,
            'error': f'下载工件版本失败: {str(e)}'
        }), 500


@artifact_security_bp.route('/artifacts/<artifact_id>/versions/<version>', methods=['DELETE'])
@token_required
def delete_artifact_version(current_user, artifact_id, version):
    """删除工件版本"""
    try:
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        success, error_msg = service.delete_artifact_version(
            artifact_id, version, current_user.get('id'), tenant_id
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': '版本删除成功'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
            
    except Exception as e:
        logger.error(f"删除工件版本失败: {e}")
        return jsonify({
            'success': False,
            'error': f'删除工件版本失败: {str(e)}'
        }), 500


@artifact_security_bp.route('/artifacts/<artifact_id>/dependencies', methods=['POST'])
@token_required
def add_artifact_dependency(current_user, artifact_id):
    """添加工件依赖"""
    try:
        data = request.get_json() or {}
        tenant_id = _get_tenant_id()
        
        target_artifact_id = data.get('target_artifact_id')
        dependency_type = data.get('dependency_type', 'required')
        version_constraint = data.get('version_constraint', '*')
        
        if not target_artifact_id:
            return jsonify({
                'success': False,
                'error': '目标工件ID不能为空'
            }), 400
        
        service = get_security_service()
        success, error_msg = service.add_dependency(
            artifact_id, target_artifact_id, dependency_type, 
            version_constraint, current_user.get('id'), tenant_id
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': '依赖添加成功'
            }), 201
        else:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
            
    except Exception as e:
        logger.error(f"添加工件依赖失败: {e}")
        return jsonify({
            'success': False,
            'error': f'添加工件依赖失败: {str(e)}'
        }), 500


@artifact_security_bp.route('/artifacts/<artifact_id>/dependencies/<target_artifact_id>', methods=['DELETE'])
@token_required
def remove_artifact_dependency(current_user, artifact_id, target_artifact_id):
    """移除工件依赖"""
    try:
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        success, error_msg = service.remove_dependency(
            artifact_id, target_artifact_id, current_user.get('id'), tenant_id
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': '依赖移除成功'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
            
    except Exception as e:
        logger.error(f"移除工件依赖失败: {e}")
        return jsonify({
            'success': False,
            'error': f'移除工件依赖失败: {str(e)}'
        }), 500


@artifact_security_bp.route('/artifacts/<artifact_id>/status', methods=['PUT'])
@token_required
def update_artifact_status(current_user, artifact_id):
    """更新工件状态"""
    try:
        data = request.get_json() or {}
        tenant_id = _get_tenant_id()
        status = data.get('status')
        
        if not status:
            return jsonify({
                'success': False,
                'error': '状态不能为空'
            }), 400
        
        from backend.services.artifact_security import ArtifactStatus
        
        try:
            artifact_status = ArtifactStatus(status)
        except ValueError:
            return jsonify({
                'success': False,
                'error': '无效的状态值'
            }), 400
        
        service = get_security_service()
        success, error_msg = service.update_artifact_status(
            artifact_id, artifact_status, current_user.get('id'), tenant_id
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': '状态更新成功'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
            
    except Exception as e:
        logger.error(f"更新工件状态失败: {e}")
        return jsonify({
            'success': False,
            'error': f'更新工件状态失败: {str(e)}'
        }), 500


# ==================== 文件API ====================

@artifact_security_bp.route('/files', methods=['GET'])
@token_required
def list_files(current_user):
    """列出文件"""
    try:
        tenant_id = _get_tenant_id()
        artifact_type = request.args.get('artifact_type')
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        
        service = get_security_service()
        files = service.list_files(
            user_id=current_user.get('id'),
            tenant_id=tenant_id,
            artifact_type=artifact_type,
            limit=limit,
            offset=offset
        )
        
        return jsonify({
            'success': True,
            'data': files,
            'count': len(files)
        }), 200
        
    except Exception as e:
        logger.error(f"列出文件失败: {e}")
        return jsonify({
            'success': False,
            'error': f'列出文件失败: {str(e)}'
        }), 500


@artifact_security_bp.route('/files/<file_id>', methods=['GET'])
@token_required
def get_file_metadata(current_user, file_id):
    """获取文件元数据"""
    try:
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        file_data = service.get_file_metadata(file_id, current_user.get('id'), tenant_id)
        
        if file_data:
            return jsonify({
                'success': True,
                'data': file_data
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': '文件不存在或无权限访问'
            }), 404
            
    except Exception as e:
        logger.error(f"获取文件元数据失败: {e}")
        return jsonify({
            'success': False,
            'error': f'获取文件元数据失败: {str(e)}'
        }), 500


@artifact_security_bp.route('/files/<file_id>/download', methods=['GET'])
@token_required
def download_file(current_user, file_id):
    """下载文件"""
    try:
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        success, error_msg, file_path = service.access_file(
            file_id, current_user.get('id'), "download", tenant_id
        )
        
        if success and file_path:
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
            
    except Exception as e:
        logger.error(f"下载文件失败: {e}")
        return jsonify({
            'success': False,
            'error': f'下载文件失败: {str(e)}'
        }), 500


@artifact_security_bp.route('/files/<file_id>', methods=['DELETE'])
@token_required
def delete_file(current_user, file_id):
    """删除文件"""
    try:
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        success, error_msg = service.delete_file(file_id, current_user.get('id'), tenant_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': '文件删除成功'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
            
    except Exception as e:
        logger.error(f"删除文件失败: {e}")
        return jsonify({
            'success': False,
            'error': f'删除文件失败: {str(e)}'
        }), 500


# ==================== 清理API ====================

@artifact_security_bp.route('/cleanup', methods=['POST'])
@token_required
def cleanup_artifacts(current_user):
    """清理工件"""
    try:
        tenant_id = _get_tenant_id()
        data = request.get_json() or {}
        retention_days = data.get('retention_days', 90)
        
        service = get_security_service()
        
        # 清理过期文件
        cleaned_files = service.cleanup_expired_files(tenant_id)
        
        # 清理旧版本
        cleaned_versions = service.cleanup_old_versions(retention_days, tenant_id)
        
        return jsonify({
            'success': True,
            'data': {
                'cleaned_files': cleaned_files,
                'cleaned_versions': cleaned_versions
            },
            'message': '清理完成'
        }), 200
        
    except Exception as e:
        logger.error(f"清理工件失败: {e}")
        return jsonify({
            'success': False,
            'error': f'清理工件失败: {str(e)}'
        }), 500


# ==================== 高级管理API ====================

@artifact_security_bp.route('/artifacts/<artifact_id>/versions/compare', methods=['POST'])
@token_required
def compare_artifact_versions(current_user, artifact_id):
    """比较工件版本"""
    try:
        data = request.get_json() or {}
        tenant_id = _get_tenant_id()
        version1 = data.get('version1')
        version2 = data.get('version2')
        
        if not version1 or not version2:
            return jsonify({
                'success': False,
                'error': '需要提供两个版本号进行比较'
            }), 400
        
        management_service = get_management_service()
        if management_service:
            result = management_service.compare_versions(
                artifact_id, version1, version2, 
                current_user.get('id'), tenant_id
            )
        else:
            return jsonify({
                'success': False,
                'error': '管理服务不可用'
            }), 500
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"比较版本失败: {e}")
        return jsonify({
            'success': False,
            'error': f'比较版本失败: {str(e)}'
        }), 500


@artifact_security_bp.route('/artifacts/<artifact_id>/dependencies/tree', methods=['GET'])
@token_required
def get_dependency_tree(current_user, artifact_id):
    """获取依赖树"""
    try:
        tenant_id = _get_tenant_id()
        
        management_service = get_management_service()
        if management_service:
            result = management_service.resolve_dependencies(artifact_id, tenant_id)
        else:
            return jsonify({
                'success': False,
                'error': '管理服务不可用'
            }), 500
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"获取依赖树失败: {e}")
        return jsonify({
            'success': False,
            'error': f'获取依赖树失败: {str(e)}'
        }), 500


@artifact_security_bp.route('/artifacts/<artifact_id>/dependents', methods=['GET'])
@token_required
def get_artifact_dependents(current_user, artifact_id):
    """获取依赖此工件的列表"""
    try:
        tenant_id = _get_tenant_id()
        
        management_service = get_management_service()
        if management_service:
            dependents = management_service.get_artifact_dependents(
                artifact_id, current_user.get('id'), tenant_id
            )
        else:
            # 回退到安全服务
            service = get_security_service()
            dependents = service.get_artifact_dependents(artifact_id)
        
        return jsonify({
            'success': True,
            'data': [
                {
                    'source_artifact_id': dep.source_artifact_id,
                    'dependency_type': dep.dependency_type,
                    'version_constraint': dep.version_constraint
                }
                for dep in dependents
            ],
            'count': len(dependents)
        }), 200
        
    except Exception as e:
        logger.error(f"获取依赖者列表失败: {e}")
        return jsonify({
            'success': False,
            'error': f'获取依赖者列表失败: {str(e)}'
        }), 500


@artifact_security_bp.route('/artifacts/<artifact_id>', methods=['PUT'])
@token_required
def update_artifact(current_user, artifact_id):
    """更新工件"""
    try:
        data = request.get_json() or {}
        tenant_id = _get_tenant_id()
        
        management_service = get_management_service()
        if management_service:
            success, error_msg = management_service.update_artifact(
                artifact_id, data, current_user.get('id'), tenant_id
            )
        else:
            return jsonify({
                'success': False,
                'error': '管理服务不可用'
            }), 500
        
        if success:
            return jsonify({
                'success': True,
                'message': '工件更新成功'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
            
    except Exception as e:
        logger.error(f"更新工件失败: {e}")
        return jsonify({
            'success': False,
            'error': f'更新工件失败: {str(e)}'
        }), 500


@artifact_security_bp.route('/artifacts/<artifact_id>', methods=['DELETE'])
@token_required
def delete_artifact(current_user, artifact_id):
    """删除工件"""
    try:
        tenant_id = _get_tenant_id()
        
        management_service = get_management_service()
        if management_service:
            success, error_msg = management_service.delete_artifact(
                artifact_id, current_user.get('id'), tenant_id
            )
        else:
            return jsonify({
                'success': False,
                'error': '管理服务不可用'
            }), 500
        
        if success:
            return jsonify({
                'success': True,
                'message': '工件删除成功'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
            
    except Exception as e:
        logger.error(f"删除工件失败: {e}")
        return jsonify({
            'success': False,
            'error': f'删除工件失败: {str(e)}'
        }), 500


@artifact_security_bp.route('/artifacts/<artifact_id>/versions', methods=['GET'])
@token_required
def get_artifact_versions(current_user, artifact_id):
    """获取工件版本列表"""
    try:
        tenant_id = _get_tenant_id()
        include_deleted = request.args.get('include_deleted', 'false').lower() == 'true'
        
        management_service = get_management_service()
        if management_service:
            versions = management_service.get_artifact_versions(
                artifact_id, current_user.get('id'), tenant_id, include_deleted
            )
        else:
            service = get_security_service()
            versions = service.get_artifact_versions(artifact_id, current_user.get('id'), tenant_id)
        
        return jsonify({
            'success': True,
            'data': [
                {
                    'id': v.id,
                    'version': v.version,
                    'status': v.status.value if hasattr(v.status, 'value') else v.status,
                    'file_size': v.file_size,
                    'file_hash': v.file_hash,
                    'mime_type': v.mime_type,
                    'changelog': v.changelog,
                    'created_by': v.created_by,
                    'created_at': v.created_at.isoformat() if v.created_at else None,
                    'tags': v.tags
                }
                for v in versions
            ],
            'count': len(versions)
        }), 200
        
    except Exception as e:
        logger.error(f"获取版本列表失败: {e}")
        return jsonify({
            'success': False,
            'error': f'获取版本列表失败: {str(e)}'
        }), 500


# ==================== 健康检查 ====================

@artifact_security_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        service = get_security_service()
        
        return jsonify({
            'success': True,
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'service': 'artifact_security'
        }), 200
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'success': False,
            'status': 'unhealthy',
            'error': str(e)
        }), 500


# ==================== 错误处理 ====================

@artifact_security_bp.errorhandler(400)
def bad_request(error):
    return jsonify({
        'success': False,
        'error': '请求参数错误'
    }), 400


@artifact_security_bp.errorhandler(401)
def unauthorized(error):
    return jsonify({
        'success': False,
        'error': '未授权访问'
    }), 401


@artifact_security_bp.errorhandler(403)
def forbidden(error):
    return jsonify({
        'success': False,
        'error': '权限不足'
    }), 403


@artifact_security_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': '资源不存在'
    }), 404


@artifact_security_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': '服务器内部错误'
    }), 500
