#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型下载 API

提供模型下载相关的 REST API 接口：
- 获取下载信息
- 生成下载链接
- 下载模型文件
- 下载训练模型
- 下载统计
"""

import logging
import os
from functools import wraps
from datetime import datetime

from flask import Blueprint, request, send_file, jsonify, g
from flask_jwt_extended import jwt_required, get_jwt_identity

from backend.utils.response import success_response, error_response
from backend.services.model_download_service import get_download_service, ModelDownloadService

logger = logging.getLogger(__name__)

# 创建蓝图
model_download_bp = Blueprint('model_download', __name__, url_prefix='/api/v1/models')

# 下载服务（懒加载）
_download_service: ModelDownloadService = None


def get_service() -> ModelDownloadService:
    """获取下载服务实例"""
    global _download_service
    if _download_service is None:
        _download_service = get_download_service(use_memory=True)
    return _download_service


def get_tenant_id() -> str:
    """获取当前租户ID"""
    return request.headers.get('X-Tenant-ID') or getattr(g, 'tenant_id', None)


def handle_download_errors(f):
    """下载API错误处理装饰器"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except FileNotFoundError as e:
            logger.warning(f"File not found in {f.__name__}: {e}")
            return error_response("模型文件不存在", 404)
        except PermissionError as e:
            logger.warning(f"Permission denied in {f.__name__}: {e}")
            return error_response("无权限下载该模型", 403)
        except ValueError as e:
            logger.warning(f"Validation error in {f.__name__}: {e}")
            return error_response(str(e), 400)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {e}", exc_info=True)
            return error_response(f"服务器内部错误: {str(e)}", 500)
    return wrapper


# ==================== 下载信息接口 ====================

@model_download_bp.route('/<model_id>/info', methods=['GET'])
@jwt_required()
@handle_download_errors
def get_model_download_info(model_id: str):
    """获取模型下载信息
    
    Path Parameters:
        model_id: 模型ID
        
    Returns:
        {
            "model_id": "string",
            "model_name": "string",
            "version": "string",
            "status": "string",
            "file_size": "integer",
            "available_formats": ["string"],
            "download_count": "integer",
            "last_downloaded": "string",
            "created_at": "string"
        }
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    service = get_service()
    info = service.get_download_info(model_id, user_id, tenant_id)
    
    return success_response(info, "获取下载信息成功")


# ==================== 下载链接接口 ====================

@model_download_bp.route('/<model_id>/download-url', methods=['GET'])
@jwt_required()
@handle_download_errors
def get_model_download_url(model_id: str):
    """获取模型下载URL（临时下载链接）
    
    Path Parameters:
        model_id: 模型ID
        
    Query Parameters:
        format: 下载格式 (default: pytorch)
        version: 模型版本 (default: latest)
        expire_hours: 过期时间（小时）(default: 24, max: 168)
        
    Returns:
        {
            "download_id": "string",
            "download_url": "string",
            "download_token": "string",
            "expire_at": "string",
            "format": "string",
            "file_name": "string",
            "file_size": "integer",
            "checksum": "string"
        }
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    # 获取查询参数
    download_format = request.args.get('format', 'pytorch')
    version = request.args.get('version')
    expire_hours = request.args.get('expire_hours', 24, type=int)
    
    service = get_service()
    result = service.generate_download_url(
        model_id=model_id,
        user_id=user_id,
        download_format=download_format,
        version=version,
        expire_hours=expire_hours,
        tenant_id=tenant_id
    )
    
    return success_response(result, "获取下载链接成功")


@model_download_bp.route('/<model_id>/download-url', methods=['POST'])
@jwt_required()
@handle_download_errors
def create_model_download_url(model_id: str):
    """创建模型下载URL（支持更多配置）
    
    Path Parameters:
        model_id: 模型ID
        
    Request Body:
        format: 下载格式 (default: pytorch)
        version: 模型版本
        expire_hours: 过期时间（小时）(default: 24)
        
    Returns:
        下载链接信息
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    data = request.get_json() or {}
    
    download_format = data.get('format', 'pytorch')
    version = data.get('version')
    expire_hours = data.get('expire_hours', 24)
    
    service = get_service()
    result = service.generate_download_url(
        model_id=model_id,
        user_id=user_id,
        download_format=download_format,
        version=version,
        expire_hours=expire_hours,
        tenant_id=tenant_id
    )
    
    return success_response(result, "创建下载链接成功", 201)


# ==================== 文件下载接口 ====================

@model_download_bp.route('/<model_id>/download', methods=['GET'])
@jwt_required()
@handle_download_errors
def download_model(model_id: str):
    """下载模型文件
    
    Path Parameters:
        model_id: 模型ID
        
    Query Parameters:
        format: 下载格式 (default: pytorch)
        token: 下载令牌（可选，用于验证预生成的下载链接）
        version: 模型版本 (default: latest)
        
    Returns:
        模型文件下载流
    """
    user_id = get_jwt_identity()
    
    # 获取查询参数
    download_format = request.args.get('format', 'pytorch')
    download_token = request.args.get('token')
    
    # 获取客户端信息
    download_ip = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')
    
    service = get_service()
    
    # 准备下载
    file_info = service.prepare_download(
        model_id=model_id,
        user_id=user_id,
        download_format=download_format,
        download_token=download_token,
        download_ip=download_ip,
        user_agent=user_agent
    )
    
    file_path = file_info['file_path']
    file_name = file_info['file_name']
    mime_type = file_info['mime_type']
    
    logger.info(f"Downloading model {model_id}, format: {download_format}, user: {user_id}")
    
    return send_file(
        file_path,
        as_attachment=True,
        download_name=file_name,
        mimetype=mime_type
    )


# ==================== 训练模型下载接口 ====================

@model_download_bp.route('/training/<session_id>/info', methods=['GET'])
@jwt_required()
@handle_download_errors
def get_training_model_info(session_id: str):
    """获取训练模型下载信息
    
    Path Parameters:
        session_id: 训练会话ID
        
    Returns:
        训练模型下载信息
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    service = get_service()
    info = service.get_training_model_download_info(session_id, user_id, tenant_id)
    
    return success_response(info, "获取训练模型信息成功")


@model_download_bp.route('/training/<session_id>/download', methods=['GET'])
@jwt_required()
@handle_download_errors
def download_trained_model(session_id: str):
    """下载训练完成的模型
    
    Path Parameters:
        session_id: 训练会话ID
        
    Query Parameters:
        format: 下载格式 (default: pytorch)
        
    Returns:
        训练模型文件下载流
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    # 获取查询参数
    download_format = request.args.get('format', 'pytorch')
    
    service = get_service()
    
    # 准备下载
    file_info = service.prepare_training_model_download(
        session_id=session_id,
        user_id=user_id,
        download_format=download_format,
        tenant_id=tenant_id
    )
    
    file_path = file_info['file_path']
    file_name = file_info['file_name']
    mime_type = file_info['mime_type']
    
    logger.info(f"Downloading training model {session_id}, format: {download_format}, user: {user_id}")
    
    return send_file(
        file_path,
        as_attachment=True,
        download_name=file_name,
        mimetype=mime_type
    )


@model_download_bp.route('/training/<session_id>/download-url', methods=['GET'])
@jwt_required()
@handle_download_errors
def get_training_model_download_url(session_id: str):
    """获取训练模型下载URL
    
    Path Parameters:
        session_id: 训练会话ID
        
    Query Parameters:
        format: 下载格式 (default: pytorch)
        expire_hours: 过期时间（小时）(default: 24)
        
    Returns:
        下载链接信息
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    download_format = request.args.get('format', 'pytorch')
    expire_hours = request.args.get('expire_hours', 24, type=int)
    
    # 获取训练模型信息
    service = get_service()
    info = service.get_training_model_download_info(session_id, user_id, tenant_id)
    
    # 构建下载URL
    download_url = f"/api/v1/models/training/{session_id}/download?format={download_format}"
    
    expire_at = datetime.utcnow()
    from datetime import timedelta
    expire_at = expire_at + timedelta(hours=expire_hours)
    
    return success_response({
        'session_id': session_id,
        'download_url': download_url,
        'expire_at': expire_at.isoformat(),
        'format': download_format,
        'file_name': f"{info['model_name']}.pt",
        'file_size': info.get('file_size', 0)
    }, "获取下载链接成功")


# ==================== 统计接口 ====================

@model_download_bp.route('/<model_id>/download-stats', methods=['GET'])
@jwt_required()
@handle_download_errors
def get_download_statistics(model_id: str):
    """获取模型下载统计
    
    Path Parameters:
        model_id: 模型ID
        
    Query Parameters:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        
    Returns:
        下载统计信息
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    # 解析日期参数
    start_date = None
    end_date = None
    
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')
    
    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        except ValueError:
            return error_response("Invalid start_date format. Use YYYY-MM-DD", 400)
    
    if end_date_str:
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        except ValueError:
            return error_response("Invalid end_date format. Use YYYY-MM-DD", 400)
    
    service = get_service()
    stats = service.get_download_statistics(
        model_id=model_id,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        tenant_id=tenant_id
    )
    
    return success_response(stats, "获取下载统计成功")


@model_download_bp.route('/download-history', methods=['GET'])
@jwt_required()
@handle_download_errors
def get_download_history():
    """获取用户下载历史
    
    Query Parameters:
        limit: 返回数量 (default: 50)
        offset: 偏移量 (default: 0)
        model_id: 模型ID过滤（可选）
        
    Returns:
        下载历史列表
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    model_id = request.args.get('model_id')
    
    service = get_service()
    result = service.get_user_download_history(
        user_id=user_id,
        limit=limit,
        offset=offset,
        model_id=model_id,
        tenant_id=tenant_id
    )
    
    return success_response(result, "获取下载历史成功")


# ==================== 格式支持接口 ====================

@model_download_bp.route('/supported-formats', methods=['GET'])
def get_supported_formats():
    """获取支持的下载格式
    
    Returns:
        支持的格式列表
    """
    service = get_service()
    
    formats = [
        {
            'format': 'pytorch',
            'extension': '.pt',
            'description': 'PyTorch 模型格式',
            'mime_type': 'application/octet-stream'
        },
        {
            'format': 'onnx',
            'extension': '.onnx',
            'description': 'ONNX 通用格式',
            'mime_type': 'application/octet-stream'
        },
        {
            'format': 'tensorflow',
            'extension': '.pb',
            'description': 'TensorFlow SavedModel 格式',
            'mime_type': 'application/x-protobuf'
        },
        {
            'format': 'torchscript',
            'extension': '.pt',
            'description': 'TorchScript 格式',
            'mime_type': 'application/octet-stream'
        },
        {
            'format': 'safetensors',
            'extension': '.safetensors',
            'description': 'SafeTensors 安全格式',
            'mime_type': 'application/octet-stream'
        },
        {
            'format': 'checkpoint',
            'extension': '.ckpt',
            'description': '检查点格式',
            'mime_type': 'application/octet-stream'
        }
    ]
    
    return success_response({
        'formats': formats,
        'default': 'pytorch'
    })


# ==================== 批量下载接口 ====================

@model_download_bp.route('/batch/download-urls', methods=['POST'])
@jwt_required()
@handle_download_errors
def batch_generate_download_urls():
    """批量生成下载URL
    
    Request Body:
        model_ids: 模型ID列表
        format: 下载格式 (default: pytorch)
        expire_hours: 过期时间（小时）(default: 24)
        
    Returns:
        下载链接列表
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    data = request.get_json() or {}
    
    model_ids = data.get('model_ids', [])
    download_format = data.get('format', 'pytorch')
    expire_hours = data.get('expire_hours', 24)
    
    if not model_ids:
        return error_response("model_ids 不能为空", 400)
    
    if len(model_ids) > 10:
        return error_response("一次最多生成10个下载链接", 400)
    
    service = get_service()
    results = []
    
    for model_id in model_ids:
        try:
            result = service.generate_download_url(
                model_id=model_id,
                user_id=user_id,
                download_format=download_format,
                expire_hours=expire_hours,
                tenant_id=tenant_id
            )
            results.append({
                'model_id': model_id,
                'success': True,
                'data': result
            })
        except Exception as e:
            results.append({
                'model_id': model_id,
                'success': False,
                'error': str(e)
            })
    
    success_count = sum(1 for r in results if r['success'])
    
    return success_response({
        'results': results,
        'total': len(model_ids),
        'success': success_count,
        'failed': len(model_ids) - success_count
    })


# ==================== 健康检查接口 ====================

@model_download_bp.route('/download/health', methods=['GET'])
def download_service_health():
    """下载服务健康检查
    
    Returns:
        服务健康状态
    """
    try:
        service = get_service()
        
        return success_response({
            'status': 'healthy',
            'service': 'model_download_service',
            'supported_formats': service.SUPPORTED_FORMATS,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return error_response(f'服务不健康: {str(e)}', 503)


# ==================== 初始化和清理 ====================

def init_download_api(app=None):
    """初始化下载API
    
    Args:
        app: Flask应用实例
    """
    if app:
        app.register_blueprint(model_download_bp)
    
    logger.info("Model download API initialized")
    return model_download_bp


def cleanup_download_api():
    """清理下载API资源"""
    global _download_service
    _download_service = None
    logger.info("Model download API cleanup completed")


__all__ = [
    'model_download_bp',
    'init_download_api',
    'cleanup_download_api',
]
