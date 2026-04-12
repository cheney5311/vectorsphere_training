#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型下载服务

提供模型下载的完整业务逻辑：
- 下载链接生成
- 文件格式转换
- 下载权限验证
- 下载统计
- 文件完整性校验
"""

import hashlib
import logging
import os
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ModelDownloadService:
    """模型下载服务
    
    提供模型下载相关的业务功能
    """
    
    # 支持的下载格式
    SUPPORTED_FORMATS = ['pytorch', 'onnx', 'tensorflow', 'torchscript', 'safetensors', 'checkpoint']
    
    # 格式文件扩展名映射
    FORMAT_EXTENSIONS = {
        'pytorch': '.pt',
        'onnx': '.onnx',
        'tensorflow': '.pb',
        'torchscript': '.pt',
        'safetensors': '.safetensors',
        'checkpoint': '.ckpt',
    }
    
    # 格式 MIME 类型映射
    FORMAT_MIME_TYPES = {
        'pytorch': 'application/octet-stream',
        'onnx': 'application/octet-stream',
        'tensorflow': 'application/x-protobuf',
        'torchscript': 'application/octet-stream',
        'safetensors': 'application/octet-stream',
        'checkpoint': 'application/octet-stream',
    }
    
    def __init__(self, use_memory: bool = True):
        """初始化服务
        
        Args:
            use_memory: 是否使用内存存储
        """
        self._use_memory = use_memory
        self._lock = threading.RLock()
        
        # 初始化仓库
        self._download_repo = None
        self._model_service = None
        self._training_history_service = None
        
        self._init_dependencies()
        
        # 下载令牌密钥
        self._secret_key = os.environ.get('DOWNLOAD_SECRET_KEY', 'vectorsphere_download_secret')
        
        # 存储基础路径
        self._storage_base = os.environ.get('MODEL_STORAGE_PATH', '/var/lib/vectorsphere/models')
        
        logger.info("ModelDownloadService initialized")
    
    def _init_dependencies(self):
        """初始化依赖服务"""
        try:
            from backend.repositories.model_download_repository import get_download_repository
            self._download_repo = get_download_repository(self._use_memory)
        except ImportError as e:
            logger.warning(f"Failed to import download repository: {e}")
        
        try:
            from backend.services.model_service import ModelService
            from backend.repositories.model_repository import ModelRepository
            self._model_service = ModelService(ModelRepository())
        except ImportError as e:
            logger.warning(f"Failed to import model service: {e}")
        
        try:
            from backend.services.training_history_service import get_training_history_service
            self._training_history_service = get_training_history_service()
        except ImportError as e:
            logger.warning(f"Failed to import training history service: {e}")
    
    # ==================== 核心下载功能 ====================
    
    def get_download_info(
        self,
        model_id: str,
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取模型下载信息
        
        Args:
            model_id: 模型ID
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            下载信息
        """
        # 获取模型信息
        model = self._get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # 检查权限
        if not self._check_download_permission(model, user_id):
            raise PermissionError("No permission to download this model")
        
        # 获取文件路径和大小
        file_path = self._get_model_file_path(model, user_id)
        file_size = 0
        if file_path and os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
        
        # 检查可用格式
        available_formats = self._get_available_formats(file_path)
        
        # 获取下载统计
        stats = {}
        if self._download_repo:
            stats = self._download_repo.get_model_statistics(model_id)
        
        return {
            'model_id': model_id,
            'model_name': getattr(model, 'name', 'Unknown'),
            'version': getattr(model, 'version', '1.0.0'),
            'status': getattr(model, 'status', 'unknown'),
            'file_size': file_size,
            'available_formats': available_formats,
            'download_count': stats.get('total_downloads', 0),
            'last_downloaded': stats.get('last_download_at'),
            'created_at': model.created_at.isoformat() if hasattr(model, 'created_at') and model.created_at else None,
        }
    
    def generate_download_url(
        self,
        model_id: str,
        user_id: str,
        download_format: str = 'pytorch',
        version: Optional[str] = None,
        expire_hours: int = 24,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """生成下载URL
        
        Args:
            model_id: 模型ID
            user_id: 用户ID
            download_format: 下载格式
            version: 版本号
            expire_hours: 过期时间（小时）
            tenant_id: 租户ID
            
        Returns:
            下载链接信息
        """
        # 验证格式
        if download_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {download_format}. Supported: {self.SUPPORTED_FORMATS}")
        
        # 验证过期时间
        if expire_hours < 1 or expire_hours > 168:  # 最长7天
            raise ValueError("expire_hours must be between 1 and 168")
        
        # 获取模型信息
        model = self._get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # 检查权限
        if not self._check_download_permission(model, user_id):
            raise PermissionError("No permission to download this model")
        
        # 检查模型状态
        model_status = getattr(model, 'status', None)
        if model_status not in ['ready', 'deployed', 'validated', 'trained']:
            raise ValueError(f"Model is not ready for download. Current status: {model_status}")
        
        # 获取文件路径
        file_path = self._get_model_file_path(model, user_id, download_format)
        if not file_path:
            raise ValueError("Model file not found")
        
        # 检查是否有活动的下载链接
        if self._download_repo:
            existing = self._download_repo.get_active_download(model_id, user_id, download_format)
            if existing:
                return {
                    'download_id': existing['id'],
                    'download_url': existing.get('download_url') or f"/api/v1/models/{model_id}/download?format={download_format}&token={existing.get('download_token', '')}",
                    'download_token': existing.get('download_token'),
                    'expire_at': existing.get('expire_at'),
                    'format': download_format,
                    'file_name': existing.get('file_name'),
                    'file_size': existing.get('file_size', 0),
                    'reused': True
                }
        
        # 计算过期时间
        expire_at = datetime.utcnow() + timedelta(hours=expire_hours)
        
        # 生成下载令牌
        download_token = self._generate_download_token(model_id, user_id, expire_at)
        
        # 获取文件信息
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        file_name = self._generate_download_filename(model, download_format)
        checksum = self._calculate_checksum(file_path) if os.path.exists(file_path) else None
        
        # 生成下载URL
        download_url = f"/api/v1/models/{model_id}/download?format={download_format}&token={download_token}"
        
        # 创建下载记录
        record = None
        if self._download_repo:
            record = self._download_repo.create_download_record(
                model_id=model_id,
                user_id=user_id,
                download_format=download_format,
                download_source='model',
                file_path=file_path,
                file_size=file_size,
                file_name=file_name,
                checksum=checksum,
                download_token=download_token,
                expire_at=expire_at,
                tenant_id=tenant_id
            )
            
            # 更新状态为就绪
            self._download_repo.update_download_record(record['id'], status='ready', download_url=download_url)
        
        logger.info(f"Generated download URL for model {model_id}, format: {download_format}")
        
        return {
            'download_id': record['id'] if record else str(uuid.uuid4()),
            'download_url': download_url,
            'download_token': download_token,
            'expire_at': expire_at.isoformat(),
            'format': download_format,
            'file_name': file_name,
            'file_size': file_size,
            'checksum': checksum,
            'reused': False
        }
    
    def prepare_download(
        self,
        model_id: str,
        user_id: str,
        download_format: str = 'pytorch',
        download_token: Optional[str] = None,
        download_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """准备下载（验证并返回文件信息）
        
        Args:
            model_id: 模型ID
            user_id: 用户ID
            download_format: 下载格式
            download_token: 下载令牌
            download_ip: 下载IP
            user_agent: 用户代理
            
        Returns:
            文件信息
        """
        # 验证令牌
        if download_token:
            record = self._validate_download_token(download_token, model_id, user_id)
            if not record:
                raise ValueError("Invalid or expired download token")
            
            # 更新下载计数
            if self._download_repo:
                self._download_repo.increment_download_count(
                    record['id'],
                    download_ip=download_ip,
                    user_agent=user_agent
                )
            
            file_path = record.get('file_path')
            file_name = record.get('file_name')
        else:
            # 无令牌，直接验证权限
            model = self._get_model(model_id)
            if not model:
                raise ValueError(f"Model {model_id} not found")
            
            if not self._check_download_permission(model, user_id):
                raise PermissionError("No permission to download this model")
            
            file_path = self._get_model_file_path(model, user_id, download_format)
            file_name = self._generate_download_filename(model, download_format)
        
        # 验证文件存在
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        mime_type = self.FORMAT_MIME_TYPES.get(download_format, 'application/octet-stream')
        
        return {
            'file_path': file_path,
            'file_name': file_name,
            'file_size': file_size,
            'mime_type': mime_type,
            'format': download_format
        }
    
    # ==================== 训练模型下载 ====================
    
    def get_training_model_download_info(
        self,
        session_id: str,
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取训练模型下载信息
        
        Args:
            session_id: 训练会话ID
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            下载信息
        """
        if not self._training_history_service:
            raise RuntimeError("Training history service not available")
        
        # 获取训练详情
        detail = self._training_history_service.get_training_history_detail(session_id, user_id, tenant_id)
        if not detail:
            raise ValueError(f"Training session {session_id} not found")
        
        # 检查状态
        if detail.get('status') != 'completed':
            raise ValueError("Training has not completed yet")
        
        # 获取输出路径
        result = detail.get('result', {})
        model_path = result.get('model_path') or result.get('output_path') or detail.get('outputPath')
        
        if not model_path:
            raise ValueError("Model output path not found")
        
        file_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
        available_formats = self._get_available_formats(model_path)
        
        return {
            'session_id': session_id,
            'model_name': detail.get('modelName', f'model_{session_id[:8]}'),
            'training_type': detail.get('trainingType', 'standard'),
            'status': detail.get('status'),
            'file_path': model_path,
            'file_size': file_size,
            'available_formats': available_formats,
            'completed_at': detail.get('endTime'),
            'accuracy': detail.get('accuracy'),
            'final_loss': detail.get('finalLoss'),
        }
    
    def prepare_training_model_download(
        self,
        session_id: str,
        user_id: str,
        download_format: str = 'pytorch',
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """准备训练模型下载
        
        Args:
            session_id: 训练会话ID
            user_id: 用户ID
            download_format: 下载格式
            tenant_id: 租户ID
            
        Returns:
            文件信息
        """
        info = self.get_training_model_download_info(session_id, user_id, tenant_id)
        
        file_path = info['file_path']
        
        # 检查格式是否可用
        if download_format not in info['available_formats']:
            if download_format != 'pytorch':
                raise ValueError(f"Format {download_format} not available for this model")
        
        # 转换格式（如果需要）
        if download_format != 'pytorch':
            converted_path = self._convert_model_format(file_path, download_format)
            if converted_path and os.path.exists(converted_path):
                file_path = converted_path
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        file_name = f"{info['model_name']}{self.FORMAT_EXTENSIONS.get(download_format, '.bin')}"
        file_size = os.path.getsize(file_path)
        mime_type = self.FORMAT_MIME_TYPES.get(download_format, 'application/octet-stream')
        
        # 记录下载
        if self._download_repo:
            self._download_repo.create_download_record(
                model_id=session_id,  # 使用 session_id 作为标识
                user_id=user_id,
                download_format=download_format,
                download_source='training',
                training_session_id=session_id,
                file_path=file_path,
                file_size=file_size,
                file_name=file_name,
                tenant_id=tenant_id
            )
        
        return {
            'file_path': file_path,
            'file_name': file_name,
            'file_size': file_size,
            'mime_type': mime_type,
            'format': download_format
        }
    
    # ==================== 统计功能 ====================
    
    def get_download_statistics(
        self,
        model_id: str,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取下载统计
        
        Args:
            model_id: 模型ID
            user_id: 用户ID
            start_date: 开始日期
            end_date: 结束日期
            tenant_id: 租户ID
            
        Returns:
            统计信息
        """
        # 验证模型权限
        model = self._get_model(model_id)
        if model and not self._check_download_permission(model, user_id):
            raise PermissionError("No permission to view statistics")
        
        if not self._download_repo:
            return {
                'model_id': model_id,
                'total_downloads': 0,
                'unique_users': 0,
                'format_breakdown': {},
                'message': 'Statistics not available'
            }
        
        return self._download_repo.get_model_statistics(model_id, start_date, end_date)
    
    def get_user_download_history(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        model_id: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取用户下载历史
        
        Args:
            user_id: 用户ID
            limit: 返回数量
            offset: 偏移量
            model_id: 模型ID过滤
            tenant_id: 租户ID
            
        Returns:
            下载历史
        """
        if not self._download_repo:
            return {
                'downloads': [],
                'total': 0,
                'limit': limit,
                'offset': offset
            }
        
        records, total = self._download_repo.list_user_downloads(user_id, limit, offset, model_id)
        
        return {
            'downloads': records,
            'total': total,
            'limit': limit,
            'offset': offset
        }
    
    # ==================== 私有方法 ====================
    
    def _get_model(self, model_id: str):
        """获取模型"""
        if self._model_service:
            return self._model_service.get_model(model_id)
        return None
    
    def _check_download_permission(self, model, user_id: str) -> bool:
        """检查下载权限"""
        if not model:
            return False
        
        # 检查所有权
        if getattr(model, 'user_id', None) == user_id:
            return True
        
        # 检查是否公开
        if getattr(model, 'is_public', False):
            return True
        
        return False
    
    def _get_model_file_path(
        self,
        model,
        user_id: str,
        download_format: str = 'pytorch'
    ) -> Optional[str]:
        """获取模型文件路径"""
        # 首先尝试从模型获取
        storage_path = getattr(model, 'storage_path', None)
        if storage_path and os.path.exists(storage_path):
            return storage_path
        
        # 尝试从训练历史获取
        if self._training_history_service:
            try:
                training_history = self._training_history_service.get_training_history(user_id, limit=100)
                model_id = getattr(model, 'id', None)
                model_name = getattr(model, 'name', None)
                
                for session in training_history.get('sessions', []):
                    if session.get('modelName') == model_id or session.get('modelName') == model_name:
                        if session.get('status') == 'completed':
                            result = session.get('result', {})
                            path = result.get('model_path') or result.get('output_path') or session.get('outputPath')
                            if path and os.path.exists(path):
                                return path
            except Exception as e:
                logger.warning(f"Failed to get model path from training history: {e}")
        
        return None
    
    def _get_available_formats(self, file_path: Optional[str]) -> List[str]:
        """获取可用的下载格式"""
        formats = ['pytorch']  # 默认支持 pytorch
        
        if not file_path:
            return formats
        
        base_path = os.path.splitext(file_path)[0]
        
        # 检查其他格式文件是否存在
        format_checks = {
            'onnx': '.onnx',
            'tensorflow': '.pb',
            'safetensors': '.safetensors',
            'checkpoint': '.ckpt',
        }
        
        for fmt, ext in format_checks.items():
            check_path = base_path + ext
            if os.path.exists(check_path):
                formats.append(fmt)
        
        return formats
    
    def _generate_download_token(self, model_id: str, user_id: str, expire_at: datetime) -> str:
        """生成下载令牌"""
        import hmac
        import base64
        
        data = f"{model_id}:{user_id}:{expire_at.timestamp()}"
        signature = hmac.new(
            self._secret_key.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        return base64.urlsafe_b64encode(signature).decode('utf-8').rstrip('=')
    
    def _validate_download_token(
        self,
        token: str,
        model_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """验证下载令牌"""
        if not self._download_repo:
            return None
        
        record = self._download_repo.get_by_token(token)
        if not record:
            return None
        
        # 验证模型ID
        if record.get('model_id') != model_id:
            return None
        
        # 验证用户ID
        if record.get('user_id') != user_id:
            return None
        
        # 验证过期时间
        expire_at_str = record.get('expire_at')
        if expire_at_str:
            expire_at = datetime.fromisoformat(expire_at_str)
            if datetime.utcnow() > expire_at:
                # 标记为过期
                self._download_repo.update_download_record(record['id'], status='expired')
                return None
        
        return record
    
    def _generate_download_filename(self, model, download_format: str) -> str:
        """生成下载文件名"""
        model_name = getattr(model, 'name', 'model')
        version = getattr(model, 'version', '1.0.0')
        extension = self.FORMAT_EXTENSIONS.get(download_format, '.bin')
        
        # 清理文件名中的非法字符
        safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in model_name)
        
        return f"{safe_name}_{version}{extension}"
    
    def _calculate_checksum(self, file_path: str, algorithm: str = 'sha256') -> str:
        """计算文件校验和"""
        if not os.path.exists(file_path):
            return ""
        
        hash_func = hashlib.sha256() if algorithm == 'sha256' else hashlib.md5()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def _convert_model_format(self, source_path: str, target_format: str) -> Optional[str]:
        """转换模型格式
        
        注意：这是一个简化实现，实际转换需要根据模型类型使用相应的转换库
        """
        if not os.path.exists(source_path):
            return None
        
        base_path = os.path.splitext(source_path)[0]
        target_ext = self.FORMAT_EXTENSIONS.get(target_format, '.bin')
        target_path = base_path + target_ext
        
        # 如果目标文件已存在，直接返回
        if os.path.exists(target_path):
            return target_path
        
        # 实际格式转换逻辑（需要根据具体框架实现）
        logger.warning(f"Model format conversion to {target_format} not implemented")
        return None


# ==================== 全局实例 ====================

_download_service: Optional[ModelDownloadService] = None


def get_download_service(use_memory: bool = True) -> ModelDownloadService:
    """获取下载服务实例
    
    Args:
        use_memory: 是否使用内存存储
        
    Returns:
        ModelDownloadService 实例
    """
    global _download_service
    
    if _download_service is None:
        _download_service = ModelDownloadService(use_memory)
    
    return _download_service


def reset_download_service():
    """重置下载服务实例"""
    global _download_service
    _download_service = None


__all__ = [
    'ModelDownloadService',
    'get_download_service',
    'reset_download_service',
]
