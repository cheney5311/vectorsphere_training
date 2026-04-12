"""文件上传工具

提供文件上传和处理功能。
"""

import os
import uuid
import hashlib
from typing import Optional, Dict, Any, Tuple
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage


def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """检查文件扩展名是否允许
    
    Args:
        filename: 文件名
        allowed_extensions: 允许的扩展名集合
        
    Returns:
        是否允许
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def save_uploaded_file(
    file: FileStorage, 
    upload_dir: str, 
    allowed_extensions: set,
    rename: bool = True,
    max_size: Optional[int] = None
) -> Tuple[bool, str, Optional[str]]:
    """保存上传的文件
    
    Args:
        file: 上传的文件对象
        upload_dir: 上传目录
        allowed_extensions: 允许的扩展名集合
        rename: 是否重命名文件
        max_size: 最大文件大小（字节）
        
    Returns:
        (成功标志, 文件路径, 错误信息)
    """
    try:
        # 检查文件是否存在
        if not file or not file.filename:
            return False, "", "未选择文件"
        
        # 检查文件大小
        if max_size and len(file.read()) > max_size:
            file.seek(0)  # 重置文件指针
            return False, "", f"文件大小超过限制 ({max_size} bytes)"
        file.seek(0)  # 重置文件指针
        
        # 检查文件扩展名
        if not allowed_file(file.filename, allowed_extensions):
            return False, "", f"不支持的文件类型，仅支持: {', '.join(allowed_extensions)}"
        
        # 确保上传目录存在
        os.makedirs(upload_dir, exist_ok=True)
        
        # 处理文件名
        if rename:
            # 生成安全的文件名
            filename = secure_filename(file.filename)
            # 添加UUID避免重复
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{uuid.uuid4().hex}{ext}"
        else:
            filename = secure_filename(file.filename)
        
        # 构建文件路径
        file_path = os.path.join(upload_dir, filename)
        
        # 保存文件
        file.save(file_path)
        
        return True, file_path, None
        
    except Exception as e:
        return False, "", f"保存文件失败: {str(e)}"


def get_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """计算文件哈希值
    
    Args:
        file_path: 文件路径
        algorithm: 哈希算法
        
    Returns:
        哈希值
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的哈希算法
    """
    try:
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
        
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {file_path}")
    except ValueError:
        raise ValueError(f"不支持的哈希算法: {algorithm}")


def get_file_info(file_path: str) -> Dict[str, Any]:
    """获取文件信息
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件信息字典
        
    Raises:
        FileNotFoundError: 文件不存在
    """
    try:
        stat = os.stat(file_path)
        
        return {
            'path': file_path,
            'size': stat.st_size,
            'created_time': stat.st_ctime,
            'modified_time': stat.st_mtime,
            'extension': os.path.splitext(file_path)[1],
            'hash': get_file_hash(file_path)
        }
        
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {file_path}")


def delete_file(file_path: str) -> bool:
    """删除文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        删除是否成功
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception:
        return False