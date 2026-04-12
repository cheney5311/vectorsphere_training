"""检查点管理器

提供模型训练检查点的保存、加载和管理功能。
"""

import os
import json
import torch
import pickle
from typing import Dict, Any, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self._ensure_checkpoint_dir()
    
    def _ensure_checkpoint_dir(self) -> None:
        """确保检查点目录存在"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        checkpoint_name: Optional[str] = None
    ) -> str:
        """保存检查点（原子写入并生成哈希校验）

        实现：
        - 先将 checkpoint_data 保存到临时文件
        - 计算文件的 SHA256 校验和并保存到同目录的 metadata 文件
        - 使用 os.replace 原子替换目标文件
        - 尝试将检查点上传到对象存储（若可用），并更新 metadata
        - 尽可能执行文件系统 sync
        """
        import hashlib
        try:
            # 生成检查点名称
            if not checkpoint_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_name = f"checkpoint_epoch_{epoch}_step_{step}_{timestamp}.pt"

            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            temp_path = checkpoint_path + ".tmp"
            meta_path = checkpoint_path + ".meta.json"

            # 构建检查点数据
            checkpoint_data = {
                'epoch': epoch,
                'step': step,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics or {},
                'model_state_dict': model.state_dict()
            }

            if optimizer:
                checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
            if scheduler:
                checkpoint_data['scheduler_state_dict'] = getattr(scheduler, 'state_dict', lambda: {})()

            # 写入临时文件
            torch.save(checkpoint_data, temp_path)

            # 计算 SHA256
            sha256 = hashlib.sha256()
            with open(temp_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            digest = sha256.hexdigest()

            # 写入元数据（原子替换方式），包含版本与校验信息
            from backend.utils.checkpoint_metadata import migrate_meta, get_meta_version
            meta = {
                'checkpoint': os.path.basename(checkpoint_path),
                'sha256': digest,
                'created_at': datetime.now().isoformat(),
                'meta_version': os.environ.get('CHECKPOINT_META_VERSION', '1.1'),
                'checksum_algo': 'sha256',
                'uploader': os.environ.get('CHECKPOINT_UPLOADER', 'checkpoint_manager')
            }
            # allow migrations for older version templates
            try:
                meta = migrate_meta(meta, target_version=get_meta_version(meta))
            except Exception:
                pass
            tmp_meta = meta_path + ".tmp"
            with open(tmp_meta, 'w') as mf:
                json.dump(meta, mf)
                mf.flush()
                try:
                    os.fsync(mf.fileno())
                except Exception:
                    pass
            os.replace(tmp_meta, meta_path)

            # 原子替换 checkpoint 文件
            os.replace(temp_path, checkpoint_path)

            # 尝试 flush 到磁盘
            try:
                fd = os.open(checkpoint_path, os.O_RDONLY)
                os.fsync(fd)
                os.close(fd)
            except Exception:
                pass

            # 尝试上传到对象存储（如果可用），但改为后台异步上传以避免阻塞训练线程
            try:
                bucket = os.environ.get('CHECKPOINT_BUCKET')
                if bucket:
                    try:
                        from backend.utils.checkpoint_storage import S3Adapter
                        adapter = S3Adapter(bucket=bucket)
                        object_name = os.path.basename(checkpoint_path)

                        # 将上传任务提交到后台线程池
                        import concurrent.futures

                        def _upload_task(cp_path, obj_name, bucket_name, meta_path_local, meta_dict):
                            max_retries = int(os.environ.get('CHECKPOINT_UPLOAD_RETRIES', '3'))
                            backoff_base = float(os.environ.get('CHECKPOINT_UPLOAD_BACKOFF', '0.5'))
                            upload_ok = False
                            last_exc = None
                            for attempt in range(1, max_retries + 1):
                                try:
                                    ok = adapter.upload(cp_path, object_name=obj_name)
                                    if ok:
                                        upload_ok = True
                                        break
                                except Exception as e:
                                    last_exc = e
                                try:
                                    import time as _time
                                    _time.sleep(backoff_base * (2 ** (attempt - 1)))
                                except Exception:
                                    pass

                            if upload_ok:
                                meta_update = {
                                    'bucket': bucket_name,
                                    'object_name': obj_name,
                                    'uploaded_at': datetime.now().isoformat(),
                                    'storage_upload_retries': attempt,
                                    'checksum_algo': 'sha256',
                                    'meta_version': '1.0',
                                    'uploader': os.environ.get('CHECKPOINT_UPLOADER', 'checkpoint_manager')
                                }
                                try:
                                    tmp_meta_local = meta_path_local + '.tmp'
                                    with open(tmp_meta_local, 'w') as mf:
                                        merged = dict(meta_dict)
                                        merged.update(meta_update)
                                        json.dump(merged, mf)
                                        mf.flush()
                                        os.fsync(mf.fileno())
                                    os.replace(tmp_meta_local, meta_path_local)
                                except Exception:
                                    logger.warning('Failed to update meta with storage info in background')
                            else:
                                logger.warning(f"Background checkpoint upload failed after {max_retries} attempts: {last_exc}")

                        try:
                            # 如果配置了持久化队列目录，则将上传任务入队到磁盘队列
                            from backend.utils.checkpoint_uploader import enqueue_upload
                            queue_dir = os.environ.get('CHECKPOINT_QUEUE_DIR')
                            if queue_dir:
                                enqueue_upload(checkpoint_path, object_name, bucket, meta_path, meta, queue_dir=queue_dir)
                            else:
                                # 否则使用线程池短期后台上传
                                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                                executor.submit(_upload_task, checkpoint_path, object_name, bucket, meta_path, meta)
                        except Exception as e:
                            logger.debug(f"Failed to schedule background upload: {e}")

                    except Exception as e:
                        logger.debug(f"Checkpoint upload skipped: {e}")
            except Exception:
                pass

            logger.info(f"Checkpoint saved atomically: {checkpoint_path}")
            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # 清理临时文件
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass
            raise
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """加载检查点并校验完整性
        - 如果存在同名 .meta.json，则校验 sha256
        """
        import hashlib
        try:
            # 检查文件是否存在
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            # 如果存在元数据文件，则验证哈希
            meta_path = checkpoint_path + ".meta.json"
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as mf:
                        meta = json.load(mf)
                    expected = meta.get('sha256')
                    if expected:
                        sha256 = hashlib.sha256()
                        with open(checkpoint_path, 'rb') as f:
                            for chunk in iter(lambda: f.read(8192), b""):
                                sha256.update(chunk)
                        actual = sha256.hexdigest()
                        if actual != expected:
                            raise ValueError(f"Checkpoint hash mismatch: expected {expected}, actual {actual}")
                except Exception as e:
                    logger.error(f"Checkpoint metadata validation failed: {e}")
                    raise

            # 加载检查点
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')

            # 加载模型状态
            model.load_state_dict(checkpoint_data['model_state_dict'])

            # 加载优化器状态
            if optimizer and 'optimizer_state_dict' in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

            # 加载调度器状态
            if scheduler and 'scheduler_state_dict' in checkpoint_data:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])

            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def save_model_weights(
        self,
        model: torch.nn.Module,
        filename: str,
        include_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> str:
        """保存模型权重
        
        Args:
            model: 模型
            filename: 文件名
            include_optimizer: 是否包含优化器状态
            optimizer: 优化器
            
        Returns:
            文件路径
        """
        try:
            filepath = os.path.join(self.checkpoint_dir, filename)
            
            save_data = {
                'model_state_dict': model.state_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
            if include_optimizer and optimizer:
                save_data['optimizer_state_dict'] = optimizer.state_dict()
            
            torch.save(save_data, filepath)
            
            logger.info(f"Model weights saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save model weights: {e}")
            raise
    
    def load_model_weights(
        self,
        filepath: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> None:
        """加载模型权重
        
        Args:
            filepath: 文件路径
            model: 模型
            optimizer: 优化器
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model weights not found: {filepath}")
            
            data = torch.load(filepath, map_location='cpu')
            model.load_state_dict(data['model_state_dict'])
            
            if optimizer and 'optimizer_state_dict' in data:
                optimizer.load_state_dict(data['optimizer_state_dict'])
            
            logger.info(f"Model weights loaded: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise
    
    def list_checkpoints(self) -> list:
        """列出所有检查点
        
        Returns:
            检查点文件列表
        """
        try:
            checkpoints = []
            for file in os.listdir(self.checkpoint_dir):
                if file.endswith('.pt') or file.endswith('.pth'):
                    filepath = os.path.join(self.checkpoint_dir, file)
                    stat = os.stat(filepath)
                    checkpoints.append({
                        'name': file,
                        'path': filepath,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
            
            # 按修改时间排序
            checkpoints.sort(key=lambda x: x['modified'], reverse=True)
            return checkpoints
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []
    
    async def get_latest_checkpoint(self, task_id: Optional[str] = None):
        """异步接口：获取最新的检查点（按修改时间）
        返回一个简单对象：{ "checkpoint_id": <name>, "path": <fullpath> }
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        latest = checkpoints[0]
        from types import SimpleNamespace
        return SimpleNamespace(checkpoint_id=latest['name'], path=latest['path'])

    async def validate_checkpoint(self, checkpoint_id: str) -> bool:
        """验证给定检查点的完整性（基于 .meta.json sha256）"""
        import hashlib
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_id)
        meta_path = checkpoint_path + ".meta.json"
        if not os.path.exists(checkpoint_path):
            return False
        if not os.path.exists(meta_path):
            # 无元数据，无法验证，认为无效
            return False
        try:
            with open(meta_path, 'r') as mf:
                meta = json.load(mf)
            expected = meta.get('sha256')
            if not expected:
                return False
            sha256 = hashlib.sha256()
            with open(checkpoint_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            actual = sha256.hexdigest()
            return actual == expected
        except Exception:
            return False

    async def restore_checkpoint(self, checkpoint_id: str) -> Union[bool, str]:
        """恢复检查点的占位实现
        - 验证检查点
        - 返回检查点路径或 False
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_id)
        if not os.path.exists(checkpoint_path):
            return False
        valid = await self.validate_checkpoint(checkpoint_id)
        if not valid:
            return False
        # 占位：实际恢复逻辑应将文件分发到目标节点或加载到内存
        return checkpoint_path

    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """删除检查点
        
        Args:
            checkpoint_name: 检查点名称
            
        Returns:
            是否删除成功
        """
        try:
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                logger.info(f"Checkpoint deleted: {checkpoint_path}")
                return True
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            return False
    
    def save_training_state(
        self,
        state: Dict[str, Any],
        filename: str = "training_state.pkl"
    ) -> str:
        """保存训练状态
        
        Args:
            state: 训练状态字典
            filename: 文件名
            
        Returns:
            文件路径
        """
        try:
            filepath = os.path.join(self.checkpoint_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"Training state saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save training state: {e}")
            raise
    
    def load_training_state(self, filename: str = "training_state.pkl") -> Dict[str, Any]:
        """加载训练状态
        
        Args:
            filename: 文件名
            
        Returns:
            训练状态字典
        """
        try:
            filepath = os.path.join(self.checkpoint_dir, filename)
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Training state not found: {filepath}")
            
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            logger.info(f"Training state loaded: {filepath}")
            return state
            
        except Exception as e:
            logger.error(f"Failed to load training state: {e}")
            raise
    
    def get_best_checkpoint(self, metric_name: str = 'accuracy', mode: str = 'max') -> Optional[str]:
        """获取最佳检查点
        
        Args:
            metric_name: 指标名称
            mode: 比较模式 ('max' 或 'min')
            
        Returns:
            最佳检查点路径
        """
        try:
            checkpoints = self.list_checkpoints()
            if not checkpoints:
                return None
            
            best_checkpoint = None
            best_value = None
            
            for checkpoint in checkpoints:
                # 这里需要从检查点文件中读取指标值
                # 简化处理，实际应该解析检查点文件
                pass
            
            return best_checkpoint
            
        except Exception as e:
            logger.error(f"Failed to get best checkpoint: {e}")
            return None


# 全局检查点管理器实例
_global_checkpoint_manager = None


def get_checkpoint_manager(checkpoint_dir: str = "./checkpoints") -> CheckpointManager:
    """获取全局检查点管理器实例

    Args:
        checkpoint_dir: 检查点目录

    Returns:
        CheckpointManager实例
    """
    global _global_checkpoint_manager
    if _global_checkpoint_manager is None:
        _global_checkpoint_manager = CheckpointManager(checkpoint_dir)
    return _global_checkpoint_manager


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    checkpoint_name: Optional[str] = None,
    checkpoint_dir: str = "./checkpoints"
) -> str:
    """保存检查点的便捷函数
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前轮次
        step: 当前步数
        metrics: 评估指标
        checkpoint_name: 检查点名称
        checkpoint_dir: 检查点目录
        
    Returns:
        检查点文件路径
    """
    manager = get_checkpoint_manager(checkpoint_dir)
    return manager.save_checkpoint(
        model, optimizer, scheduler, epoch, step, metrics, checkpoint_name
    )


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    checkpoint_dir: str = "./checkpoints"
) -> Dict[str, Any]:
    """加载检查点的便捷函数
    
    Args:
        checkpoint_path: 检查点文件路径
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        checkpoint_dir: 检查点目录
        
    Returns:
        检查点数据
    """
    manager = get_checkpoint_manager(checkpoint_dir)
    return manager.load_checkpoint(checkpoint_path, model, optimizer, scheduler)


def generate_presigned_url_for_checkpoint(checkpoint_filename: str, expires_in: int = 3600) -> Optional[str]:
    """为指定检查点生成带签名的访问 URL

    - 优先从对应的 .meta.json 中读取 storage 信息（bucket/object_name），
      若存在且配置了 S3Adapter，则返回 S3 的 presigned URL。
    - 否则，当本地文件存在时返回 file:// 路径；否则返回 None。
    """
    try:
        checkpoint_path = os.path.join(get_checkpoint_manager().checkpoint_dir, checkpoint_filename)
        meta_path = checkpoint_path + '.meta.json'
        # 若 meta 中包含 storage 信息，优先生成 presigned URL
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as mf:
                    meta = json.load(mf)
                bucket = meta.get('bucket')
                object_name = meta.get('object_name') or meta.get('checkpoint')
                if bucket and object_name:
                    from backend.utils.checkpoint_storage import S3Adapter
                    adapter = S3Adapter(bucket=bucket)
                    url = adapter.generate_presigned_url(object_name, expires_in=expires_in)
                    if url:
                        return url
            except Exception:
                pass
        # 回退：若本地文件存在，返回 file:// URL
        if os.path.exists(checkpoint_path):
            return 'file://' + checkpoint_path
        return None
    except Exception:
        return None