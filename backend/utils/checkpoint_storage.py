"""统一的检查点对象存储适配器抽象与实现。

提供一个轻量封装 `S3Adapter`，基于 `backend.modules.checkpoint.storage_s3`，在没有真实 S3/MinIO 时可自动回退到本地 mock 存储。

接口：
- upload(object_name, local_path, bucket=None) -> bool
- download(object_name, local_path, bucket=None) -> bool
- list_versions(prefix, bucket=None) -> list
- generate_presigned_url(object_name, bucket=None, expires_in=3600) -> str

注意：本模块应对上层代码（如 `CheckpointManager`）透明。上层只需调用 `S3Adapter(...).upload(...)` 等方法。
"""
from typing import Optional, List, Dict
import logging
import os

logger = logging.getLogger(__name__)

try:
    from backend.modules.checkpoint import storage_s3 as _storage_s3
except Exception:
    _storage_s3 = None


class S3Adapter:
    def __init__(self, bucket: Optional[str] = None):
        self.bucket = bucket or os.getenv('CHECKPOINT_S3_BUCKET', 'vectorsphere-checkpoints')
        # If storage module unavailable, fall back to mock behavior via storage_s3 (itself handles mock)
        if _storage_s3 is None:
            raise RuntimeError('checkpoint.storage_s3 module not available')

    def upload(self, object_name: str, local_path: str, bucket: Optional[str] = None) -> bool:
        target_bucket = bucket or self.bucket
        try:
            meta = _storage_s3.upload_file(local_path, target_bucket, object_name)
            logger.info(f"Uploaded checkpoint {object_name} to {target_bucket}: {meta}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload checkpoint {object_name} to {target_bucket}: {e}")
            return False

    def download(self, object_name: str, local_path: str, bucket: Optional[str] = None) -> bool:
        target_bucket = bucket or self.bucket
        try:
            _storage_s3.download_file(target_bucket, object_name, local_path)
            logger.info(f"Downloaded checkpoint {object_name} from {target_bucket} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download checkpoint {object_name} from {target_bucket}: {e}")
            return False

    def list_versions(self, prefix: str, bucket: Optional[str] = None) -> List[Dict[str, Optional[str]]]:
        target_bucket = bucket or self.bucket
        try:
            return _storage_s3.list_versions(target_bucket, prefix)
        except Exception as e:
            logger.error(f"Failed to list versions for prefix {prefix} in {target_bucket}: {e}")
            return []

    def generate_presigned_url(self, object_name: str, bucket: Optional[str] = None, expires_in: int = 3600) -> str:
        target_bucket = bucket or self.bucket
        try:
            return _storage_s3.generate_presigned_url(target_bucket, object_name, expires_in)
        except Exception as e:
            logger.error(f"Failed to generate presigned url for {object_name} in {target_bucket}: {e}")
            raise
