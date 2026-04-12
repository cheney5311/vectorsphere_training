"""S3 / Object storage adapter for checkpoint storage.

- Uses boto3 when available and configured via environment variables.
- Falls back to local filesystem mock when boto3 unavailable or MOCK_OBJECT_STORAGE=1.

API:
- upload_file(local_path, bucket, key) -> dict (e.g., {"etag":..., "version_id":...})
- download_file(bucket, key, local_path)
- list_versions(bucket, key_prefix) -> list of versions
- generate_presigned_url(bucket, key, expires_in=3600) -> url
"""
import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# 尝试导入 botocore 异常类，失败则创建占位符
from botocore.exceptions import BotoCoreError, ClientError

USE_MOCK = os.getenv("MOCK_OBJECT_STORAGE", "0") == "1"
_s3_client = None

if not USE_MOCK:
    try:
        import boto3
        _s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION') or None,
            endpoint_url=os.getenv('S3_ENDPOINT') or None,
        )
        logger.info("S3 adapter: boto3 client initialized")
    except Exception:
        logger.warning("boto3 not available or failed to init; switching to local mock storage")
        USE_MOCK = True

# Local mock directory
_MOCK_ROOT = os.getenv('MOCK_STORAGE_ROOT', '/tmp/vectorsphere_object_storage')
os.makedirs(_MOCK_ROOT, exist_ok=True)


def _mock_path(bucket: str, key: str) -> str:
    safe_bucket = bucket.replace('/', '_')
    return os.path.join(_MOCK_ROOT, safe_bucket, key)


def upload_file(local_path: str, bucket: str, key: str) -> Dict[str, Optional[str]]:
    """Upload file to object storage (or mock). Returns metadata dict."""
    logger.debug(f"upload_file: {local_path} -> {bucket}/{key}")
    if USE_MOCK:
        dest = _mock_path(bucket, key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(local_path, 'rb') as fr, open(dest, 'wb') as fw:
            fw.write(fr.read())
        logger.info(f"Mock upload completed: {bucket}/{key}")
        return {"etag": None, "version_id": None}

    try:
        resp = _s3_client.upload_file(local_path, bucket, key)
        # boto3 upload_file returns None on success; fetch head_object for metadata
        head = _s3_client.head_object(Bucket=bucket, Key=key)
        return {"etag": head.get('ETag'), "version_id": head.get('VersionId')}
    except (BotoCoreError, ClientError) as e:
        logger.error(f"S3 upload failed: {e}")
        raise


def download_file(bucket: str, key: str, local_path: str) -> None:
    logger.debug(f"download_file: {bucket}/{key} -> {local_path}")
    if USE_MOCK:
        src = _mock_path(bucket, key)
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(src, 'rb') as fr, open(local_path, 'wb') as fw:
            fw.write(fr.read())
        return

    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        _s3_client.download_file(bucket, key, local_path)
    except (BotoCoreError, ClientError) as e:
        logger.error(f"S3 download failed: {e}")
        raise


def list_versions(bucket: str, key_prefix: str) -> List[Dict[str, str]]:
    logger.debug(f"list_versions: {bucket} prefix={key_prefix}")
    if USE_MOCK:
        root = os.path.join(_MOCK_ROOT, bucket)
        out = []
        if not os.path.exists(root):
            return out
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                if rel.startswith(key_prefix):
                    out.append({"key": rel, "version_id": None})
        return out

    try:
        resp = _s3_client.list_object_versions(Bucket=bucket, Prefix=key_prefix)
        versions = resp.get('Versions', []) + resp.get('DeleteMarkers', [])
        return [{"key": v['Key'], "version_id": v.get('VersionId')} for v in versions]
    except Exception as e:
        logger.error(f"S3 list versions failed: {e}")
        raise


def generate_presigned_url(bucket: str, key: str, expires_in: int = 3600) -> str:
    logger.debug(f"generate_presigned_url: {bucket}/{key} expires={expires_in}")
    if USE_MOCK:
        # For mock, return a file:// URL
        path = _mock_path(bucket, key)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return f"file://{path}"

    try:
        return _s3_client.generate_presigned_url('get_object', Params={'Bucket': bucket, 'Key': key}, ExpiresIn=expires_in)
    except Exception as e:
        logger.error(f"S3 presign failed: {e}")
        raise
