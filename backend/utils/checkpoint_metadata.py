"""Checkpoint metadata versioning and governance utilities.

- Current meta schema versions are supported and migrations provided.
- API:
  - validate_meta(meta: dict) -> bool
  - get_meta_version(meta: dict) -> str
  - migrate_meta(meta: dict, target_version: str) -> dict
  - canonicalize_meta_file(path: str) -> dict  # read and ensure version

- Minimal migrations implemented:
  - 1.0 -> 1.1: ensure fields `bucket` and `object_name` exist (may be None), add `meta_version` updated.
"""
from typing import Dict, Any
import json
import os
import logging

logger = logging.getLogger(__name__)

CURRENT_VERSION = '1.1'


def get_meta_version(meta: Dict[str, Any]) -> str:
    return str(meta.get('meta_version', meta.get('version', '1.0')))


def validate_meta(meta: Dict[str, Any]) -> bool:
    """Basic validation for known schema fields."""
    try:
        v = get_meta_version(meta)
        if v == '1.0':
            # require checkpoint and sha256
            return 'checkpoint' in meta and 'sha256' in meta
        elif v == '1.1':
            return 'checkpoint' in meta and 'sha256' in meta and 'meta_version' in meta
        else:
            # unknown versions are considered invalid
            return False
    except Exception:
        return False


def migrate_1_0_to_1_1(meta: Dict[str, Any]) -> Dict[str, Any]:
    new = dict(meta)
    # normalize keys
    new['meta_version'] = '1.1'
    # ensure bucket/object_name fields exist
    if 'bucket' not in new:
        new['bucket'] = None
    if 'object_name' not in new:
        # try derive from checkpoint filename
        ck = new.get('checkpoint')
        new['object_name'] = ck if ck else None
    # ensure checksum_algo exists
    if 'checksum_algo' not in new:
        new['checksum_algo'] = 'sha256'
    return new


def migrate_meta(meta: Dict[str, Any], target_version: str = CURRENT_VERSION) -> Dict[str, Any]:
    v = get_meta_version(meta)
    if v == target_version:
        return meta
    if v == '1.0' and target_version == '1.1':
        return migrate_1_0_to_1_1(meta)
    # for unsupported migrations, raise
    raise RuntimeError(f"Unsupported migration: {v} -> {target_version}")


def canonicalize_meta_file(path: str, target_version: str = CURRENT_VERSION) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r') as f:
        meta = json.load(f)
    if get_meta_version(meta) != target_version:
        migrated = migrate_meta(meta, target_version)
        # write back atomically
        tmp = path + '.tmp'
        with open(tmp, 'w') as tf:
            json.dump(migrated, tf)
            tf.flush()
            try:
                os.fsync(tf.fileno())
            except Exception:
                pass
        os.replace(tmp, path)
        return migrated
    return meta
