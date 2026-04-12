#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型管理 API 模块

提供完整的模型管理 REST API 接口：
- 模型 CRUD 操作
- 模型版本管理
- 模型部署
- 模型下载
- 模型管理（性能、验证、比较、导入导出）
"""

from backend.api.model.model_api import (
    model_bp,
    get_model_service,
    init_model_api,
    cleanup_model_api
)

from backend.api.model.model_download_api import (
    model_download_bp,
    init_download_api,
    cleanup_download_api
)

from backend.api.model.model_management_api import (
    model_management_bp,
    init_management_api,
    cleanup_management_api
)

__all__ = [
    # 模型基础管理
    'model_bp',
    'get_model_service',
    'init_model_api',
    'cleanup_model_api',
    
    # 模型下载
    'model_download_bp',
    'init_download_api',
    'cleanup_download_api',
    
    # 模型高级管理
    'model_management_bp',
    'init_management_api',
    'cleanup_management_api',
]
