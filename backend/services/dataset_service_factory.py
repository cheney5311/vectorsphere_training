"""数据集服务工厂

提供数据集相关服务的工厂类，用于统一管理各种数据集服务的创建和访问。
"""

import sys
import os
from typing import Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.repositories.dataset_repository import DatasetRepository
from backend.services.dataset_service import DatasetService
from backend.services.data_quality_service import DataQualityService
from backend.services.data_preprocessing_service import DataPreprocessingService
from backend.services.data_discovery_service import DataDiscoveryService


class DatasetServiceFactory:
    """数据集服务工厂类"""

    _instance: Optional['DatasetServiceFactory'] = None
    _dataset_repository: Optional[DatasetRepository] = None

    def __new__(cls) -> 'DatasetServiceFactory':
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化数据集服务工厂"""
        if not hasattr(self, '_initialized'):
            self._dataset_repository = DatasetRepository()
            self._initialized = True

    def get_dataset_service(self) -> DatasetService:
        """获取数据集服务实例
        
        Returns:
            DatasetService: 数据集服务实例
        """
        if not self._dataset_repository:
            self._dataset_repository = DatasetRepository()
        return DatasetService(self._dataset_repository)

    def get_data_quality_service(self) -> DataQualityService:
        """获取数据质量服务实例
        
        Returns:
            DataQualityService: 数据质量服务实例
        """
        if not self._dataset_repository:
            self._dataset_repository = DatasetRepository()
        return DataQualityService(self._dataset_repository)

    def get_data_preprocessing_service(self) -> DataPreprocessingService:
        """获取数据预处理服务实例
        
        Returns:
            DataPreprocessingService: 数据预处理服务实例
        """
        if not self._dataset_repository:
            self._dataset_repository = DatasetRepository()
        return DataPreprocessingService(self._dataset_repository)

    def get_data_discovery_service(self) -> DataDiscoveryService:
        """获取数据发现服务实例
        
        Returns:
            DataDiscoveryService: 数据发现服务实例
        """
        if not self._dataset_repository:
            self._dataset_repository = DatasetRepository()
        return DataDiscoveryService(self._dataset_repository)

    def get_dataset_repository(self) -> DatasetRepository:
        """获取数据集仓库实例
        
        Returns:
            DatasetRepository: 数据集仓库实例
        """
        if not self._dataset_repository:
            self._dataset_repository = DatasetRepository()
        return self._dataset_repository
