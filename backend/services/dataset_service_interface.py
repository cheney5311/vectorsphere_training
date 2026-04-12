"""数据集服务接口

定义数据集服务的接口规范。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# 修复导入错误，使用正确的模块路径
from backend.schemas.project_models import Dataset


class DatasetServiceInterface(ABC):
    """数据集服务接口"""
    
    @abstractmethod
    def create_dataset(
        self, 
        user_id: str, 
        name: str, 
        description: Optional[str] = None,
        dataset_type: str = "text",
        format: str = "json",
        storage_path: str = "",
        config: Optional[Dict[str, Any]] = None
    ) -> Dataset:
        """创建数据集
        
        Args:
            user_id: 用户ID
            name: 数据集名称
            description: 数据集描述
            dataset_type: 数据集类型
            format: 数据格式
            storage_path: 存储路径
            config: 配置信息
            
        Returns:
            Dataset: 创建的数据集对象
        """
        pass
        
    @abstractmethod
    def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """获取数据集
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            Dataset: 数据集对象，如果不存在则返回None
        """
        pass
        
    @abstractmethod
    def list_datasets(
        self, 
        user_id: str, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[Dataset]:
        """获取用户数据集列表
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            List[Dataset]: 数据集列表
        """
        pass
        
    @abstractmethod
    def update_dataset(
        self, 
        dataset_id: str, 
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dataset:
        """更新数据集
        
        Args:
            dataset_id: 数据集ID
            name: 数据集名称
            description: 数据集描述
            config: 配置信息
            
        Returns:
            Dataset: 更新后的数据集对象
        """
        pass
        
    @abstractmethod
    def delete_dataset(self, dataset_id: str) -> bool:
        """删除数据集
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            bool: 删除成功返回True，否则返回False
        """
        pass
        
    @abstractmethod
    def process_dataset(self, dataset_id: str) -> Dataset:
        """处理数据集
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            Dataset: 处理后的数据集对象
        """
        pass
        
    @abstractmethod
    def validate_dataset(self, dataset_id: str, validation_result: Dict[str, Any]) -> Dataset:
        """验证数据集
        
        Args:
            dataset_id: 数据集ID
            validation_result: 验证结果
            
        Returns:
            Dataset: 验证后的数据集对象
        """
        pass
        
    @abstractmethod
    def mark_dataset_ready(self, dataset_id: str) -> Dataset:
        """标记数据集为就绪状态
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            Dataset: 更新后的数据集对象
        """
        pass


class DataQualityServiceInterface(ABC):
    """数据质量服务接口"""
    
    @abstractmethod
    def assess_data_quality(self, dataset_id: str) -> Dict[str, Any]:
        """评估数据质量
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            Dict[str, Any]: 数据质量评估结果
        """
        pass
        
    @abstractmethod
    def detect_data_issues(self, dataset_id: str) -> List[Dict[str, Any]]:
        """检测数据问题
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            List[Dict[str, Any]]: 检测到的数据问题列表
        """
        pass
        
    @abstractmethod
    def clean_data(self, dataset_id: str, cleaning_config: Optional[Dict[str, Any]] = None) -> Dataset:
        """清理数据
        
        Args:
            dataset_id: 数据集ID
            cleaning_config: 清理配置
            
        Returns:
            Dataset: 清理后的数据集对象
        """
        pass
        
    @abstractmethod
    def generate_quality_report(self, dataset_id: str) -> Dict[str, Any]:
        """生成数据质量报告
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            Dict[str, Any]: 数据质量报告
        """
        pass


class DataPreprocessingServiceInterface(ABC):
    """数据预处理服务接口"""
    
    @abstractmethod
    def preprocess_dataset(self, dataset_id: str, preprocessing_config: Dict[str, Any]) -> Dataset:
        """预处理数据集
        
        Args:
            dataset_id: 数据集ID
            preprocessing_config: 预处理配置
            
        Returns:
            Dataset: 预处理后的数据集对象
        """
        pass
        
    @abstractmethod
    def perform_feature_engineering(self, dataset_id: str, features_config: Dict[str, Any]) -> Dataset:
        """执行特征工程
        
        Args:
            dataset_id: 数据集ID
            features_config: 特征工程配置
            
        Returns:
            Dataset: 特征工程后的数据集对象
        """
        pass
        
    @abstractmethod
    def perform_data_augmentation(self, dataset_id: str, augmentation_config: Dict[str, Any]) -> Dataset:
        """执行数据增强
        
        Args:
            dataset_id: 数据集ID
            augmentation_config: 数据增强配置
            
        Returns:
            Dataset: 数据增强后的数据集对象
        """
        pass
        
    @abstractmethod
    def split_dataset(self, dataset_id: str, split_config: Dict[str, Any]) -> Dict[str, Any]:
        """分割数据集
        
        Args:
            dataset_id: 数据集ID
            split_config: 分割配置
            
        Returns:
            Dict[str, Any]: 分割后的数据集信息
        """
        pass


class DataDiscoveryServiceInterface(ABC):
    """数据发现与接入服务接口"""
    
    @abstractmethod
    def scan_data_sources(self, user_id: str, scan_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """扫描数据源
        
        Args:
            user_id: 用户ID
            scan_config: 扫描配置
            
        Returns:
            List[Dict[str, Any]]: 发现的数据源列表
        """
        pass
        
    @abstractmethod
    def discover_datasets(self, user_id: str, discovery_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """发现数据集
        
        Args:
            user_id: 用户ID
            discovery_config: 发现配置
            
        Returns:
            List[Dict[str, Any]]: 发现的数据集列表
        """
        pass
        
    @abstractmethod
    def auto_ingest_dataset(self, user_id: str, source_info: Dict[str, Any]) -> Dataset:
        """自动接入数据集
        
        Args:
            user_id: 用户ID
            source_info: 数据源信息
            
        Returns:
            Dataset: 接入的数据集对象
        """
        pass
        
    @abstractmethod
    def infer_schema(self, dataset_id: str) -> Dict[str, Any]:
        """推断数据模式
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            Dict[str, Any]: 推断的数据模式
        """
        pass
        
    @abstractmethod
    def auto_transform(self, dataset_id: str, transform_config: Dict[str, Any]) -> Dataset:
        """自动转换数据
        
        Args:
            dataset_id: 数据集ID
            transform_config: 转换配置
            
        Returns:
            Dataset: 转换后的数据集对象
        """
        pass
        
    @abstractmethod
    def setup_incremental_sync(self, dataset_id: str, sync_config: Dict[str, Any]) -> Dataset:
        """设置增量同步
        
        Args:
            dataset_id: 数据集ID
            sync_config: 同步配置
            
        Returns:
            Dataset: 配置后的数据集对象
        """
        pass