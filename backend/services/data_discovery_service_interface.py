"""数据发现服务接口

定义数据发现与接入服务的抽象接口。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from backend.schemas.project_models import Dataset


class DataDiscoveryServiceInterface(ABC):
    """数据发现服务接口
    
    定义数据发现与接入服务的标准接口。
    """
    
    @abstractmethod
    def discover(self, dataset_id: str) -> Dict[str, Any]:
        """执行数据发现并返回发现信息
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            发现结果字典
        """
        pass

    @abstractmethod
    def list_discoveries(
        self, 
        user_id: str, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """列出用户的数据发现记录
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            发现记录列表
        """
        pass
    
    @abstractmethod
    def scan_data_sources(
        self, 
        user_id: str, 
        scan_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """扫描数据源
        
        Args:
            user_id: 用户ID
            scan_config: 扫描配置
            
        Returns:
            发现的数据源列表
        """
        pass
    
    @abstractmethod
    def discover_datasets(
        self, 
        user_id: str, 
        discovery_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """发现数据集
        
        Args:
            user_id: 用户ID
            discovery_config: 发现配置
            
        Returns:
            发现的数据集列表
        """
        pass
    
    @abstractmethod
    def auto_ingest_dataset(
        self, 
        user_id: str, 
        source_info: Dict[str, Any]
    ) -> Dataset:
        """自动接入数据集
        
        Args:
            user_id: 用户ID
            source_info: 数据源信息
            
        Returns:
            接入的数据集对象
        """
        pass
    
    @abstractmethod
    def infer_schema(self, dataset_id: str) -> Dict[str, Any]:
        """推断数据模式
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            推断的数据模式
        """
        pass
    
    @abstractmethod
    def auto_transform(
        self, 
        dataset_id: str, 
        transform_config: Dict[str, Any]
    ) -> Dataset:
        """自动转换数据
        
        Args:
            dataset_id: 数据集ID
            transform_config: 转换配置
            
        Returns:
            转换后的数据集对象
        """
        pass
    
    @abstractmethod
    def setup_incremental_sync(
        self, 
        dataset_id: str, 
        sync_config: Dict[str, Any]
    ) -> Dataset:
        """设置增量同步
        
        Args:
            dataset_id: 数据集ID
            sync_config: 同步配置
            
        Returns:
            配置后的数据集对象
        """
        pass