"""数据预处理服务接口

定义数据预处理服务的抽象接口。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from backend.schemas.project_models import Dataset


class DataPreprocessingServiceInterface(ABC):
    """数据预处理服务接口
    
    定义数据预处理服务的标准接口。
    """
    
    @abstractmethod
    def preprocess(self, dataset_id: str, config: Optional[Dict[str, Any]] = None) -> Dataset:
        """执行数据预处理，返回更新后的数据集对象
        
        Args:
            dataset_id: 数据集ID
            config: 预处理配置
            
        Returns:
            更新后的数据集对象
        """
        pass
    
    @abstractmethod
    def preprocess_dataset(
        self, 
        dataset_id: str, 
        preprocessing_config: Dict[str, Any]
    ) -> Dataset:
        """预处理数据集
        
        Args:
            dataset_id: 数据集ID
            preprocessing_config: 预处理配置
            
        Returns:
            预处理后的数据集对象
        """
        pass
    
    @abstractmethod
    def perform_feature_engineering(
        self, 
        dataset_id: str, 
        features_config: Dict[str, Any]
    ) -> Dataset:
        """执行特征工程
        
        Args:
            dataset_id: 数据集ID
            features_config: 特征工程配置
            
        Returns:
            特征工程后的数据集对象
        """
        pass
    
    @abstractmethod
    def perform_data_augmentation(
        self, 
        dataset_id: str, 
        augmentation_config: Dict[str, Any]
    ) -> Dataset:
        """执行数据增强
        
        Args:
            dataset_id: 数据集ID
            augmentation_config: 数据增强配置
            
        Returns:
            数据增强后的数据集对象
        """
        pass
    
    @abstractmethod
    def split_dataset(
        self, 
        dataset_id: str, 
        split_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """分割数据集
        
        Args:
            dataset_id: 数据集ID
            split_config: 分割配置
            
        Returns:
            分割后的数据集信息
        """
        pass
    
    @abstractmethod
    def get_task(self, task_id: str) -> Dict[str, Any]:
        """获取预处理任务详情
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务详情
        """
        pass
    
    @abstractmethod
    def list_tasks(
        self,
        dataset_id: str,
        status_filter: Optional[List[str]] = None,
        task_type_filter: Optional[List[str]] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """获取数据集的任务列表
        
        Args:
            dataset_id: 数据集ID
            status_filter: 状态过滤
            task_type_filter: 任务类型过滤
            page: 页码
            page_size: 每页大小
            
        Returns:
            任务列表和分页信息
        """
        pass
    
    @abstractmethod
    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """取消预处理任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            更新后的任务信息
        """
        pass
    
    @abstractmethod
    def get_preprocessing_history(
        self,
        dataset_id: str,
        operation_type_filter: Optional[List[str]] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """获取预处理历史记录
        
        Args:
            dataset_id: 数据集ID
            operation_type_filter: 操作类型过滤
            page: 页码
            page_size: 每页大小
            
        Returns:
            历史记录列表和分页信息
        """
        pass
    
    @abstractmethod
    def create_pipeline(
        self,
        user_id: str,
        name: str,
        operations: List[Dict[str, Any]],
        description: Optional[str] = None,
        is_template: bool = False,
        is_public: bool = False,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建预处理流水线
        
        Args:
            user_id: 用户ID
            name: 流水线名称
            operations: 操作列表
            description: 描述
            is_template: 是否为模板
            is_public: 是否公开
            tenant_id: 租户ID
            
        Returns:
            创建的流水线信息
        """
        pass
    
    @abstractmethod
    def get_pipeline(self, pipeline_id: str) -> Dict[str, Any]:
        """获取流水线详情
        
        Args:
            pipeline_id: 流水线ID
            
        Returns:
            流水线详情
        """
        pass
    
    @abstractmethod
    def list_pipelines(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        is_template: Optional[bool] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """获取用户的流水线列表
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            is_template: 是否为模板
            page: 页码
            page_size: 每页大小
            
        Returns:
            流水线列表和分页信息
        """
        pass
    
    @abstractmethod
    def execute_pipeline(
        self,
        dataset_id: str,
        pipeline_id: str
    ) -> Dataset:
        """执行预处理流水线
        
        Args:
            dataset_id: 数据集ID
            pipeline_id: 流水线ID
            
        Returns:
            处理后的数据集
        """
        pass
