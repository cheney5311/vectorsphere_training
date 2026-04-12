"""模型服务接口

定义模型服务的接口规范。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# 修复导入错误，使用正确的模块路径
from backend.schemas.model import Model


class ModelServiceInterface(ABC):
    """模型服务接口"""
    
    @abstractmethod
    def create_model(
        self, 
        user_id: str, 
        name: str, 
        description: Optional[str] = None,
        version: str = "1.0.0",
        model_type: str = "llm",
        architecture: str = "transformer",
        framework: str = "pytorch",
        storage_path: str = "",
        config: Optional[Dict[str, Any]] = None
    ) -> Model:
        """创建模型
        
        Args:
            user_id: 用户ID
            name: 模型名称
            description: 模型描述
            version: 版本号
            model_type: 模型类型
            architecture: 架构类型
            framework: 框架类型
            storage_path: 存储路径
            config: 配置信息
            
        Returns:
            Model: 创建的模型对象
        """
        pass
        
    @abstractmethod
    def get_model(self, model_id: str) -> Optional[Model]:
        """获取模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            Model: 模型对象，如果不存在则返回None
        """
        pass
        
    @abstractmethod
    def list_models(
        self, 
        user_id: str, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[Model]:
        """获取用户模型列表
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            List[Model]: 模型列表
        """
        pass
        
    @abstractmethod
    def update_model(
        self, 
        model_id: str, 
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Model:
        """更新模型
        
        Args:
            model_id: 模型ID
            name: 模型名称
            description: 模型描述
            config: 配置信息
            
        Returns:
            Model: 更新后的模型对象
        """
        pass
        
    @abstractmethod
    def delete_model(self, model_id: str) -> bool:
        """删除模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            bool: 删除成功返回True，否则返回False
        """
        pass
        
    @abstractmethod
    def process_model(self, model_id: str) -> Model:
        """处理模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            Model: 处理后的模型对象
        """
        pass
        
    @abstractmethod
    def validate_model(self, model_id: str, metrics: Dict[str, float]) -> Model:
        """验证模型
        
        Args:
            model_id: 模型ID
            metrics: 性能指标
            
        Returns:
            Model: 验证后的模型对象
        """
        pass
        
    @abstractmethod
    def mark_model_ready(self, model_id: str) -> Model:
        """标记模型为就绪状态
        
        Args:
            model_id: 模型ID
            
        Returns:
            Model: 更新后的模型对象
        """
        pass
        
    @abstractmethod
    def deploy_model(self, model_id: str) -> Model:
        """部署模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            Model: 部署后的模型对象
        """
        pass