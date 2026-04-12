"""模型相关的数据模型

定义模型相关的数据类和模型。
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class Model:
    """模型数据类"""
    user_id: str
    name: str
    model_id: str
    description: Optional[str] = None
    version: str = "1.0.0"
    model_type: str = "llm"
    architecture: str = "transformer"
    framework: str = "pytorch"
    storage_path: str = ""
    config: Optional[Dict[str, Any]] = None
    training_session_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.config is None:
            self.config = {}


@dataclass
class ModelMetadata:
    """模型元数据"""
    model_id: str
    size_mb: float
    parameters_count: int
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    training_time: Optional[float] = None
    inference_time: Optional[float] = None
    memory_usage: Optional[float] = None
    
    
@dataclass
class ModelVersion:
    """模型版本"""
    model_id: str
    version: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    is_active: bool = False
    
    def __post_init__(self):
        """初始化后处理"""
        if self.created_at is None:
            self.created_at = datetime.now()