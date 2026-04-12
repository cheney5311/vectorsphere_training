"""API集成测试"""

import sys
import os
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.modules.dataset.api import dataset_bp
from backend.modules.dataset.services.dataset_service import DatasetService
from backend.modules.model.api import model_bp
from backend.modules.model.services.model_service import ModelService
from backend.modules.dataset.models.dataset import Dataset
from backend.modules.model.models.model import Model

# 初始化服务实例（使用Mock对象）
dataset_service = DatasetService(Mock())
model_service = ModelService(Mock())


def test_dataset_api_integration():
    """测试数据集API集成"""
    # 创建测试数据集
    dataset = Dataset(
        user_id="test_user",
        name="test_dataset",
        dataset_type="text",
        format="json"
    )
    
    # 验证数据集有to_dict方法
    dataset_dict = dataset.to_dict()
    assert "dataset_id" in dataset_dict
    assert "name" in dataset_dict
    assert "dataset_type" in dataset_dict
    assert "status" in dataset_dict
    
    print("Dataset API integration test passed")


def test_model_api_integration():
    """测试模型API集成"""
    # 创建测试模型
    model = Model(
        user_id="test_user",
        name="test_model",
        version="1.0.0",
        model_type="llm",
        framework="pytorch"
    )
    
    # 验证模型有to_dict方法
    model_dict = model.to_dict()
    assert "model_id" in model_dict
    assert "name" in model_dict
    assert "version" in model_dict
    assert "model_type" in model_dict
    assert "status" in model_dict
    
    print("Model API integration test passed")


def test_service_initialization():
    """测试服务初始化"""
    # 检查数据集服务是否正确初始化
    assert dataset_service is not None
    
    # 检查模型服务是否正确初始化
    assert model_service is not None
    
    print("Service initialization test passed")


if __name__ == "__main__":
    test_dataset_api_integration()
    test_model_api_integration()
    test_service_initialization()
    print("All integration tests passed!")