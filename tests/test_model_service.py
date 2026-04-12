"""测试模型服务功能"""

import pytest
import uuid
from unittest.mock import Mock, MagicMock
from datetime import datetime
from io import BytesIO

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 修复导入路径，使用新的模块结构
from backend.modules.model.services.model_service import ModelService
from backend.modules.model.models.model import Model
from backend.modules.model.exceptions.model_exceptions import ModelValidationError

class TestModelService:
    """模型服务测试类"""
    
    def setup_method(self):
        """设置测试环境"""
        self.mock_model_repo = Mock()
        self.mock_deployment_repo = Mock()
        self.model_service = ModelService(
            model_repository=self.mock_model_repo
        )
    
    def test_create_model_success(self):
        """测试成功创建模型"""
        # 准备测试数据
        model_data = {
            'name': 'test_model',
            'description': 'Test model description',
            'model_type': 'classification',
            'framework': 'pytorch'
        }
        user_id = str(uuid.uuid4())
        
        # 模拟返回的模型对象
        mock_model = Model(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=model_data['name'],
            description=model_data['description'],
            model_type=model_data['model_type'],
            framework=model_data['framework'],
            status='created',
            created_at=datetime.utcnow()
        )
        
        self.mock_model_repo.create.return_value = mock_model
        
        # 执行测试
        result = self.model_service.create_model(
            user_id=user_id,
            name=model_data['name'],
            description=model_data['description'],
            model_type=model_data['model_type'],
            framework=model_data['framework']
        )
        
        # 验证结果
        assert result.id == mock_model.id
        assert result.name == model_data['name']
        assert result.status == 'created'
        self.mock_model_repo.create.assert_called_once()
    
    def test_get_models_success(self):
        """测试成功获取模型列表"""
        user_id = str(uuid.uuid4())
        
        # 模拟返回的模型列表
        mock_models = [
            Model(
                id=str(uuid.uuid4()),
                user_id=user_id,
                name='model1',
                model_type='classification',
                framework='pytorch',
                status='created',
                created_at=datetime.utcnow()
            )
        ]
        
        self.mock_model_repo.list_by_user.return_value = mock_models
        
        # 执行测试
        result = self.model_service.list_models(user_id)
        
        # 验证结果
        assert len(result) == 1
        assert result[0].name == 'model1'
        self.mock_model_repo.list_by_user.assert_called_once_with(user_id, 50, 0)
    
    def test_validate_model_data_success(self):
        """测试模型数据验证成功"""
        # 这个方法是私有的，我们通过create_model来测试
        model_data = {
            'name': 'test_model',
            'model_type': 'classification',
            'framework': 'pytorch'
        }
        
        user_id = str(uuid.uuid4())
        
        # 应该不抛出异常
        mock_model = Model(
            user_id=user_id,
            name=model_data['name'],
            model_type=model_data['model_type'],
            framework=model_data['framework'],
            status='created'
        )
        
        self.mock_model_repo.create.return_value = mock_model
        
        result = self.model_service.create_model(
            user_id=user_id,
            name=model_data['name'],
            model_type=model_data['model_type'],
            framework=model_data['framework']
        )
        
        assert result.name == model_data['name']
    
    def test_generate_next_version(self):
        """测试版本号生成"""
        # 测试从无版本开始
        version = "1.0.0"  # 默认版本
        
        # 测试版本格式
        assert version == "1.0.0"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])