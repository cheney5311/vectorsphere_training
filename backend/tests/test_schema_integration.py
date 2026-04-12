"""JSON Schema集成测试

测试新创建的JSON Schema文件的有效性和集成功能。
"""
import pytest
import json
import os
from unittest.mock import Mock, patch

from backend.core.schema_manager import SchemaManager, get_schema_manager
from backend.core.validation import validate_json_schema
from flask import Flask

# 定义测试用的Schema数据
TRAINING_CREATE_SCHEMA = {
    "type": "object",
    "properties": {
        "model_name": {"type": "string"},
        "dataset_path": {"type": "string"},
        "training_config": {
            "type": "object",
            "properties": {
                "epochs": {"type": "integer", "minimum": 1},
                "batch_size": {"type": "integer", "maximum": 1024},
                "learning_rate": {"type": "number", "maximum": 1.0},
                "optimizer": {"type": "string", "enum": ["adam", "sgd", "rmsprop"]},
                "scheduler": {"type": "string"}
            },
            "required": ["epochs", "batch_size", "learning_rate", "optimizer"]
        },
        "gpu_config": {"type": "object"},
        "checkpoint_config": {"type": "object"},
        "metadata": {"type": "object"}
    },
    "required": ["model_name", "dataset_path", "training_config"]
}

MODEL_DEPLOYMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "model_id": {"type": "string"},
        "deployment_name": {"type": "string"},
        "deployment_config": {
            "type": "object",
            "properties": {
                "instance_type": {"type": "string"},
                "min_instances": {"type": "integer"},
                "max_instances": {"type": "integer"},
                "auto_scaling": {
                    "type": "object",
                    "properties": {
                        "target_cpu_utilization": {"type": "integer", "maximum": 100}
                    }
                },
                "health_check": {"type": "object"}
            },
            "required": ["instance_type", "min_instances", "max_instances"]
        },
        "environment_variables": {"type": "object"},
        "resource_limits": {"type": "object"},
        "metadata": {"type": "object"}
    },
    "required": ["model_id", "deployment_name", "deployment_config"]
}

GPU_ALLOCATION_SCHEMA = {
    "type": "object",
    "properties": {
        "resource_type": {"type": "string", "enum": ["gpu"]},
        "quantity": {"type": "integer", "minimum": 1},
        "gpu_requirements": {"type": "object"},
        "duration": {
            "type": "object", 
            "properties": {
                "type": {"type": "string"},
                "value": {"type": "integer", "minimum": 1}
            }
        },
        "scheduling": {"type": "object"},
        "resource_limits": {"type": "object"},
        "metadata": {"type": "object"}
    },
    "required": ["resource_type", "quantity"]
}

def register_test_schemas(manager):
    """注册测试用的Schema"""
    manager.add_schema("training_create", TRAINING_CREATE_SCHEMA)
    manager.add_schema("model_deployment", MODEL_DEPLOYMENT_SCHEMA)
    manager.add_schema("gpu_allocation", GPU_ALLOCATION_SCHEMA)

class TestSchemaFiles:
    """Schema文件测试"""
    
    def setup_method(self):
        """测试设置"""
        self.schema_manager = SchemaManager(schema_dir=None)
        register_test_schemas(self.schema_manager)
    
    def test_training_create_schema(self):
        """测试训练创建schema"""
        valid_data = {
            "model_name": "bert-base-uncased",
            "dataset_path": "/data/training/dataset.json",
            "training_config": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "scheduler": "linear"
            },
            "gpu_config": {
                "gpu_count": 2,
                "gpu_type": "V100",
                "memory_per_gpu": "16GB"
            },
            "checkpoint_config": {
                "save_every": 1000,
                "max_checkpoints": 5,
                "checkpoint_dir": "/checkpoints/bert-training"
            },
            "metadata": {
                "description": "BERT model training for NLP tasks",
                "tags": ["nlp", "bert", "transformer"]
            }
        }
        
        is_valid, error = self.schema_manager.validate_data(valid_data, "training_create")
        assert is_valid is True, f"Validation errors: {error}"
    
    def test_training_create_schema_invalid_data(self):
        """测试训练创建schema无效数据"""
        invalid_data = {
            "model_name": "bert-base-uncased"
        }
        
        # 注意参数顺序：data, schema_name
        is_valid, error = self.schema_manager.validate_data(invalid_data, "training_create")
        assert is_valid is False
        assert error is not None
    
    def test_training_create_schema_invalid_ranges(self):
        """测试训练创建schema范围验证"""
        invalid_data = {
            "model_name": "bert-base-uncased",
            "dataset_path": "/data/training/dataset.json",
            "training_config": {
                "epochs": 0,
                "batch_size": 2048,
                "learning_rate": 2.0,
                "optimizer": "invalid_optimizer"
            }
        }
        
        is_valid, error = self.schema_manager.validate_data(invalid_data, "training_create")
        assert is_valid is False
        assert error is not None
    
    def test_model_deployment_schema(self):
        """测试模型部署schema"""
        valid_data = {
            "model_id": "model-12345",
            "deployment_name": "bert-production",
            "deployment_config": {
                "instance_type": "ml.m5.large",
                "min_instances": 1,
                "max_instances": 10,
                "auto_scaling": {
                    "target_cpu_utilization": 70
                },
                "health_check": {
                    "enabled": True
                }
            },
            "environment_variables": {
                "MODEL_VERSION": "1.0.0"
            },
            "resource_limits": {
                "cpu_limit": "2000m"
            },
            "metadata": {
                "description": "Production deployment"
            }
        }
        
        is_valid, error = self.schema_manager.validate_data(valid_data, "model_deployment")
        assert is_valid is True, f"Validation errors: {error}"
    
    def test_model_deployment_schema_invalid_data(self):
        """测试模型部署schema无效数据"""
        invalid_data = {
            "model_id": "model-12345",
            "deployment_name": "bert-production",
            "deployment_config": {
                # Missing instance_type to force validation error
                "min_instances": 10,
                "max_instances": 5
            }
        }
        
        is_valid, error = self.schema_manager.validate_data(invalid_data, "model_deployment")
        assert is_valid is False
        assert error is not None
    
    def test_gpu_allocation_schema(self):
        """测试GPU分配schema"""
        valid_data = {
            "resource_type": "gpu",
            "quantity": 4,
            "gpu_requirements": {
                "gpu_type": "V100"
            },
            "duration": {
                "type": "hours",
                "value": 24
            },
            "scheduling": {
                "priority": "high"
            },
            "resource_limits": {
                "memory_gb": 64
            },
            "metadata": {
                "project_id": "project-123"
            }
        }
        
        is_valid, error = self.schema_manager.validate_data(valid_data, "gpu_allocation")
        assert is_valid is True, f"Validation errors: {error}"
    
    def test_gpu_allocation_schema_invalid_data(self):
        """测试GPU分配schema无效数据"""
        invalid_data = {
            "resource_type": "invalid_type",
            "quantity": 0,
            "gpu_requirements": {},
            "duration": {
                "type": "invalid_type",
                "value": -1
            }
        }
        
        is_valid, error = self.schema_manager.validate_data(invalid_data, "gpu_allocation")
        assert is_valid is False
        assert error is not None


class TestSchemaIntegrationWithFlask:
    """Schema与Flask集成测试"""
    
    def setup_method(self):
        """测试设置"""
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # 初始化SchemaManager并注册测试schema
        self.schema_manager = SchemaManager(schema_dir=None)
        register_test_schemas(self.schema_manager)
        
        # Patch get_schema_manager to return our initialized manager
        # MUST patch in schema_manager module because validate_json_data uses it there
        self.manager_patcher = patch('backend.core.schema_manager.get_schema_manager', return_value=self.schema_manager)
        self.manager_patcher.start()
        
        # 强制启用严格模式
        self.config_patcher = patch('backend.core.validation.validation_config.strict_mode', True)
        self.config_patcher.start()

    def teardown_method(self):
        self.manager_patcher.stop()
        self.config_patcher.stop()
    
    def test_training_endpoint_with_schema(self):
        """测试带schema验证的训练端点"""
        @self.app.route('/api/training', methods=['POST'])
        @validate_json_schema("training_create")
        def create_training():
            return {"status": "training_created", "id": "training-123"}
        
        # 有效请求
        valid_data = {
            "model_name": "bert-base-uncased",
            "dataset_path": "/data/training/dataset.json",
            "training_config": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": "adam"
            }
        }
        
        response = self.client.post('/api/training',
                                  json=valid_data,
                                  content_type='application/json')
        assert response.status_code == 200
        
        # 无效请求
        invalid_data = {
            "model_name": "bert-base-uncased"
        }
        
        response = self.client.post('/api/training',
                                  json=invalid_data,
                                  content_type='application/json')
        assert response.status_code == 400
    
    def test_deployment_endpoint_with_schema(self):
        """测试带schema验证的部署端点"""
        @self.app.route('/api/deployments', methods=['POST'])
        @validate_json_schema("model_deployment")
        def create_deployment():
            return {"status": "deployment_created", "id": "deployment-123"}
        
        # 有效请求
        valid_data = {
            "model_id": "model-12345",
            "deployment_name": "bert-production",
            "deployment_config": {
                "instance_type": "ml.m5.large",
                "min_instances": 1,
                "max_instances": 10
            }
        }
        
        response = self.client.post('/api/deployments',
                                  json=valid_data,
                                  content_type='application/json')
        assert response.status_code == 200
        
        # 无效请求
        invalid_data = {
            "model_id": "model-12345",
            "deployment_name": "bert-production",
            "deployment_config": {
                # Missing instance_type to force validation error
                "min_instances": 10,
                "max_instances": 5
            }
        }
        
        response = self.client.post('/api/deployments',
                                  json=invalid_data,
                                  content_type='application/json')
        assert response.status_code == 400
    
    def test_gpu_allocation_endpoint_with_schema(self):
        """测试带schema验证的GPU分配端点"""
        @self.app.route('/api/gpu/allocate', methods=['POST'])
        @validate_json_schema("gpu_allocation")
        def allocate_gpu():
            return {"status": "allocation_created", "id": "allocation-123"}
        
        # 有效请求
        valid_data = {
            "resource_type": "gpu",
            "quantity": 2,
            "gpu_requirements": {
                "gpu_type": "V100",
                "memory_per_gpu": "16GB"
            },
            "duration": {
                "type": "hours",
                "value": 8
            }
        }
        
        response = self.client.post('/api/gpu/allocate',
                                  json=valid_data,
                                  content_type='application/json')
        assert response.status_code == 200
        
        # 无效请求
        invalid_data = {
            "resource_type": "gpu",
            "quantity": 0,
            "gpu_requirements": {},
            "duration": {
                "type": "hours",
                "value": 8
            }
        }
        
        response = self.client.post('/api/gpu/allocate',
                                  json=invalid_data,
                                  content_type='application/json')
        assert response.status_code == 400


class TestSchemaVersioning:
    """Schema版本控制测试"""
    
    def setup_method(self):
        """测试设置"""
        self.schema_manager = SchemaManager(schema_dir=None)
    
    def test_schema_backward_compatibility(self):
        """测试schema向后兼容性"""
        old_schema = {
            "type": "object",
            "properties": {
                "model_name": {"type": "string"},
                "dataset_path": {"type": "string"}
            },
            "required": ["model_name", "dataset_path"]
        }
        
        new_schema = {
            "type": "object",
            "properties": {
                "model_name": {"type": "string"},
                "dataset_path": {"type": "string"},
                "training_config": {
                    "type": "object",
                    "properties": {
                        "epochs": {"type": "integer", "minimum": 1}
                    }
                }
            },
            "required": ["model_name", "dataset_path"]
        }
        
        self.schema_manager.add_schema("training", old_schema, version="1.0")
        self.schema_manager.add_schema("training", new_schema, version="2.0")
        
        old_data = {
            "model_name": "bert-base",
            "dataset_path": "/data/dataset.json"
        }
        
        is_valid_v1, _ = self.schema_manager.validate_data(old_data, "training", version="1.0")
        is_valid_v2, _ = self.schema_manager.validate_data(old_data, "training", version="2.0")
        
        assert is_valid_v1 is True
        assert is_valid_v2 is True
    
    def test_schema_compatibility_check(self):
        """测试schema兼容性检查"""
        old_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
        
        compatible_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["name"]
        }
        
        incompatible_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "required_field": {"type": "string"}
            },
            "required": ["name", "required_field"]
        }
        
        self.schema_manager.add_schema("test", old_schema, version="1.0")
        self.schema_manager.add_schema("test", compatible_schema, version="1.1")
        self.schema_manager.add_schema("test", incompatible_schema, version="2.0")
        
        result_compatible = self.schema_manager.check_compatibility("test", "1.0", "1.1")
        assert result_compatible['compatible'] is True
        
        result_incompatible = self.schema_manager.check_compatibility("test", "1.0", "2.0")
        assert result_incompatible['compatible'] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
