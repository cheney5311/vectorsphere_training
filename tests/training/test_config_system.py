"""配置系统测试用例

测试统一配置管理器和配置工厂的功能。
"""

import pytest
import tempfile
import shutil
import json
import yaml
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.config.unified_config_manager import UnifiedConfigManager
from training.config.config_factory import ConfigFactory
from training.unified_training_system import TrainingConfig
from training.core import ConfigurationError, ValidationError


class TestConfigFactory:
    """配置工厂测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.factory = ConfigFactory()
    
    def test_available_templates(self):
        """测试可用模板"""
        templates = self.factory.get_available_templates()
        
        # 检查基础模板
        assert 'quick_start' in templates
        assert 'moe_training' in templates
        assert 'multimodal_training' in templates
        assert 'distributed_training' in templates
        
        # 检查场景化训练模板
        assert 'scheduled_training' in templates
        assert 'advanced_model_training' in templates
        assert 'multimodal_scenario' in templates
    
    def test_quick_start_template(self):
        """测试快速开始模板"""
        template = self.factory.get_template('quick_start')
        
        assert 'global' in template
        assert 'model' in template
        assert 'training' in template
        assert 'data' in template
        
        # 检查默认值
        assert template['training']['num_epochs'] == 3
        assert template['training']['batch_size'] == 8
        assert template['model']['model_type'] == 'transformer'
    
    def test_scheduled_training_template(self):
        """测试三阶段训练模板"""
        template = self.factory.get_template('scheduled_training')
        
        assert 'scenario' in template
        assert template['scenario']['type'] == 'scheduled_training'
        assert 'stages' in template['scenario']
        
        stages = template['scenario']['stages']
        assert 'warmup' in stages
        assert 'main' in stages
        assert 'fine_tune' in stages
        
        # 检查阶段配置
        assert stages['warmup']['epochs'] == 2
        assert stages['warmup']['lr_multiplier'] == 0.1
        assert stages['main']['lr_multiplier'] == 1.0
    
    def test_multimodal_scenario_template(self):
        """测试多模态场景模板"""
        template = self.factory.get_template('multimodal_scenario')
        
        assert template['scenario']['type'] == 'multimodal'
        assert 'modalities' in template['scenario']
        assert 'fusion_strategy' in template['scenario']
        assert 'modality_weights' in template['scenario']
        
        modalities = template['scenario']['modalities']
        assert 'text' in modalities
        assert 'image' in modalities
        
        weights = template['scenario']['modality_weights']
        assert weights['text'] == 0.5
        assert weights['image'] == 0.5
    
    def test_advanced_model_template(self):
        """测试高级模型训练模板"""
        template = self.factory.get_template('advanced_model_training')
        
        assert template['scenario']['type'] == 'advanced_model'
        assert 'optimization' in template['scenario']
        assert 'regularization' in template['scenario']
        
        optimization = template['scenario']['optimization']
        assert optimization['use_mixed_precision'] == True
        assert optimization['gradient_accumulation_steps'] == 4
        
        regularization = template['scenario']['regularization']
        assert regularization['dropout_rate'] == 0.1
        assert regularization['weight_decay'] == 0.01
    
    def test_create_training_config(self):
        """测试创建训练配置"""
        # 使用模板创建配置
        config = self.factory.create_training_config(
            template_name='quick_start',
            extra_params={'training': {'num_epochs': 10}}
        )
        
        assert isinstance(config, dict)
        assert config['training']['num_epochs'] == 10  # 覆盖的值
        assert config['training']['batch_size'] == 8   # 模板默认值
    
    def test_invalid_template(self):
        """测试无效模板"""
        with pytest.raises(KeyError):
            self.factory.get_template('nonexistent_template')
    
    def test_template_customization(self):
        """测试模板自定义"""
        base_template = self.factory.get_template('quick_start')
        
        # 自定义参数
        custom_params = {
            'training': {
                'num_epochs': 20,
                'learning_rate': 1e-3
            },
            'model': {
                'model_name': 'custom_model'
            }
        }
        
        config = self.factory.create_training_config(
            template_name='quick_start',
            extra_params=custom_params
        )
        
        assert config['training']['num_epochs'] == 20
        assert config['training']['learning_rate'] == 1e-3
        assert config['model']['model_name'] == 'custom_model'
        # 保持其他默认值
        assert config['training']['batch_size'] == 8


class TestConfigManager:
    """配置管理器测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager()
        self.config_manager.config_paths = [self.temp_dir]
    
    def teardown_method(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_load_json_config(self):
        """测试加载JSON配置"""
        config_data = {
            'global': {
                'project_name': 'test_project',
                'output_dir': './outputs'
            },
            'training': {
                'num_epochs': 10,
                'batch_size': 16
            }
        }
        
        config_file = Path(self.temp_dir) / 'test_config.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        loaded_config = self.config_manager.load_config(str(config_file))
        
        assert loaded_config['global']['project_name'] == 'test_project'
        assert loaded_config['training']['num_epochs'] == 10
        assert loaded_config['training']['batch_size'] == 16
    
    def test_load_yaml_config(self):
        """测试加载YAML配置"""
        config_data = {
            'global': {
                'project_name': 'yaml_test_project',
                'output_dir': './yaml_outputs'
            },
            'model': {
                'model_name': 'yaml_model',
                'model_type': 'transformer'
            }
        }
        
        config_file = Path(self.temp_dir) / 'test_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        loaded_config = self.config_manager.load_config(str(config_file))
        
        assert loaded_config['global']['project_name'] == 'yaml_test_project'
        assert loaded_config['model']['model_name'] == 'yaml_model'
    
    def test_create_config_from_template(self):
        """测试从模板创建配置"""
        config = self.config_manager.create_config(
            template='quick_start',
            extra_params={
                'global': {'project_name': 'template_test'},
                'training': {'num_epochs': 15}
            }
        )
        
        assert isinstance(config, TrainingConfig)
        assert config.project_name == 'template_test'
        assert config.num_epochs == 15
    
    def test_validate_config(self):
        """测试配置验证"""
        valid_config = {
            'global': {
                'project_name': 'valid_project',
                'output_dir': './outputs'
            },
            'training': {
                'num_epochs': 10,
                'batch_size': 16,
                'learning_rate': 1e-4
            }
        }
        
        # 有效配置应该通过验证
        assert self.config_manager.validate_config(valid_config) == True
        
        # 无效配置应该抛出异常
        invalid_config = {
            'global': {
                'project_name': 'invalid_project'
                # 缺少output_dir
            },
            'training': {
                'num_epochs': -1,  # 无效值
                'batch_size': 0    # 无效值
            }
        }
        
        with pytest.raises(ValidationError):
            self.config_manager.validate_config(invalid_config)
    
    def test_scenario_config_validation(self):
        """测试场景配置验证"""
        # 测试三阶段训练场景配置
        scheduled_config = {
            'scenario': {
                'type': 'scheduled_training',
                'stages': {
                    'warmup': {'epochs': 2, 'lr_multiplier': 0.1},
                    'main': {'epochs': 5, 'lr_multiplier': 1.0},
                    'fine_tune': {'epochs': 3, 'lr_multiplier': 0.1}
                }
            }
        }
        
        assert self.config_manager._validate_scenario_config(scheduled_config) == True
        
        # 测试多模态场景配置
        multimodal_config = {
            'scenario': {
                'type': 'multimodal',
                'modalities': ['text', 'image'],
                'fusion_strategy': 'late_fusion',
                'modality_weights': {
                    'text': 0.6,
                    'image': 0.4
                }
            }
        }
        
        assert self.config_manager._validate_scenario_config(multimodal_config) == True
    
    def test_invalid_scenario_config(self):
        """测试无效场景配置"""
        # 缺少必要字段的配置
        invalid_config = {
            'scenario': {
                'type': 'scheduled_training'
                # 缺少stages配置
            }
        }
        
        with pytest.raises(ValidationError):
            self.config_manager._validate_scenario_config(invalid_config)
        
        # 权重总和不为1的多模态配置
        invalid_multimodal_config = {
            'scenario': {
                'type': 'multimodal',
                'modalities': ['text', 'image'],
                'modality_weights': {
                    'text': 0.7,
                    'image': 0.5  # 总和 > 1
                }
            }
        }
        
        with pytest.raises(ValidationError):
            self.config_manager._validate_scenario_config(invalid_multimodal_config)
    
    def test_create_scenario_config(self):
        """测试创建场景配置"""
        scenario_config = self.config_manager.create_scenario_config(
            scenario_type='scheduled_training',
            scenario_params={
                'stages': {
                    'warmup': {'epochs': 3, 'lr_multiplier': 0.05},
                    'main': {'epochs': 7, 'lr_multiplier': 1.0},
                    'fine_tune': {'epochs': 2, 'lr_multiplier': 0.05}
                }
            }
        )
        
        assert scenario_config['scenario']['type'] == 'scheduled_training'
        assert scenario_config['scenario']['stages']['warmup']['epochs'] == 3
        assert scenario_config['scenario']['stages']['warmup']['lr_multiplier'] == 0.05
    
    def test_get_scenario_templates(self):
        """测试获取场景模板"""
        templates = self.config_manager.get_scenario_templates()
        
        assert 'scheduled_training' in templates
        assert 'advanced_model' in templates
        assert 'multimodal' in templates
        
        # 检查模板结构
        scheduled_template = templates['scheduled_training']
        assert 'stages' in scheduled_template
        assert 'warmup' in scheduled_template['stages']
    
    def test_export_import_scenario_config(self):
        """测试导出和导入场景配置"""
        scenario_config = {
            'scenario': {
                'type': 'scheduled_training',
                'stages': {
                    'warmup': {'epochs': 2, 'lr_multiplier': 0.1},
                    'main': {'epochs': 5, 'lr_multiplier': 1.0},
                    'fine_tune': {'epochs': 3, 'lr_multiplier': 0.1}
                }
            }
        }
        
        # 导出配置
        export_path = Path(self.temp_dir) / 'exported_config.json'
        self.config_manager.export_scenario_config(scenario_config, str(export_path))
        
        # 验证文件存在
        assert export_path.exists()
        
        # 导入配置
        imported_config = self.config_manager.import_scenario_config(str(export_path))
        
        # 验证导入的配置
        assert imported_config['scenario']['type'] == 'scheduled_training'
        assert imported_config['scenario']['stages']['warmup']['epochs'] == 2
    
    def test_config_caching(self):
        """测试配置缓存"""
        config_data = {
            'global': {'project_name': 'cache_test'},
            'training': {'num_epochs': 5}
        }
        
        config_file = Path(self.temp_dir) / 'cache_test.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # 第一次加载
        config1 = self.config_manager.load_config(str(config_file))
        
        # 第二次加载（应该从缓存获取）
        config2 = self.config_manager.load_config(str(config_file))
        
        assert config1 == config2
        assert str(config_file) in self.config_manager.config_cache
    
    def test_config_file_not_found(self):
        """测试配置文件不存在"""
        with pytest.raises(FileNotFoundError):
            self.config_manager.load_config('/nonexistent/config.json')
    
    def test_invalid_config_format(self):
        """测试无效配置格式"""
        # 创建无效的JSON文件
        invalid_file = Path(self.temp_dir) / 'invalid.json'
        with open(invalid_file, 'w') as f:
            f.write('invalid json content {')
        
        with pytest.raises(json.JSONDecodeError):
            self.config_manager.load_config(str(invalid_file))


class TestConfigIntegration:
    """配置系统集成测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager()
        self.config_factory = ConfigFactory()
    
    def teardown_method(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_config_workflow(self):
        """测试端到端配置工作流"""
        # 1. 从模板创建配置
        template_config = self.config_factory.create_training_config(
            template_name='scheduled_training',
            extra_params={
                'global': {'project_name': 'e2e_test'},
                'training': {'num_epochs': 12}
            }
        )
        
        # 2. 保存配置到文件
        config_file = Path(self.temp_dir) / 'e2e_config.json'
        with open(config_file, 'w') as f:
            json.dump(template_config, f, indent=2)
        
        # 3. 加载配置
        loaded_config = self.config_manager.load_config(str(config_file))
        
        # 4. 验证配置
        assert self.config_manager.validate_config(loaded_config) == True
        
        # 5. 创建训练配置对象
        training_config = self.config_manager.create_config(
            config_dict=loaded_config
        )
        
        assert isinstance(training_config, TrainingConfig)
        assert training_config.project_name == 'e2e_test'
        assert training_config.num_epochs == 12
    
    def test_scenario_config_workflow(self):
        """测试场景配置工作流"""
        # 1. 创建场景配置
        scenario_config = self.config_manager.create_scenario_config(
            scenario_type='multimodal',
            scenario_params={
                'modalities': ['text', 'image', 'audio'],
                'fusion_strategy': 'early_fusion',
                'modality_weights': {
                    'text': 0.4,
                    'image': 0.4,
                    'audio': 0.2
                }
            }
        )
        
        # 2. 验证场景配置
        assert self.config_manager._validate_scenario_config(scenario_config) == True
        
        # 3. 导出场景配置
        export_path = Path(self.temp_dir) / 'scenario_config.yaml'
        self.config_manager.export_scenario_config(scenario_config, str(export_path))
        
        # 4. 导入场景配置
        imported_config = self.config_manager.import_scenario_config(str(export_path))
        
        # 5. 验证导入的配置
        assert imported_config['scenario']['type'] == 'multimodal'
        assert len(imported_config['scenario']['modalities']) == 3
        assert imported_config['scenario']['fusion_strategy'] == 'early_fusion'
    
    def test_config_template_customization(self):
        """测试配置模板自定义"""
        # 获取基础模板
        base_template = self.config_factory.get_template('advanced_model_training')
        
        # 自定义模板
        custom_params = {
            'scenario': {
                'optimization': {
                    'gradient_accumulation_steps': 8,
                    'max_grad_norm': 0.5
                },
                'regularization': {
                    'dropout_rate': 0.2
                }
            }
        }
        
        # 创建自定义配置
        custom_config = self.config_factory.create_training_config(
            template_name='advanced_model_training',
            extra_params=custom_params
        )
        
        # 验证自定义值
        assert custom_config['scenario']['optimization']['gradient_accumulation_steps'] == 8
        assert custom_config['scenario']['optimization']['max_grad_norm'] == 0.5
        assert custom_config['scenario']['regularization']['dropout_rate'] == 0.2
        
        # 验证保留的默认值
        assert custom_config['scenario']['optimization']['use_mixed_precision'] == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])