"""训练场景测试用例

测试各种训练场景的功能和集成。
"""

import pytest
import torch
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import json
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.scenarios import (
    ScheduledTrainingScenario, AdvancedModelScenario, 
    MultiModalScenario, ScenarioManager
)
from training.config.unified_config_manager import UnifiedConfigManager
from training.core import TrainingError, ConfigurationError


class TestScheduledTrainingScenario:
    """三阶段训练场景测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'global': {
                'project_name': 'test_scheduled_training',
                'output_dir': self.temp_dir,
                'seed': 42
            },
            'model': {
                'model_name': 'test_model',
                'model_type': 'transformer'
            },
            'training': {
                'num_epochs': 6,
                'batch_size': 4,
                'learning_rate': 1e-4
            },
            'scenario': {
                'type': 'scheduled_training',
                'stages': {
                    'warmup': {'epochs': 2, 'lr_multiplier': 0.1},
                    'main': {'epochs': 3, 'lr_multiplier': 1.0},
                    'fine_tune': {'epochs': 1, 'lr_multiplier': 0.1}
                }
            }
        }
        self.scenario = ScheduledTrainingScenario(self.config)
    
    def teardown_method(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_scenario_initialization(self):
        """测试场景初始化"""
        assert self.scenario.scenario_type == 'scheduled_training'
        assert len(self.scenario.stages) == 3
        assert 'warmup' in self.scenario.stages
        assert 'main' in self.scenario.stages
        assert 'fine_tune' in self.scenario.stages
    
    def test_stage_configuration(self):
        """测试阶段配置"""
        warmup_stage = self.scenario.stages['warmup']
        assert warmup_stage['epochs'] == 2
        assert warmup_stage['lr_multiplier'] == 0.1
        
        main_stage = self.scenario.stages['main']
        assert main_stage['epochs'] == 3
        assert main_stage['lr_multiplier'] == 1.0
    
    def test_validate_config(self):
        """测试配置验证"""
        # 测试有效配置
        assert self.scenario.validate_config() == True
        
        # 测试无效配置
        invalid_config = self.config.copy()
        del invalid_config['scenario']['stages']
        invalid_scenario = ScheduledTrainingScenario(invalid_config)
        
        with pytest.raises(ConfigurationError):
            invalid_scenario.validate_config()
    
    @patch('training.scenarios.scheduled_training_scenario.torch')
    def test_prepare_training(self, mock_torch):
        """测试训练准备"""
        # 模拟模型和数据
        mock_model = Mock()
        mock_train_loader = Mock()
        mock_val_loader = Mock()
        
        result = self.scenario.prepare_training(
            model=mock_model,
            train_loader=mock_train_loader,
            val_loader=mock_val_loader
        )
        
        assert 'optimizer' in result
        assert 'scheduler' in result
        assert 'criterion' in result
    
    def test_get_stage_config(self):
        """测试获取阶段配置"""
        warmup_config = self.scenario.get_stage_config('warmup')
        assert warmup_config['epochs'] == 2
        assert warmup_config['lr_multiplier'] == 0.1
        
        # 测试不存在的阶段
        with pytest.raises(KeyError):
            self.scenario.get_stage_config('nonexistent')


class TestAdvancedModelScenario:
    """高级模型训练场景测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'global': {
                'project_name': 'test_advanced_model',
                'output_dir': self.temp_dir
            },
            'model': {
                'model_name': 'advanced_transformer',
                'model_type': 'transformer',
                'use_moe': True,
                'num_experts': 8
            },
            'training': {
                'num_epochs': 10,
                'batch_size': 8,
                'learning_rate': 2e-4,
                'use_gradient_checkpointing': True
            },
            'scenario': {
                'type': 'advanced_model',
                'optimization': {
                    'use_mixed_precision': True,
                    'gradient_accumulation_steps': 4,
                    'max_grad_norm': 1.0
                },
                'regularization': {
                    'dropout_rate': 0.1,
                    'weight_decay': 0.01
                }
            }
        }
        self.scenario = AdvancedModelScenario(self.config)
    
    def teardown_method(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_scenario_initialization(self):
        """测试场景初始化"""
        assert self.scenario.scenario_type == 'advanced_model'
        assert self.scenario.use_moe == True
        assert self.scenario.num_experts == 8
    
    def test_optimization_config(self):
        """测试优化配置"""
        opt_config = self.scenario.optimization_config
        assert opt_config['use_mixed_precision'] == True
        assert opt_config['gradient_accumulation_steps'] == 4
        assert opt_config['max_grad_norm'] == 1.0
    
    def test_model_preparation(self):
        """测试模型准备"""
        with patch('training.scenarios.advanced_model_scenario.torch.nn.Module') as mock_model:
            mock_model_instance = Mock()
            mock_model.return_value = mock_model_instance
            
            prepared_model = self.scenario.prepare_model()
            assert prepared_model is not None
    
    def test_validate_config(self):
        """测试配置验证"""
        assert self.scenario.validate_config() == True
        
        # 测试缺少必要配置
        invalid_config = self.config.copy()
        del invalid_config['scenario']['optimization']
        invalid_scenario = AdvancedModelScenario(invalid_config)
        
        with pytest.raises(ConfigurationError):
            invalid_scenario.validate_config()


class TestMultiModalScenario:
    """多模态训练场景测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'global': {
                'project_name': 'test_multimodal',
                'output_dir': self.temp_dir
            },
            'model': {
                'model_name': 'multimodal_transformer',
                'model_type': 'multimodal'
            },
            'training': {
                'num_epochs': 5,
                'batch_size': 4,
                'learning_rate': 1e-4
            },
            'scenario': {
                'type': 'multimodal',
                'modalities': ['text', 'image', 'audio'],
                'fusion_strategy': 'late_fusion',
                'modality_weights': {
                    'text': 0.4,
                    'image': 0.4,
                    'audio': 0.2
                }
            }
        }
        self.scenario = MultiModalScenario(self.config)
    
    def teardown_method(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_scenario_initialization(self):
        """测试场景初始化"""
        assert self.scenario.scenario_type == 'multimodal'
        assert len(self.scenario.modalities) == 3
        assert 'text' in self.scenario.modalities
        assert 'image' in self.scenario.modalities
        assert 'audio' in self.scenario.modalities
    
    def test_fusion_strategy(self):
        """测试融合策略"""
        assert self.scenario.fusion_strategy == 'late_fusion'
        
        # 测试权重配置
        weights = self.scenario.modality_weights
        assert weights['text'] == 0.4
        assert weights['image'] == 0.4
        assert weights['audio'] == 0.2
        assert sum(weights.values()) == 1.0
    
    def test_modality_validation(self):
        """测试模态验证"""
        assert self.scenario.validate_config() == True
        
        # 测试无效的模态权重
        invalid_config = self.config.copy()
        invalid_config['scenario']['modality_weights'] = {
            'text': 0.5,
            'image': 0.6  # 总和超过1.0
        }
        invalid_scenario = MultiModalScenario(invalid_config)
        
        with pytest.raises(ConfigurationError):
            invalid_scenario.validate_config()
    
    def test_data_preparation(self):
        """测试数据准备"""
        with patch('training.scenarios.multimodal_scenario.MultiModalDataset') as mock_dataset:
            mock_dataset_instance = Mock()
            mock_dataset.return_value = mock_dataset_instance
            
            result = self.scenario.prepare_data('/fake/data/path')
            assert result is not None


class TestScenarioManager:
    """场景管理器测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ScenarioManager()
    
    def teardown_method(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_scenario_registration(self):
        """测试场景注册"""
        # 检查默认场景是否已注册
        scenarios = self.manager.get_available_scenarios()
        assert 'scheduled_training' in scenarios
        assert 'advanced_model' in scenarios
        assert 'multimodal' in scenarios
    
    def test_create_scenario(self):
        """测试创建场景"""
        config = {
            'scenario': {
                'type': 'scheduled_training',
                'stages': {
                    'warmup': {'epochs': 1, 'lr_multiplier': 0.1},
                    'main': {'epochs': 2, 'lr_multiplier': 1.0},
                    'fine_tune': {'epochs': 1, 'lr_multiplier': 0.1}
                }
            }
        }
        
        scenario = self.manager.create_scenario(config)
        assert isinstance(scenario, ScheduledTrainingScenario)
        assert scenario.scenario_type == 'scheduled_training'
    
    def test_invalid_scenario_type(self):
        """测试无效场景类型"""
        config = {
            'scenario': {
                'type': 'nonexistent_scenario'
            }
        }
        
        with pytest.raises(ValueError):
            self.manager.create_scenario(config)
    
    def test_scenario_validation(self):
        """测试场景验证"""
        valid_config = {
            'scenario': {
                'type': 'scheduled_training',
                'stages': {
                    'warmup': {'epochs': 1, 'lr_multiplier': 0.1},
                    'main': {'epochs': 2, 'lr_multiplier': 1.0},
                    'fine_tune': {'epochs': 1, 'lr_multiplier': 0.1}
                }
            }
        }
        
        assert self.manager.validate_scenario_config(valid_config) == True
        
        # 测试无效配置
        invalid_config = {
            'scenario': {
                'type': 'scheduled_training'
                # 缺少stages配置
            }
        }
        
        with pytest.raises(ConfigurationError):
            self.manager.validate_scenario_config(invalid_config)


class TestScenarioIntegration:
    """场景集成测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ScenarioManager()
    
    def teardown_method(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_scenario_execution(self):
        """测试端到端场景执行"""
        config = {
            'global': {
                'project_name': 'integration_test',
                'output_dir': self.temp_dir
            },
            'model': {
                'model_name': 'test_model',
                'model_type': 'transformer'
            },
            'training': {
                'num_epochs': 3,
                'batch_size': 2,
                'learning_rate': 1e-4
            },
            'scenario': {
                'type': 'scheduled_training',
                'stages': {
                    'warmup': {'epochs': 1, 'lr_multiplier': 0.1},
                    'main': {'epochs': 1, 'lr_multiplier': 1.0},
                    'fine_tune': {'epochs': 1, 'lr_multiplier': 0.1}
                }
            }
        }
        
        # 创建场景
        scenario = self.manager.create_scenario(config)
        assert scenario is not None
        
        # 验证配置
        assert scenario.validate_config() == True
        
        # 模拟训练准备
        with patch('torch.nn.Module') as mock_model, \
             patch('torch.utils.data.DataLoader') as mock_loader:
            
            mock_model_instance = Mock()
            mock_loader_instance = Mock()
            
            result = scenario.prepare_training(
                model=mock_model_instance,
                train_loader=mock_loader_instance,
                val_loader=mock_loader_instance
            )
            
            assert result is not None
    
    def test_scenario_switching(self):
        """测试场景切换"""
        # 创建不同类型的场景
        scheduled_config = {
            'scenario': {
                'type': 'scheduled_training',
                'stages': {
                    'warmup': {'epochs': 1, 'lr_multiplier': 0.1},
                    'main': {'epochs': 2, 'lr_multiplier': 1.0},
                    'fine_tune': {'epochs': 1, 'lr_multiplier': 0.1}
                }
            }
        }
        
        multimodal_config = {
            'scenario': {
                'type': 'multimodal',
                'modalities': ['text', 'image'],
                'fusion_strategy': 'early_fusion',
                'modality_weights': {
                    'text': 0.6,
                    'image': 0.4
                }
            }
        }
        
        # 创建场景
        scheduled_scenario = self.manager.create_scenario(scheduled_config)
        multimodal_scenario = self.manager.create_scenario(multimodal_config)
        
        assert isinstance(scheduled_scenario, ScheduledTrainingScenario)
        assert isinstance(multimodal_scenario, MultiModalScenario)
        
        assert scheduled_scenario.scenario_type == 'scheduled_training'
        assert multimodal_scenario.scenario_type == 'multimodal'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])