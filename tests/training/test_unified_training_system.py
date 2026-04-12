"""统一训练系统测试用例

测试统一训练系统的核心功能和三阶段训练流程。
"""

import pytest
import torch
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch, call
from pathlib import Path
import json
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.unified_training_system import UnifiedTrainingSystem, TrainingConfig
from training.core import TrainingError, ConfigurationError


class TestTrainingConfig:
    """训练配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = TrainingConfig()
        
        assert config.model_name == "custom_model"
        assert config.task_type == "classification"
        assert config.num_epochs == 10
        assert config.batch_size == 16
        assert config.learning_rate == 2e-5
        assert config.use_fp16 == True
        assert config.use_distributed == False
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = TrainingConfig(
            model_name="test_model",
            task_type="generation",
            num_epochs=20,
            batch_size=32,
            learning_rate=1e-4,
            use_fp16=False
        )
        
        assert config.model_name == "test_model"
        assert config.task_type == "generation"
        assert config.num_epochs == 20
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.use_fp16 == False
    
    def test_three_stage_config(self):
        """测试三阶段训练配置"""
        config = TrainingConfig(
            use_three_stage=True,
            three_stage_config={
                'warmup_epochs': 2,
                'main_epochs': 5,
                'fine_tune_epochs': 3,
                'warmup_lr_ratio': 0.1,
                'fine_tune_lr_ratio': 0.1
            }
        )
        
        assert config.use_three_stage == True
        assert config.three_stage_config['warmup_epochs'] == 2
        assert config.three_stage_config['main_epochs'] == 5
        assert config.three_stage_config['fine_tune_epochs'] == 3


class TestUnifiedTrainingSystem:
    """统一训练系统测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TrainingConfig(
            model_name="test_model",
            output_dir=self.temp_dir,
            num_epochs=3,
            batch_size=4,
            learning_rate=1e-4
        )
        self.training_system = UnifiedTrainingSystem(self.config)
    
    def teardown_method(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_system_initialization(self):
        """测试系统初始化"""
        assert self.training_system.config == self.config
        assert self.training_system.device is not None
        assert self.training_system.logger is not None
    
    @patch('training.unified_training_system.torch.cuda.is_available')
    def test_device_selection(self, mock_cuda_available):
        """测试设备选择"""
        # 测试CUDA可用
        mock_cuda_available.return_value = True
        system = UnifiedTrainingSystem(self.config)
        assert 'cuda' in str(system.device)
        
        # 测试CUDA不可用
        mock_cuda_available.return_value = False
        system = UnifiedTrainingSystem(self.config)
        assert str(system.device) == 'cpu'
    
    def test_prepare_components(self):
        """测试组件准备"""
        # 模拟模型和数据
        mock_model = Mock(spec=torch.nn.Module)
        mock_train_loader = Mock()
        mock_val_loader = Mock()
        
        with patch.object(self.training_system, '_setup_optimizer') as mock_setup_opt, \
             patch.object(self.training_system, '_setup_scheduler') as mock_setup_sched, \
             patch.object(self.training_system, '_setup_criterion') as mock_setup_crit:
            
            mock_setup_opt.return_value = Mock()
            mock_setup_sched.return_value = Mock()
            mock_setup_crit.return_value = Mock()
            
            result = self.training_system._prepare_components(
                mock_model, mock_train_loader, mock_val_loader
            )
            
            assert 'model' in result
            assert 'optimizer' in result
            assert 'scheduler' in result
            assert 'criterion' in result
            
            mock_setup_opt.assert_called_once()
            mock_setup_sched.assert_called_once()
            mock_setup_crit.assert_called_once()
    
    def test_lr_finder(self):
        """测试学习率查找"""
        mock_model = Mock(spec=torch.nn.Module)
        mock_train_loader = Mock()
        
        with patch('training.unified_training_system.LRFinder') as mock_lr_finder:
            mock_finder_instance = Mock()
            mock_lr_finder.return_value = mock_finder_instance
            mock_finder_instance.range_test.return_value = None
            mock_finder_instance.plot.return_value = None
            
            self.training_system._find_optimal_lr(mock_model, mock_train_loader)
            
            mock_lr_finder.assert_called_once()
            mock_finder_instance.range_test.assert_called_once()
    
    @patch('training.unified_training_system.torch.save')
    def test_save_checkpoint(self, mock_torch_save):
        """测试检查点保存"""
        mock_model = Mock()
        mock_optimizer = Mock()
        mock_scheduler = Mock()
        
        checkpoint_data = {
            'epoch': 5,
            'model_state_dict': mock_model.state_dict(),
            'optimizer_state_dict': mock_optimizer.state_dict(),
            'scheduler_state_dict': mock_scheduler.state_dict(),
            'loss': 0.5,
            'metrics': {'accuracy': 0.85}
        }
        
        self.training_system._save_checkpoint(
            checkpoint_data, 
            epoch=5, 
            is_best=True
        )
        
        # 验证保存调用
        assert mock_torch_save.call_count >= 1
    
    def test_standard_training_mode(self):
        """测试标准训练模式"""
        mock_model = Mock(spec=torch.nn.Module)
        mock_train_loader = Mock()
        mock_val_loader = Mock()
        
        # 设置配置为标准训练
        self.config.training_mode = "standard"
        
        with patch.object(self.training_system, '_standard_train') as mock_standard_train:
            mock_standard_train.return_value = {
                'final_loss': 0.3,
                'final_metrics': {'accuracy': 0.9},
                'training_time': 100.0
            }
            
            result = self.training_system.train(
                mock_model, mock_train_loader, mock_val_loader
            )
            
            mock_standard_train.assert_called_once()
            assert 'final_loss' in result
            assert 'final_metrics' in result


class TestThreeStageTraining:
    """三阶段训练测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TrainingConfig(
            model_name="test_model",
            output_dir=self.temp_dir,
            training_mode="three_stage",
            use_three_stage=True,
            three_stage_config={
                'warmup_epochs': 2,
                'main_epochs': 3,
                'fine_tune_epochs': 2,
                'warmup_lr_ratio': 0.1,
                'fine_tune_lr_ratio': 0.1
            }
        )
        self.training_system = UnifiedTrainingSystem(self.config)
    
    def teardown_method(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_three_stage_config_validation(self):
        """测试三阶段配置验证"""
        assert self.config.use_three_stage == True
        assert self.config.three_stage_config['warmup_epochs'] == 2
        assert self.config.three_stage_config['main_epochs'] == 3
        assert self.config.three_stage_config['fine_tune_epochs'] == 2
    
    def test_three_stage_training_mode(self):
        """测试三阶段训练模式"""
        mock_model = Mock(spec=torch.nn.Module)
        mock_train_loader = Mock()
        mock_val_loader = Mock()
        
        with patch.object(self.training_system, '_three_stage_train') as mock_three_stage_train:
            mock_three_stage_train.return_value = {
                'warmup_results': {'loss': 0.8, 'accuracy': 0.6},
                'main_results': {'loss': 0.4, 'accuracy': 0.8},
                'fine_tune_results': {'loss': 0.2, 'accuracy': 0.9},
                'final_loss': 0.2,
                'final_metrics': {'accuracy': 0.9},
                'training_time': 150.0
            }
            
            result = self.training_system.train(
                mock_model, mock_train_loader, mock_val_loader
            )
            
            mock_three_stage_train.assert_called_once()
            assert 'warmup_results' in result
            assert 'main_results' in result
            assert 'fine_tune_results' in result
    
    @patch('training.unified_training_system.torch.optim.AdamW')
    @patch('training.unified_training_system.torch.nn.CrossEntropyLoss')
    def test_three_stage_execution(self, mock_criterion, mock_optimizer):
        """测试三阶段执行流程"""
        # 模拟组件
        mock_model = Mock(spec=torch.nn.Module)
        mock_train_loader = Mock()
        mock_val_loader = Mock()
        
        # 模拟数据批次
        mock_batch = {
            'input_ids': torch.randn(2, 10),
            'labels': torch.randint(0, 2, (2,))
        }
        mock_train_loader.__iter__ = Mock(return_value=iter([mock_batch]))
        mock_train_loader.__len__ = Mock(return_value=1)
        mock_val_loader.__iter__ = Mock(return_value=iter([mock_batch]))
        mock_val_loader.__len__ = Mock(return_value=1)
        
        # 模拟模型输出
        mock_model.return_value = Mock(loss=torch.tensor(0.5))
        mock_model.train = Mock()
        mock_model.eval = Mock()
        
        # 模拟优化器和损失函数
        mock_optimizer_instance = Mock()
        mock_optimizer.return_value = mock_optimizer_instance
        mock_criterion_instance = Mock()
        mock_criterion.return_value = mock_criterion_instance
        mock_criterion_instance.return_value = torch.tensor(0.5)
        
        with patch.object(self.training_system, '_setup_scheduler') as mock_setup_scheduler, \
             patch.object(self.training_system, '_validate_batch') as mock_validate, \
             patch.object(self.training_system, '_save_checkpoint') as mock_save:
            
            mock_scheduler = Mock()
            mock_setup_scheduler.return_value = mock_scheduler
            mock_validate.return_value = (torch.tensor(0.3), {'accuracy': 0.85})
            
            result = self.training_system._three_stage_train(
                mock_model, mock_train_loader, mock_val_loader
            )
            
            # 验证结果结构
            assert 'warmup_results' in result
            assert 'main_results' in result
            assert 'fine_tune_results' in result
            assert 'final_loss' in result
            assert 'final_metrics' in result
            assert 'training_time' in result
    
    def test_stage_configuration(self):
        """测试阶段配置"""
        stage_config = self.training_system._get_stage_config('warmup')
        assert stage_config['epochs'] == 2
        assert stage_config['lr_ratio'] == 0.1
        
        stage_config = self.training_system._get_stage_config('main')
        assert stage_config['epochs'] == 3
        assert stage_config['lr_ratio'] == 1.0
        
        stage_config = self.training_system._get_stage_config('fine_tune')
        assert stage_config['epochs'] == 2
        assert stage_config['lr_ratio'] == 0.1
    
    def test_invalid_stage_config(self):
        """测试无效阶段配置"""
        with pytest.raises(ValueError):
            self.training_system._get_stage_config('invalid_stage')


class TestTrainingIntegration:
    """训练集成测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_training(self):
        """测试端到端训练流程"""
        config = TrainingConfig(
            model_name="integration_test_model",
            output_dir=self.temp_dir,
            num_epochs=2,
            batch_size=2,
            learning_rate=1e-4,
            training_mode="standard"
        )
        
        training_system = UnifiedTrainingSystem(config)
        
        # 创建简单的模型和数据
        with patch('torch.nn.Module') as mock_model_class, \
             patch('torch.utils.data.DataLoader') as mock_dataloader_class:
            
            mock_model = Mock(spec=torch.nn.Module)
            mock_model_class.return_value = mock_model
            
            mock_train_loader = Mock()
            mock_val_loader = Mock()
            mock_dataloader_class.return_value = mock_train_loader
            
            # 模拟训练过程
            with patch.object(training_system, '_standard_train') as mock_train:
                mock_train.return_value = {
                    'final_loss': 0.2,
                    'final_metrics': {'accuracy': 0.9},
                    'training_time': 50.0
                }
                
                result = training_system.train(
                    mock_model, mock_train_loader, mock_val_loader
                )
                
                assert result is not None
                assert 'final_loss' in result
                assert 'final_metrics' in result
    
    def test_training_with_different_modes(self):
        """测试不同训练模式"""
        modes = ['standard', 'three_stage']
        
        for mode in modes:
            config = TrainingConfig(
                model_name=f"test_model_{mode}",
                output_dir=self.temp_dir,
                training_mode=mode,
                num_epochs=2
            )
            
            if mode == 'three_stage':
                config.use_three_stage = True
                config.three_stage_config = {
                    'warmup_epochs': 1,
                    'main_epochs': 1,
                    'fine_tune_epochs': 1,
                    'warmup_lr_ratio': 0.1,
                    'fine_tune_lr_ratio': 0.1
                }
            
            training_system = UnifiedTrainingSystem(config)
            assert training_system.config.training_mode == mode
    
    def test_error_handling(self):
        """测试错误处理"""
        config = TrainingConfig(
            model_name="error_test_model",
            output_dir=self.temp_dir
        )
        
        training_system = UnifiedTrainingSystem(config)
        
        # 测试无效输入
        with pytest.raises(TypeError):
            training_system.train(None, None, None)
    
    def test_checkpoint_loading(self):
        """测试检查点加载"""
        config = TrainingConfig(
            model_name="checkpoint_test_model",
            output_dir=self.temp_dir,
            resume_from_checkpoint=True
        )
        
        training_system = UnifiedTrainingSystem(config)
        
        # 模拟检查点文件
        checkpoint_path = Path(self.temp_dir) / "checkpoint.pth"
        mock_checkpoint = {
            'epoch': 5,
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'loss': 0.3
        }
        
        with patch('torch.load') as mock_load, \
             patch('os.path.exists') as mock_exists:
            
            mock_exists.return_value = True
            mock_load.return_value = mock_checkpoint
            
            loaded_checkpoint = training_system._load_checkpoint(str(checkpoint_path))
            assert loaded_checkpoint['epoch'] == 5
            assert loaded_checkpoint['loss'] == 0.3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])