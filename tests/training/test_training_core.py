"""训练核心组件测试用例

测试训练核心组件的功能，包括训练器、优化器、调度器等。
"""

import pytest
import torch
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.core.unified_trainer import UnifiedTrainer
from training.core.optimizer_factory import OptimizerFactory
from training.core.scheduler_factory import SchedulerFactory
from training.core.device_manager import DeviceManager
from training.core.checkpoint_manager import CheckpointManager
from training.config.unified_config_manager import UnifiedConfigManager
from training.core import TrainingError, ConfigurationError


class TestDeviceManager:
    """设备管理器测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.device_manager = DeviceManager()
    
    def test_device_detection(self):
        """测试设备检测"""
        device = self.device_manager.get_device()
        
        # 设备应该是torch.device对象
        assert isinstance(device, torch.device)
        
        # 设备类型应该是cpu或cuda
        assert device.type in ['cpu', 'cuda']
    
    def test_cuda_availability(self):
        """测试CUDA可用性检测"""
        cuda_available = self.device_manager.is_cuda_available()
        
        # 应该返回布尔值
        assert isinstance(cuda_available, bool)
        
        # 如果CUDA可用，设备应该是cuda
        if cuda_available:
            device = self.device_manager.get_device()
            assert device.type == 'cuda'
    
    def test_device_count(self):
        """测试设备数量"""
        device_count = self.device_manager.get_device_count()
        
        # 设备数量应该是正整数
        assert isinstance(device_count, int)
        assert device_count > 0
    
    def test_memory_info(self):
        """测试内存信息"""
        memory_info = self.device_manager.get_memory_info()
        
        assert 'total' in memory_info
        assert 'available' in memory_info
        assert 'used' in memory_info
        
        # 内存值应该是非负数
        assert memory_info['total'] >= 0
        assert memory_info['available'] >= 0
        assert memory_info['used'] >= 0
    
    def test_device_selection_strategy(self):
        """测试设备选择策略"""
        # 测试自动选择
        device = self.device_manager.select_device('auto')
        assert isinstance(device, torch.device)
        
        # 测试强制CPU
        cpu_device = self.device_manager.select_device('cpu')
        assert cpu_device.type == 'cpu'
        
        # 测试指定设备ID
        if self.device_manager.is_cuda_available():
            cuda_device = self.device_manager.select_device('cuda:0')
            assert cuda_device.type == 'cuda'
            assert cuda_device.index == 0


class TestOptimizerFactory:
    """优化器工厂测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.factory = OptimizerFactory()
        
        # 创建简单的模型用于测试
        self.model = torch.nn.Linear(10, 1)
    
    def test_create_adam_optimizer(self):
        """测试创建Adam优化器"""
        optimizer = self.factory.create_optimizer(
            model=self.model,
            optimizer_type='adam',
            learning_rate=1e-3,
            weight_decay=1e-4
        )
        
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]['lr'] == 1e-3
        assert optimizer.param_groups[0]['weight_decay'] == 1e-4
    
    def test_create_sgd_optimizer(self):
        """测试创建SGD优化器"""
        optimizer = self.factory.create_optimizer(
            model=self.model,
            optimizer_type='sgd',
            learning_rate=1e-2,
            momentum=0.9
        )
        
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]['lr'] == 1e-2
        assert optimizer.param_groups[0]['momentum'] == 0.9
    
    def test_create_adamw_optimizer(self):
        """测试创建AdamW优化器"""
        optimizer = self.factory.create_optimizer(
            model=self.model,
            optimizer_type='adamw',
            learning_rate=5e-4,
            weight_decay=1e-2
        )
        
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]['lr'] == 5e-4
        assert optimizer.param_groups[0]['weight_decay'] == 1e-2
    
    def test_invalid_optimizer_type(self):
        """测试无效优化器类型"""
        with pytest.raises(ValueError):
            self.factory.create_optimizer(
                model=self.model,
                optimizer_type='invalid_optimizer',
                learning_rate=1e-3
            )
    
    def test_optimizer_parameters(self):
        """测试优化器参数设置"""
        # 测试自定义参数
        optimizer = self.factory.create_optimizer(
            model=self.model,
            optimizer_type='adam',
            learning_rate=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        assert optimizer.param_groups[0]['betas'] == (0.9, 0.999)
        assert optimizer.param_groups[0]['eps'] == 1e-8


class TestSchedulerFactory:
    """调度器工厂测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.factory = SchedulerFactory()
        
        # 创建简单的模型和优化器用于测试
        self.model = torch.nn.Linear(10, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def test_create_step_lr_scheduler(self):
        """测试创建StepLR调度器"""
        scheduler = self.factory.create_scheduler(
            optimizer=self.optimizer,
            scheduler_type='step_lr',
            step_size=10,
            gamma=0.1
        )
        
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 10
        assert scheduler.gamma == 0.1
    
    def test_create_cosine_scheduler(self):
        """测试创建余弦退火调度器"""
        scheduler = self.factory.create_scheduler(
            optimizer=self.optimizer,
            scheduler_type='cosine_annealing',
            T_max=100,
            eta_min=1e-6
        )
        
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 100
        assert scheduler.eta_min == 1e-6
    
    def test_create_exponential_scheduler(self):
        """测试创建指数衰减调度器"""
        scheduler = self.factory.create_scheduler(
            optimizer=self.optimizer,
            scheduler_type='exponential',
            gamma=0.95
        )
        
        assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
        assert scheduler.gamma == 0.95
    
    def test_create_warmup_scheduler(self):
        """测试创建预热调度器"""
        scheduler = self.factory.create_scheduler(
            optimizer=self.optimizer,
            scheduler_type='linear_warmup',
            warmup_steps=1000,
            total_steps=10000
        )
        
        # 预热调度器应该是自定义类型
        assert hasattr(scheduler, 'step')
        assert hasattr(scheduler, 'get_last_lr')
    
    def test_invalid_scheduler_type(self):
        """测试无效调度器类型"""
        with pytest.raises(ValueError):
            self.factory.create_scheduler(
                optimizer=self.optimizer,
                scheduler_type='invalid_scheduler'
            )
    
    def test_scheduler_step(self):
        """测试调度器步进"""
        scheduler = self.factory.create_scheduler(
            optimizer=self.optimizer,
            scheduler_type='step_lr',
            step_size=5,
            gamma=0.5
        )
        
        initial_lr = self.optimizer.param_groups[0]['lr']
        
        # 前5步学习率不变
        for _ in range(5):
            scheduler.step()
        
        # 第6步学习率应该减半
        scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        
        assert abs(current_lr - initial_lr * 0.5) < 1e-8


class TestCheckpointManager:
    """检查点管理器测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(self.temp_dir)
        
        # 创建简单的模型用于测试
        self.model = torch.nn.Linear(10, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def teardown_method(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_save_checkpoint(self):
        """测试保存检查点"""
        checkpoint_data = {
            'epoch': 10,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': 0.5,
            'metrics': {'accuracy': 0.85}
        }
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            checkpoint_data,
            filename='test_checkpoint.pth'
        )
        
        # 检查文件是否存在
        assert os.path.exists(checkpoint_path)
        
        # 检查文件路径
        expected_path = os.path.join(self.temp_dir, 'test_checkpoint.pth')
        assert checkpoint_path == expected_path
    
    def test_load_checkpoint(self):
        """测试加载检查点"""
        # 先保存检查点
        original_data = {
            'epoch': 15,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': 0.3,
            'metrics': {'accuracy': 0.9}
        }
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            original_data,
            filename='load_test.pth'
        )
        
        # 加载检查点
        loaded_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # 验证数据
        assert loaded_data['epoch'] == 15
        assert loaded_data['loss'] == 0.3
        assert loaded_data['metrics']['accuracy'] == 0.9
    
    def test_get_latest_checkpoint(self):
        """测试获取最新检查点"""
        # 保存多个检查点
        for i in range(3):
            checkpoint_data = {
                'epoch': i + 1,
                'model_state_dict': self.model.state_dict(),
                'loss': 1.0 - i * 0.1
            }
            
            self.checkpoint_manager.save_checkpoint(
                checkpoint_data,
                filename=f'checkpoint_epoch_{i+1}.pth'
            )
        
        # 获取最新检查点
        latest_path = self.checkpoint_manager.get_latest_checkpoint()
        
        assert latest_path is not None
        assert 'checkpoint_epoch_3.pth' in latest_path
    
    def test_list_checkpoints(self):
        """测试列出检查点"""
        # 保存几个检查点
        checkpoint_names = ['checkpoint_1.pth', 'checkpoint_2.pth', 'best_model.pth']
        
        for name in checkpoint_names:
            checkpoint_data = {
                'epoch': 1,
                'model_state_dict': self.model.state_dict()
            }
            self.checkpoint_manager.save_checkpoint(checkpoint_data, filename=name)
        
        # 列出检查点
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        assert len(checkpoints) == 3
        for name in checkpoint_names:
            assert any(name in cp for cp in checkpoints)
    
    def test_delete_checkpoint(self):
        """测试删除检查点"""
        # 保存检查点
        checkpoint_data = {
            'epoch': 1,
            'model_state_dict': self.model.state_dict()
        }
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            checkpoint_data,
            filename='to_delete.pth'
        )
        
        # 确认文件存在
        assert os.path.exists(checkpoint_path)
        
        # 删除检查点
        self.checkpoint_manager.delete_checkpoint('to_delete.pth')
        
        # 确认文件已删除
        assert not os.path.exists(checkpoint_path)
    
    def test_checkpoint_not_found(self):
        """测试检查点文件不存在"""
        with pytest.raises(FileNotFoundError):
            self.checkpoint_manager.load_checkpoint('nonexistent.pth')
    
    def test_auto_cleanup(self):
        """测试自动清理旧检查点"""
        # 设置最大保留数量
        self.checkpoint_manager.max_checkpoints = 3
        
        # 保存5个检查点
        for i in range(5):
            checkpoint_data = {
                'epoch': i + 1,
                'model_state_dict': self.model.state_dict()
            }
            
            self.checkpoint_manager.save_checkpoint(
                checkpoint_data,
                filename=f'auto_cleanup_{i+1}.pth'
            )
        
        # 应该只保留最新的3个
        checkpoints = self.checkpoint_manager.list_checkpoints()
        assert len(checkpoints) <= 3
        
        # 最新的检查点应该存在
        latest_checkpoints = ['auto_cleanup_3.pth', 'auto_cleanup_4.pth', 'auto_cleanup_5.pth']
        for name in latest_checkpoints:
            assert any(name in cp for cp in checkpoints)


class TestUnifiedTrainer:
    """统一训练器测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建训练配置
        self.config = TrainingConfig(
            project_name='test_project',
            output_dir=self.temp_dir,
            num_epochs=3,
            batch_size=4,
            learning_rate=1e-3,
            model_name='test_model',
            model_type='linear'
        )
        
        # 创建简单的模型和数据
        self.model = torch.nn.Linear(10, 1)
        
        # 创建模拟数据加载器
        self.train_data = [(torch.randn(4, 10), torch.randn(4, 1)) for _ in range(5)]
        self.val_data = [(torch.randn(4, 10), torch.randn(4, 1)) for _ in range(3)]
        
        self.trainer = UnifiedTrainer(self.config)
    
    def teardown_method(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """测试训练器初始化"""
        assert self.trainer.config == self.config
        assert self.trainer.device is not None
        assert self.trainer.checkpoint_manager is not None
    
    def test_setup_model(self):
        """测试模型设置"""
        self.trainer.setup_model(self.model)
        
        assert self.trainer.model is not None
        assert self.trainer.optimizer is not None
        assert self.trainer.scheduler is not None
    
    def test_setup_data(self):
        """测试数据设置"""
        self.trainer.setup_data(self.train_data, self.val_data)
        
        assert self.trainer.train_loader is not None
        assert self.trainer.val_loader is not None
    
    @patch('training.core.trainer.UnifiedTrainer._train_epoch')
    @patch('training.core.trainer.UnifiedTrainer._validate_epoch')
    def test_train_method(self, mock_validate, mock_train):
        """测试训练方法"""
        # 设置模拟返回值
        mock_train.return_value = {'loss': 0.5, 'accuracy': 0.8}
        mock_validate.return_value = {'loss': 0.4, 'accuracy': 0.85}
        
        # 设置模型和数据
        self.trainer.setup_model(self.model)
        self.trainer.setup_data(self.train_data, self.val_data)
        
        # 执行训练
        history = self.trainer.train()
        
        # 验证调用
        assert mock_train.call_count == self.config.num_epochs
        assert mock_validate.call_count == self.config.num_epochs
        
        # 验证历史记录
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == self.config.num_epochs
    
    def test_save_load_checkpoint(self):
        """测试保存和加载检查点"""
        # 设置模型
        self.trainer.setup_model(self.model)
        
        # 保存检查点
        checkpoint_path = self.trainer.save_checkpoint(
            epoch=5,
            metrics={'loss': 0.3, 'accuracy': 0.9}
        )
        
        assert os.path.exists(checkpoint_path)
        
        # 创建新的训练器并加载检查点
        new_trainer = UnifiedTrainer(self.config)
        new_trainer.setup_model(torch.nn.Linear(10, 1))
        
        loaded_epoch, loaded_metrics = new_trainer.load_checkpoint(checkpoint_path)
        
        assert loaded_epoch == 5
        assert loaded_metrics['loss'] == 0.3
        assert loaded_metrics['accuracy'] == 0.9
    
    def test_learning_rate_finding(self):
        """测试学习率查找"""
        self.trainer.setup_model(self.model)
        self.trainer.setup_data(self.train_data, self.val_data)
        
        # 执行学习率查找
        lr_finder_results = self.trainer.find_learning_rate(
            start_lr=1e-6,
            end_lr=1e-1,
            num_iter=10
        )
        
        assert 'learning_rates' in lr_finder_results
        assert 'losses' in lr_finder_results
        assert len(lr_finder_results['learning_rates']) == 10
        assert len(lr_finder_results['losses']) == 10
    
    def test_early_stopping(self):
        """测试早停机制"""
        # 启用早停
        self.config.early_stopping = True
        self.config.patience = 2
        
        trainer = UnifiedTrainer(self.config)
        trainer.setup_model(self.model)
        trainer.setup_data(self.train_data, self.val_data)
        
        # 模拟验证损失不改善的情况
        with patch.object(trainer, '_validate_epoch') as mock_validate:
            mock_validate.side_effect = [
                {'loss': 1.0},  # epoch 1
                {'loss': 1.1},  # epoch 2 (worse)
                {'loss': 1.2},  # epoch 3 (worse)
            ]
            
            with patch.object(trainer, '_train_epoch') as mock_train:
                mock_train.return_value = {'loss': 0.5}
                
                history = trainer.train()
                
                # 应该在第3个epoch后停止（patience=2）
                assert len(history['train_loss']) <= 3
    
    def test_gradient_clipping(self):
        """测试梯度裁剪"""
        # 启用梯度裁剪
        self.config.max_grad_norm = 1.0
        
        trainer = UnifiedTrainer(self.config)
        trainer.setup_model(self.model)
        
        # 创建大梯度
        for param in self.model.parameters():
            param.grad = torch.randn_like(param) * 10  # 大梯度
        
        # 应用梯度裁剪
        trainer._clip_gradients()
        
        # 检查梯度范数
        total_norm = 0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        assert total_norm <= self.config.max_grad_norm + 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])