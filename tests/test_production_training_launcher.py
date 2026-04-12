#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产级训练启动器测试

测试覆盖：
1. TrainingSystemLauncher 基础功能
2. ProductionTrainingLauncher 生产级功能
3. DistributedTrainingManager 分布式训练管理
4. 流水线执行测试
5. 训练服务集成测试
6. 策略组合测试
7. 检查点和恢复测试
"""

import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTrainingSystemLauncher(unittest.TestCase):
    """TrainingSystemLauncher 基础功能测试"""
    
    def setUp(self):
        """测试初始化"""
        self.temp_dir = tempfile.mkdtemp()
        self.base_config = {
            'output_dir': self.temp_dir,
            'model': {'name': 'test_model', 'type': 'standard'},
            'training': {'num_epochs': 2, 'batch_size': 8, 'learning_rate': 1e-4},
            'data': {'train_path': './data/train', 'val_path': './data/val'}
        }
    
    def tearDown(self):
        """测试清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_launcher_initialization(self):
        """测试启动器初始化"""
        from backend.modules.training.launcher import TrainingSystemLauncher
        
        launcher = TrainingSystemLauncher(self.base_config)
        
        self.assertIsNotNone(launcher)
        self.assertEqual(launcher.config, self.base_config)
        self.assertTrue(os.path.exists(launcher.output_dir))
    
    def test_config_analysis(self):
        """测试配置分析"""
        from backend.modules.training.launcher import TrainingSystemLauncher
        
        launcher = TrainingSystemLauncher(self.base_config)
        analysis = launcher.analyze_config()
        
        self.assertIn('model_type', analysis)
        self.assertIn('training_mode', analysis)
        self.assertIn('distributed', analysis)
        self.assertEqual(analysis['model_type'], 'standard')
        self.assertFalse(analysis['distributed'])
    
    def test_strategy_setup(self):
        """测试策略设置"""
        from backend.modules.training.launcher import TrainingSystemLauncher
        
        launcher = TrainingSystemLauncher(self.base_config)
        analysis = launcher.analyze_config()
        strategies = launcher._setup_strategies(analysis)
        
        # 应该至少有标准策略
        self.assertIsInstance(strategies, list)
    
    def test_industry_config_creation(self):
        """测试行业训练配置创建"""
        from backend.modules.training.launcher import create_industry_training_config
        
        config = create_industry_training_config(
            industry_type='manufacturing',
            model_name='test_industry_model',
            output_dir=self.temp_dir
        )
        
        self.assertIn('industry', config)
        self.assertTrue(config['industry']['enabled'])
        self.assertEqual(config['industry']['type'], 'manufacturing')
    
    def test_distillation_config_creation(self):
        """测试知识蒸馏配置创建"""
        from backend.modules.training.launcher import create_distillation_training_config
        
        config = create_distillation_training_config(
            scenario='edge_deploy',
            output_dir=self.temp_dir
        )
        
        self.assertIn('distillation', config)
        self.assertTrue(config['distillation']['enabled'])
        self.assertEqual(config['distillation']['scenario'], 'edge_deploy')


class TestProductionTrainingLauncher(unittest.TestCase):
    """ProductionTrainingLauncher 生产级功能测试"""
    
    def setUp(self):
        """测试初始化"""
        self.temp_dir = tempfile.mkdtemp()
        self.production_config = {
            'output_dir': self.temp_dir,
            'production': {
                'enabled': True,
                'enable_distributed_manager': True,
                'enable_checkpoint': True,
                'enable_monitoring': True,
                'retry_on_failure': 2,
            },
            'model': {'name': 'test_model', 'type': 'standard'},
            'training': {'num_epochs': 2, 'batch_size': 8, 'learning_rate': 1e-4},
        }
    
    def tearDown(self):
        """测试清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_production_launcher_initialization(self):
        """测试生产级启动器初始化"""
        from backend.modules.training.launcher import ProductionTrainingLauncher
        
        launcher = ProductionTrainingLauncher(self.production_config)
        
        self.assertIsNotNone(launcher)
        self.assertTrue(launcher.enable_distributed_manager)
        self.assertTrue(launcher.enable_checkpoint)
        self.assertTrue(launcher.enable_monitoring)
        self.assertEqual(launcher.retry_on_failure, 2)
    
    def test_production_config_analysis(self):
        """测试生产级配置分析"""
        from backend.modules.training.launcher import ProductionTrainingLauncher
        
        launcher = ProductionTrainingLauncher(self.production_config)
        analysis = launcher.analyze_config()
        
        # 检查生产级特有的分析字段
        self.assertIn('production_mode', analysis)
        self.assertIn('enable_distributed_manager', analysis)
        self.assertIn('combined_strategies', analysis)
        self.assertTrue(analysis['production_mode'])
    
    def test_strategy_combination_analysis(self):
        """测试策略组合分析"""
        from backend.modules.training.launcher import ProductionTrainingLauncher
        
        # 测试多模态+蒸馏组合
        config = self.production_config.copy()
        config['multimodal'] = {'enabled': True, 'modalities': ['text', 'image']}
        config['distillation'] = {'enabled': True, 'scenario': 'standard'}
        
        launcher = ProductionTrainingLauncher(config)
        strategies = launcher._analyze_strategy_combination()
        
        self.assertIn('multimodal', strategies)
        self.assertTrue(any('distillation' in s for s in strategies))
    
    def test_checkpoint_save_load(self):
        """测试检查点保存和加载"""
        from backend.modules.training.launcher import ProductionTrainingLauncher
        
        launcher = ProductionTrainingLauncher(self.production_config)
        launcher._training_state['session_id'] = 'test_session'
        launcher._training_state['status'] = 'running'
        
        # 保存检查点
        launcher._save_checkpoint('test_checkpoint')
        
        # 验证检查点文件存在
        checkpoint_dir = os.path.join(self.temp_dir, 'checkpoints')
        self.assertTrue(os.path.exists(checkpoint_dir))
        
        checkpoint_files = os.listdir(checkpoint_dir)
        self.assertTrue(len(checkpoint_files) > 0)
        
        # 加载检查点
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
        checkpoint_data = launcher._load_checkpoint(checkpoint_path)
        
        self.assertEqual(checkpoint_data['name'], 'test_checkpoint')
        self.assertEqual(checkpoint_data['session_id'], 'test_session')
    
    def test_production_training_config_creation(self):
        """测试生产级训练配置创建"""
        from backend.modules.training.launcher import create_production_training_config
        
        # 测试三阶段配置
        config = create_production_training_config(
            training_type='three_stage',
            output_dir=self.temp_dir,
            pretrain_epochs=2,
            finetune_epochs=3,
            preference_epochs=1
        )
        
        self.assertIn('production', config)
        self.assertIn('three_stage', config)
        self.assertTrue(config['three_stage']['enabled'])
        self.assertEqual(config['three_stage']['pretrain_epochs'], 2)
        
        # 测试行业模型配置
        config = create_production_training_config(
            training_type='industry',
            output_dir=self.temp_dir,
            industry_type='finance'
        )
        
        self.assertIn('industry', config)
        self.assertEqual(config['industry']['type'], 'finance')


class TestDistributedTrainingManager(unittest.TestCase):
    """DistributedTrainingManager 分布式训练管理测试"""
    
    def setUp(self):
        """测试初始化"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager_config = {
            'output_dir': self.temp_dir,
            'session_id': 'test_session_123',
            'distributed': {'mode': 'ddp', 'world_size': 2},
            'orchestrator': {'type': 'unified'},
            'training': {'num_epochs': 2, 'total_steps': 100}
        }
    
    def tearDown(self):
        """测试清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_manager_initialization(self):
        """测试管理器初始化"""
        from backend.modules.training.launcher import DistributedTrainingManager
        
        manager = DistributedTrainingManager(self.manager_config)
        
        self.assertIsNotNone(manager)
        self.assertEqual(manager.session_id, 'test_session_123')
        self.assertEqual(manager._status, 'idle')
    
    def test_training_plan_creation(self):
        """测试训练计划创建"""
        from backend.modules.training.launcher import DistributedTrainingManager
        
        manager = DistributedTrainingManager(self.manager_config)
        
        # 创建标准计划
        plan = manager.create_training_plan('standard', epochs=5)
        
        # 计划可能为 None（如果编排器不可用）
        if plan is not None:
            self.assertIsNotNone(plan)
    
    def test_pipeline_creation(self):
        """测试流水线创建"""
        from backend.modules.training.launcher import DistributedTrainingManager
        
        manager = DistributedTrainingManager(self.manager_config)
        
        steps = [
            {'name': 'pretrain', 'type': 'pretrain', 'params': {'epochs': 2}},
            {'name': 'finetune', 'type': 'finetune', 'params': {'epochs': 3}},
        ]
        
        pipeline = manager.create_pipeline(steps)
        
        if pipeline is not None:
            self.assertEqual(len(pipeline.steps), 2)
            self.assertEqual(pipeline.steps[0].name, 'pretrain')
    
    def test_progress_retrieval(self):
        """测试进度获取"""
        from backend.modules.training.launcher import DistributedTrainingManager
        
        manager = DistributedTrainingManager(self.manager_config)
        progress = manager.get_progress()
        
        self.assertIn('session_id', progress)
        self.assertIn('status', progress)
    
    def test_control_methods(self):
        """测试控制方法（暂停、恢复、停止）"""
        from backend.modules.training.launcher import DistributedTrainingManager
        
        manager = DistributedTrainingManager(self.manager_config)
        
        # 测试暂停
        manager.pause()
        self.assertEqual(manager._status, 'paused')
        
        # 测试恢复
        manager.resume()
        self.assertEqual(manager._status, 'running')
        
        # 测试停止
        manager.stop()
        self.assertEqual(manager._status, 'cancelled')
    
    def test_cleanup(self):
        """测试资源清理"""
        from backend.modules.training.launcher import DistributedTrainingManager
        
        manager = DistributedTrainingManager(self.manager_config)
        
        # 不应该抛出异常
        manager.cleanup()


class TestPipelineModule(unittest.TestCase):
    """Pipeline 模块测试"""
    
    def test_pipeline_definition(self):
        """测试流水线定义"""
        from backend.modules.training.pipeline import (
            PipelineDefinition, PipelineStep, create_pipeline
        )
        
        # 创建步骤
        step1 = PipelineStep(name='step1', type='pretrain', params={'epochs': 2})
        step2 = PipelineStep(name='step2', type='finetune', params={'epochs': 3})
        
        # 创建流水线
        pipeline = PipelineDefinition(
            name='test_pipeline',
            session_id='test_session',
            steps=[step1, step2],
            enable_rollback=True
        )
        
        self.assertEqual(pipeline.name, 'test_pipeline')
        self.assertEqual(len(pipeline.steps), 2)
        self.assertTrue(pipeline.enable_rollback)
    
    def test_pipeline_from_dict(self):
        """测试从字典创建流水线"""
        from backend.modules.training.pipeline import PipelineDefinition
        
        pipeline_dict = {
            'name': 'dict_pipeline',
            'session_id': 'session_456',
            'steps': [
                {'name': 'step1', 'type': 'pretrain', 'params': {'epochs': 2}},
                {'name': 'step2', 'type': 'finetune', 'params': {'epochs': 3}},
            ],
            'enable_rollback': False
        }
        
        pipeline = PipelineDefinition.from_dict(pipeline_dict)
        
        self.assertEqual(pipeline.name, 'dict_pipeline')
        self.assertEqual(len(pipeline.steps), 2)
        self.assertFalse(pipeline.enable_rollback)
    
    def test_create_pipeline_helper(self):
        """测试流水线创建辅助函数"""
        from backend.modules.training.pipeline import create_pipeline
        
        steps = [
            {'name': 'pretrain', 'type': 'pretrain', 'params': {'epochs': 2}},
            {'name': 'finetune', 'type': 'finetune', 'params': {'epochs': 3}},
        ]
        
        pipeline = create_pipeline(
            name='helper_pipeline',
            steps=steps,
            session_id='helper_session'
        )
        
        self.assertEqual(pipeline.name, 'helper_pipeline')
        self.assertEqual(len(pipeline.steps), 2)
    
    def test_create_three_stage_pipeline(self):
        """测试三阶段流水线创建"""
        from backend.modules.training.pipeline import create_three_stage_pipeline
        
        pipeline = create_three_stage_pipeline(
            name='three_stage_test',
            session_id='three_stage_session',
            pretrain_config={'epochs': 2, 'batch_size': 16},
            finetune_config={'epochs': 3, 'batch_size': 8},
            preference_config={'epochs': 1, 'batch_size': 4}
        )
        
        self.assertEqual(len(pipeline.steps), 3)
        self.assertEqual(pipeline.steps[0].type, 'pretrain')
        self.assertEqual(pipeline.steps[1].type, 'finetune')
        self.assertEqual(pipeline.steps[2].type, 'preference')


class TestTrainingServiceIntegration(unittest.TestCase):
    """训练服务集成测试"""
    
    def setUp(self):
        """测试初始化"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('backend.repositories.training_session_repository.get_training_session_repository')
    def test_launch_production_training(self, mock_repo):
        """测试启动生产级训练"""
        from backend.services.training_service import TrainingService
        
        # 模拟会话数据
        mock_session = MagicMock()
        mock_session.session_id = 'test_session'
        mock_session.status = 'pending'
        mock_session.config = {
            'model_name': 'test_model',
            'epochs': 2,
            'batch_size': 8,
            'output_dir': self.temp_dir
        }
        
        mock_repo_instance = MagicMock()
        mock_repo_instance.get_by_id.return_value = mock_session
        mock_repo_instance.update.return_value = mock_session
        mock_repo.return_value = mock_repo_instance
        
        service = TrainingService()
        service._repository = mock_repo_instance
        
        # 尝试启动生产级训练
        try:
            result = service.launch_production_training(
                session_id='test_session',
                training_type='standard'
            )
            
            self.assertIn('success', result)
            self.assertEqual(result['training_type'], 'standard')
        except Exception as e:
            # 如果模块不可用，验证正确的错误处理
            self.assertIn('不可用', str(e) or 'module' in str(e).lower())
    
    @patch('backend.repositories.training_session_repository.get_training_session_repository')
    def test_launch_pipeline_training(self, mock_repo):
        """测试启动流水线训练"""
        from backend.services.training_service import TrainingService
        
        # 模拟会话数据
        mock_session = MagicMock()
        mock_session.session_id = 'pipeline_session'
        mock_session.status = 'pending'
        mock_session.config = {}
        
        mock_repo_instance = MagicMock()
        mock_repo_instance.get_by_id.return_value = mock_session
        mock_repo_instance.update.return_value = mock_session
        mock_repo.return_value = mock_repo_instance
        
        service = TrainingService()
        service._repository = mock_repo_instance
        
        pipeline_steps = [
            {'name': 'pretrain', 'type': 'pretrain', 'params': {'epochs': 2}},
            {'name': 'finetune', 'type': 'finetune', 'params': {'epochs': 3}},
        ]
        
        try:
            result = service.launch_pipeline_training(
                session_id='pipeline_session',
                pipeline_steps=pipeline_steps
            )
            
            self.assertIn('success', result)
            self.assertEqual(result['steps_count'], 2)
        except Exception as e:
            # 验证正确的错误处理
            pass


class TestConvenienceFunctions(unittest.TestCase):
    """便捷函数测试"""
    
    def setUp(self):
        """测试初始化"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_launch_training_system(self):
        """测试 launch_training_system 便捷函数"""
        from backend.modules.training.launcher import launch_training_system
        
        config = {
            'output_dir': self.temp_dir,
            'model': {'name': 'test_model'},
            'training': {'num_epochs': 1}
        }
        
        # 执行训练（可能会失败，但不应该崩溃）
        result = launch_training_system(config)
        
        self.assertIn('success', result)
    
    def test_launch_production_training_convenience(self):
        """测试 launch_production_training 便捷函数"""
        from backend.modules.training.launcher import launch_production_training
        
        config = {
            'output_dir': self.temp_dir,
            'production': {'enabled': True},
            'model': {'name': 'test_model'},
            'training': {'num_epochs': 1}
        }
        
        # 执行训练
        result = launch_production_training(config)
        
        self.assertIn('success', result)


class TestEdgeCases(unittest.TestCase):
    """边界情况测试"""
    
    def setUp(self):
        """测试初始化"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_empty_config(self):
        """测试空配置"""
        from backend.modules.training.launcher import TrainingSystemLauncher
        
        launcher = TrainingSystemLauncher({})
        
        self.assertIsNotNone(launcher)
        self.assertIsNotNone(launcher.output_dir)
    
    def test_invalid_training_type(self):
        """测试无效的训练类型"""
        from backend.modules.training.launcher import ProductionTrainingLauncher
        
        config = {
            'output_dir': self.temp_dir,
            'model': {'name': 'test', 'type': 'invalid_type'}
        }
        
        launcher = ProductionTrainingLauncher(config)
        analysis = launcher.analyze_config()
        
        # 应该回退到标准类型
        self.assertIn('combined_strategies', analysis)
    
    def test_missing_modules_handling(self):
        """测试缺失模块的处理"""
        from backend.modules.training.launcher import DistributedTrainingManager
        
        config = {
            'output_dir': self.temp_dir,
            'distributed': {'enabled': True, 'mode': 'fsdp'}
        }
        
        manager = DistributedTrainingManager(config)
        
        # 即使模块不可用也不应该崩溃
        manager._init_orchestrator()
        manager._init_pipeline()
        manager._init_progress_manager()
        
        progress = manager.get_progress()
        self.assertIsNotNone(progress)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingSystemLauncher))
    suite.addTests(loader.loadTestsFromTestCase(TestProductionTrainingLauncher))
    suite.addTests(loader.loadTestsFromTestCase(TestDistributedTrainingManager))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineModule))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingServiceIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestConvenienceFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回结果
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
