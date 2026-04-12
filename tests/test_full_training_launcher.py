"""完整训练启动器测试

测试训练启动器的所有功能，包括不同训练模式的选择和启动。
"""

import sys
import os
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_standard_training():
    """测试标准训练模式"""
    print("开始测试标准训练模式...")
    
    try:
        # 导入必要的模块
        from backend_new.modules.training.launcher.training_launcher import TrainingSystemLauncher
        
        # 创建标准训练配置
        config = {
            'model': {
                'name': 'test_standard_model',
                'type': 'standard',
                'task_type': 'causal_lm'
            },
            'data': {
                'type': 'text',
                'train_path': './data/train',
                'val_path': './data/val'
            },
            'training': {
                'mode': 'standard',
                'num_epochs': 1,
                'batch_size': 8,
                'learning_rate': 2e-5,
                'fp16': True
            },
            'monitoring': {
                'logging_steps': 10,
                'save_steps': 100,
                'eval_steps': 50
            },
            'distributed': {
                'enabled': False
            },
            'distillation': {
                'enabled': False
            },
            'multimodal': {
                'enabled': False
            },
            'three_stage': {
                'enabled': False
            },
            'scenario': {
                'enabled': False
            }
        }
        
        # 创建启动器
        launcher = TrainingSystemLauncher(config)
        
        # 分析配置
        analysis = launcher.analyze_config()
        
        # 选择训练器
        trainer = launcher.select_trainer(analysis)
        
        print(f"✓ 标准训练模式测试成功: {type(trainer).__name__}")
        return True
        
    except Exception as e:
        print(f"❌ 标准训练模式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multimodal_training():
    """测试多模态训练模式"""
    print("开始测试多模态训练模式...")
    
    try:
        # 导入必要的模块
        from backend_new.modules.training.launcher.training_launcher import TrainingSystemLauncher
        
        # 创建多模态训练配置
        config = {
            'model': {
                'name': 'test_multimodal_model',
                'type': 'multimodal'
            },
            'data': {
                'type': 'multimodal',
                'train_path': './data/multimodal/train',
                'val_path': './data/multimodal/val'
            },
            'training': {
                'mode': 'multimodal',
                'num_epochs': 1,
                'batch_size': 4,
                'learning_rate': 2e-5
            },
            'multimodal': {
                'enabled': True,
                'modalities': ['text', 'image'],
                'text_model_name': 'bert-base-uncased',
                'image_model_name': 'resnet50'
            },
            'distributed': {
                'enabled': False
            },
            'distillation': {
                'enabled': False
            },
            'three_stage': {
                'enabled': False
            },
            'scenario': {
                'enabled': False
            }
        }
        
        # 创建启动器
        launcher = TrainingSystemLauncher(config)
        
        # 分析配置
        analysis = launcher.analyze_config()
        
        # 选择训练器
        trainer = launcher.select_trainer(analysis)
        
        print(f"✓ 多模态训练模式测试成功: {type(trainer).__name__}")
        return True
        
    except Exception as e:
        print(f"❌ 多模态训练模式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_distillation_training():
    """测试知识蒸馏训练模式"""
    print("开始测试知识蒸馏训练模式...")
    
    try:
        # 导入必要的模块
        from backend_new.modules.training.launcher.training_launcher import TrainingSystemLauncher
        
        # 创建知识蒸馏训练配置
        config = {
            'model': {
                'name': 'test_distill_model'
            },
            'data': {
                'type': 'text',
                'train_path': './data/train'
            },
            'training': {
                'mode': 'distillation',
                'num_epochs': 1
            },
            'distillation': {
                'enabled': True,
                'teacher_model_path': './models/teacher',
                'student_model_path': './models/student'
            },
            'distributed': {
                'enabled': False
            },
            'multimodal': {
                'enabled': False
            },
            'three_stage': {
                'enabled': False
            },
            'scenario': {
                'enabled': False
            }
        }
        
        # 创建启动器
        launcher = TrainingSystemLauncher(config)
        
        # 分析配置
        analysis = launcher.analyze_config()
        
        # 选择训练器
        trainer = launcher.select_trainer(analysis)
        
        print(f"✓ 知识蒸馏训练模式测试成功: {type(trainer).__name__}")
        return True
        
    except Exception as e:
        print(f"❌ 知识蒸馏训练模式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scenario_training():
    """测试场景化训练模式"""
    print("开始测试场景化训练模式...")
    
    try:
        # 导入必要的模块
        from backend_new.modules.training.launcher.training_launcher import TrainingSystemLauncher
        
        # 创建场景化训练配置
        config = {
            'model': {
                'name': 'test_scenario_model'
            },
            'data': {
                'type': 'text',
                'train_path': './data/train'
            },
            'training': {
                'mode': 'scenario',
                'num_epochs': 1
            },
            'scenario': {
                'enabled': True,
                'type': 'basic_model'
            },
            'distributed': {
                'enabled': False
            },
            'distillation': {
                'enabled': False
            },
            'multimodal': {
                'enabled': False
            },
            'three_stage': {
                'enabled': False
            }
        }
        
        # 创建启动器
        launcher = TrainingSystemLauncher(config)
        
        # 分析配置
        analysis = launcher.analyze_config()
        
        # 选择训练器
        trainer = launcher.select_trainer(analysis)
        
        print(f"✓ 场景化训练模式测试成功: {type(trainer).__name__}")
        return True
        
    except Exception as e:
        print(f"❌ 场景化训练模式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始完整训练启动器测试...")
    
    tests = [
        test_standard_training,
        test_multimodal_training,
        test_distillation_training,
        test_scenario_training
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"测试执行失败: {e}")
            failed += 1
        print()  # 空行分隔
    
    print(f"测试完成! 通过: {passed}, 失败: {failed}")
    
    if failed == 0:
        print("🎉 所有测试都通过了!")
        return True
    else:
        print("❌ 部分测试失败。")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)