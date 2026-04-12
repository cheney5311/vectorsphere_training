"""训练启动器测试

测试训练启动器是否能根据配置正确选择和启动训练模式。
"""

import sys
import os
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_training_launcher():
    """测试训练启动器"""
    print("开始测试训练启动器...")
    
    try:
        # 导入必要的模块
        from backend_new.modules.training.launcher.training_launcher import TrainingSystemLauncher
        
        print("✓ 模块导入成功")
        
        # 创建测试配置
        test_config = {
            'model': {
                'name': 'test_model',
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
        launcher = TrainingSystemLauncher(test_config)
        print("✓ 训练启动器创建成功")
        
        # 分析配置
        analysis = launcher.analyze_config()
        print("✓ 配置分析成功")
        
        # 选择训练器
        trainer = launcher.select_trainer(analysis)
        print(f"✓ 训练器选择成功: {type(trainer).__name__}")
        
        print("\n🎉 训练启动器测试完成！")
        return True
        
    except Exception as e:
        print(f"\n❌ 训练启动器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_training_launcher()
    if success:
        print("\n✅ 训练启动器测试完成！")
    else:
        print("\n❌ 测试失败，请检查问题。")