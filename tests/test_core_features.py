#!/usr/bin/env python3
"""核心功能测试脚本"""

import sys
import os
import json
import time
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_download_api():
    """测试模型下载API导入"""
    print("=== 测试模型下载API导入 ===")
    
    try:
        from backend.modules.model.api.model_download_api import model_download_bp
        print("✓ 模型下载API导入成功")
    except Exception as e:
        print(f"✗ 模型下载API导入失败: {e}")


def test_three_stage_training():
    """测试三步训练流程"""
    print("\n=== 测试三步训练流程 ===")
    
    try:
        from backend.modules.training.three_stage.three_stage_trainer import ThreeStageTrainer, ThreeStageConfig, StageConfig
        
        # 创建三步训练配置
        config = ThreeStageConfig(
            base_model_path="gpt2",
            output_dir="./test_output",
            pretrain=StageConfig(
                enabled=True,
                epochs=1,
                learning_rate=1e-4
            ),
            finetune=StageConfig(
                enabled=True,
                epochs=1,
                learning_rate=2e-5
            ),
            preference=StageConfig(
                enabled=True,
                epochs=1,
                learning_rate=1e-5
            )
        )
        
        # 创建训练器
        trainer = ThreeStageTrainer(config)
        print("✓ 三步训练器创建成功")
        
        print("✓ 三步训练流程测试完成")
        
    except Exception as e:
        print(f"✗ 三步训练测试失败: {e}")


def test_training_progress_monitoring():
    """测试训练进度监控"""
    print("\n=== 测试训练进度监控 ===")
    
    try:
        from backend.modules.training.progress.progress_manager import TrainingProgressManager, TrainingProgress
        
        # 创建进度管理器
        progress_manager = TrainingProgressManager()
        print("✓ 训练进度管理器创建成功")
        
        # 创建进度跟踪器
        session_id = "test_session_001"
        progress = progress_manager.create_progress_tracker(session_id, total_steps=100, total_epochs=10)
        print("✓ 进度跟踪器创建成功")
        
        # 更新进度
        progress_manager.update_progress(
            session_id,
            current_step=50,
            current_epoch=5,
            progress=50.0,
            train_loss=0.5,
            train_accuracy=0.85
        )
        print("✓ 进度更新测试完成")
        
        # 获取统计信息
        stats = progress_manager.get_training_statistics(session_id)
        print("✓ 训练统计信息获取成功")
        print(f"  当前进度: {stats['progress']:.1f}%")
        print(f"  当前轮次: {stats['current_epoch']}")
        
    except Exception as e:
        print(f"✗ 训练进度监控测试失败: {e}")


def test_robustness_features():
    """测试健壮性功能"""
    print("\n=== 测试健壮性功能 ===")
    
    try:
        from backend.modules.training.services.training_execution_service import TrainingExecutionService
        
        # 创建训练执行服务
        execution_service = TrainingExecutionService()
        print("✓ 训练执行服务创建成功")
        
        # 测试配置验证
        valid_config = {
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 0.001
        }
        
        try:
            execution_service._validate_training_config(valid_config)
            print("✓ 训练配置验证通过")
        except Exception as e:
            print(f"✗ 训练配置验证失败: {e}")
            
        # 测试异常配置
        invalid_config = {
            "epochs": -1,  # 无效值
            "batch_size": 8,
            "learning_rate": 0.001
        }
        
        try:
            execution_service._validate_training_config(invalid_config)
            print("✗ 异常配置验证应该失败但通过了")
        except Exception as e:
            print("✓ 异常配置验证正确失败")
            
    except Exception as e:
        print(f"✗ 健壮性功能测试失败: {e}")


def main():
    """主测试函数"""
    print("开始测试VectorSphere训练平台核心功能...")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 执行各项测试
    test_model_download_api()
    test_three_stage_training()
    test_training_progress_monitoring()
    test_robustness_features()
    
    print("\n=== 核心功能测试完成 ===")
    print("所有核心功能模块测试已完成，请检查以上结果。")


if __name__ == "__main__":
    main()