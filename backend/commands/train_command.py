# -*- coding: utf-8 -*-
"""
Train command implementation

命令行训练模型，支持多种训练模式：标准训练、分布式训练、三阶段训练等。
"""

import argparse
import json
import os
import random
import sys
import threading
import time
import uuid
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_command import BaseCommand


class TrainCommand(BaseCommand):
    """Command to train models"""
    
    def __init__(self):
        super().__init__()
        self._training_session = None
        self._stop_event = threading.Event()
    
    def get_command_name(self) -> str:
        return 'train'
    
    def get_command_help(self) -> str:
        return '命令行训练模型'
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add train command specific arguments"""
        # 基础配置
        parser.add_argument('--config', type=str, help='配置文件路径')
        parser.add_argument('--data', type=str, help='训练数据路径')
        parser.add_argument('--model', type=str, required=True, help='模型名称')
        parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
        parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
        parser.add_argument('--learning-rate', type=float, default=0.001, help='学习率')
        parser.add_argument('--output-dir', type=str, default='./models', help='模型输出目录')
        parser.add_argument('--dry-run', action='store_true', help='仅模拟训练过程')
        parser.add_argument('--user-id', type=str, default='cli_user', help='用户ID')
        parser.add_argument('--tenant-id', type=str, help='租户ID（可选）')
        parser.add_argument('--session-name', type=str, help='训练会话名称')
        
        # 添加训练模式选择参数
        parser.add_argument('--training-mode', type=str, choices=[
            'standard',      # 标准训练
            'distributed',   # 分布式训练
            'multimodal',    # 多模态训练
            'distillation',  # 知识蒸馏
            'three-stage',   # 三阶段训练
            'pipeline',      # 流水线训练
            'scenario'       # 场景化训练
        ], default='standard', help='训练模式')
        
        # 添加场景化训练的特定参数
        parser.add_argument('--scenario-type', type=str, choices=[
            'basic_model',         # 基础模型训练
            'scheduled_task',      # 定时任务训练
            'advanced_model',      # 高级模型训练
            'research_experiment', # 研究实验训练
            'production_finetune'  # 生产微调训练
        ], default='basic_model', help='场景类型（仅在训练模式为scenario时有效）')
        
        # 添加分布式训练参数
        parser.add_argument('--world-size', type=int, default=1, help='分布式训练的节点数')
        parser.add_argument('--master-port', type=str, default='12355', help='分布式训练的主节点端口')
        
        # 添加多模态训练参数
        parser.add_argument('--modality-types', type=str, nargs='+', 
                          choices=['text', 'image', 'audio'], 
                          default=['text'], 
                          help='多模态类型（仅在训练模式为multimodal时有效）')
        
        # 添加知识蒸馏参数
        parser.add_argument('--teacher-model', type=str, help='教师模型路径（仅在训练模式为distillation时有效）')
        
        # 添加三阶段训练参数
        parser.add_argument('--three-stage-config', type=str, help='三阶段训练配置文件路径')
        parser.add_argument('--pt-enabled', action='store_true', help='启用预训练阶段')
        parser.add_argument('--pt-epochs', type=int, default=3, help='预训练轮数')
        parser.add_argument('--sft-enabled', action='store_true', help='启用监督微调阶段')
        parser.add_argument('--sft-epochs', type=int, default=3, help='监督微调轮数')
        parser.add_argument('--dpo-enabled', action='store_true', help='启用偏好优化阶段')
        parser.add_argument('--dpo-epochs', type=int, default=3, help='偏好优化轮数')
        parser.add_argument('--dpo-beta', type=float, default=0.1, help='DPO beta参数')
        
        # 流水线训练参数
        parser.add_argument('--pipeline-config', type=str, help='流水线配置文件路径')
        parser.add_argument('--pipeline-steps', type=str, nargs='+', help='流水线步骤')
        
        # 进度和日志
        parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
        parser.add_argument('--progress-interval', type=int, default=10, help='进度更新间隔（秒）')
        parser.add_argument('--save-interval', type=int, default=1000, help='模型保存间隔（步）')
    
    def validate_args(self, args: argparse.Namespace) -> bool:
        """Validate train command arguments"""
        if not args.model:
            print("错误: 请指定模型名称 --model")
            return False
            
        if args.epochs <= 0:
            print("错误: 训练轮数必须大于0")
            return False
            
        if args.batch_size <= 0:
            print("错误: 批次大小必须大于0")
            return False
            
        if args.learning_rate <= 0:
            print("错误: 学习率必须大于0")
            return False
            
        if args.data and not os.path.exists(args.data):
            print(f"警告: 训练数据文件不存在: {args.data}")
            
        # 验证特定模式的参数
        if args.training_mode == 'distillation' and not args.teacher_model:
            print("错误: 知识蒸馏模式需要指定教师模型 --teacher-model")
            return False
            
        if args.training_mode == 'distributed' and args.world_size <= 1:
            print("警告: 分布式训练的节点数应大于1")
            
        return True
    
    def setup(self, args: argparse.Namespace) -> bool:
        """Setup before training"""
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        return True
    
    def execute(self, args: argparse.Namespace) -> int:
        """Execute the train command"""
        try:
            self._show_training_info(args)
            
            if args.dry_run:
                print("模拟训练模式 (Dry Run)")
                return self._simulate_training(args)
            else:
                return self._real_training(args)
                
        except KeyboardInterrupt:
            print("\n训练被用户中断")
            self._stop_event.set()
            return 1
        except Exception as e:
            print(f"训练失败: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    def _show_training_info(self, args: argparse.Namespace):
        """显示训练信息"""
        print()
        print("=" * 60)
        print(f"开始训练模型: {args.model}")
        print("=" * 60)
        print(f"训练配置:")
        print(f"  • 模型名称: {args.model}")
        print(f"  • 训练轮数: {args.epochs}")
        print(f"  • 批次大小: {args.batch_size}")
        print(f"  • 学习率: {args.learning_rate}")
        print(f"  • 输出目录: {args.output_dir}")
        print(f"  • 训练模式: {args.training_mode}")
        
        if args.training_mode == 'scenario':
            print(f"  • 场景类型: {args.scenario_type}")
        elif args.training_mode == 'distributed':
            print(f"  • 节点数: {args.world_size}")
            print(f"  • 主节点端口: {args.master_port}")
        elif args.training_mode == 'multimodal':
            print(f"  • 模态类型: {', '.join(args.modality_types)}")
        elif args.training_mode == 'distillation':
            print(f"  • 教师模型: {args.teacher_model}")
        elif args.training_mode == 'three-stage':
            print(f"  • 预训练: {'启用' if args.pt_enabled else '禁用'}")
            print(f"  • 监督微调: {'启用' if args.sft_enabled else '禁用'}")
            print(f"  • 偏好优化: {'启用' if args.dpo_enabled else '禁用'}")
        
        print("-" * 60)
    
    def _simulate_training(self, args: argparse.Namespace) -> int:
        """Simulate training process"""
        print("\n准备训练数据...")
        time.sleep(1)
        
        if args.data and os.path.exists(args.data):
            print(f"加载数据文件: {args.data}")
        else:
            print("使用默认示例数据")
        
        print(f"\n开始模拟训练 ({args.epochs} 轮)...")
        
        for epoch in range(1, args.epochs + 1):
            if self._stop_event.is_set():
                print("\n训练已停止")
                return 1
                
            time.sleep(0.5)
            loss = random.uniform(0.1, 1.0) / epoch
            acc = random.uniform(0.7, 0.95) * epoch / args.epochs
            
            # 显示进度条
            progress = epoch / args.epochs * 100
            bar_length = 30
            filled_length = int(bar_length * epoch // args.epochs)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            print(f"  Epoch {epoch:2d}/{args.epochs} [{bar}] {progress:5.1f}% - "
                  f"Loss: {loss:.4f} - Accuracy: {acc:.4f}")
        
        model_path = os.path.join(args.output_dir, f"{args.model}_model.pth")
        print(f"\n模型训练完成!")
        print(f"模型保存路径: {model_path}")
        
        return 0
    
    def _real_training(self, args: argparse.Namespace) -> int:
        """Real training implementation"""
        print("\n启动真实训练...")
        
        # 根据训练模式选择不同的训练方法
        if args.training_mode == 'three-stage':
            return self._run_three_stage_training(args)
        elif args.training_mode == 'pipeline':
            return self._run_pipeline_training(args)
        elif args.training_mode == 'distributed':
            return self._run_distributed_training(args)
        else:
            return self._run_standard_training(args)
    
    def _run_three_stage_training(self, args: argparse.Namespace) -> int:
        """运行三阶段训练"""
        try:
            from backend.services.three_stage_training_service import get_three_stage_training_service
            
            service = get_three_stage_training_service()
            tenant_id = args.tenant_id or str(uuid.uuid4())
            user_id = args.user_id
            
            # 构建配置
            config = self._build_three_stage_config(args)
            
            print(f"\n创建三阶段训练会话...")
            
            # 创建并启动训练
            result = service.create_and_start(
                name=args.session_name or f"CLI训练: {args.model}",
                model_name=args.model,
                config=config,
                tenant_id=tenant_id,
                user_id=user_id,
                description=f"命令行启动的三阶段训练: {args.model}"
            )
            
            session_id = result.get('session_id')
            print(f"训练会话已创建: {session_id}")
            
            # 监控训练进度
            return self._monitor_training_progress(service, session_id, tenant_id, args)
            
        except Exception as e:
            print(f"三阶段训练失败: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    def _run_pipeline_training(self, args: argparse.Namespace) -> int:
        """运行流水线训练"""
        try:
            from backend.services.pipeline_service import get_pipeline_service
            
            service = get_pipeline_service()
            tenant_id = args.tenant_id or str(uuid.uuid4())
            user_id = args.user_id
            
            # 构建步骤配置
            steps_config = self._build_pipeline_steps(args)
            
            print(f"\n创建训练流水线...")
            
            # 创建流水线
            pipeline = service.create_pipeline(
                name=args.session_name or f"CLI流水线: {args.model}",
                steps_config=steps_config,
                tenant_id=tenant_id,
                user_id=user_id,
                model_name=args.model,
                description=f"命令行启动的训练流水线: {args.model}"
            )
            
            pipeline_id = pipeline.get('pipeline_id')
            print(f"流水线已创建: {pipeline_id}")
            
            # 启动流水线
            result = service.start_pipeline(
                pipeline_id=pipeline_id,
                tenant_id=tenant_id,
                user_id=user_id
            )
            
            if result.get('success'):
                print(f"流水线已启动: {result.get('execution_id')}")
                return self._monitor_pipeline_progress(service, result.get('execution_id'), tenant_id, args)
            else:
                print(f"流水线启动失败: {result.get('error')}")
                return 1
            
        except Exception as e:
            print(f"流水线训练失败: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    def _run_distributed_training(self, args: argparse.Namespace) -> int:
        """运行分布式训练"""
        try:
            from backend.modules.training.launcher.training_launcher import launch_training_system
            
            config = self._build_training_config(args)
            config['distributed'] = {
                'enabled': True,
                'world_size': args.world_size,
                'master_port': args.master_port
            }
            
            print(f"\n启动分布式训练 (节点数: {args.world_size})...")
            
            result = launch_training_system(config)
            
            if result.get('success'):
                print("分布式训练完成!")
                return 0
            else:
                print(f"分布式训练失败: {result.get('error')}")
                return 1
                
        except Exception as e:
            print(f"分布式训练失败: {e}")
            return 1
    
    def _run_standard_training(self, args: argparse.Namespace) -> int:
        """运行标准训练"""
        try:
            from backend.modules.training.launcher.training_launcher import launch_training_system
            
            config = self._build_training_config(args)
            
            print(f"\n启动标准训练...")
            
            result = launch_training_system(config)
            
            if result.get('success'):
                print("训练完成!")
                if 'message' in result:
                    print(f"{result['message']}")
                return 0
            else:
                print(f"训练失败: {result.get('error')}")
                return 1
                
        except ImportError:
            print("训练模块不可用，使用模拟训练")
            return self._simulate_training(args)
        except Exception as e:
            print(f"标准训练失败: {e}")
            return 1
    
    def _build_three_stage_config(self, args: argparse.Namespace) -> Dict[str, Any]:
        """构建三阶段训练配置"""
        config = {
            'model_name': args.model,
            'base_model_path': args.model,
            'output_dir': args.output_dir,
            'pass_model_between_stages': True,
            'stages': {}
        }
        
        # 从配置文件加载
        if args.three_stage_config and os.path.exists(args.three_stage_config):
            with open(args.three_stage_config, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                config.update(file_config)
        else:
            # 使用命令行参数
            if args.pt_enabled:
                config['stages']['pt'] = {
                    'enabled': True,
                    'epochs': args.pt_epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'data_path': args.data or './data/pt'
                }
            
            if args.sft_enabled:
                config['stages']['sft'] = {
                    'enabled': True,
                    'epochs': args.sft_epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'data_path': args.data or './data/sft'
                }
            
            if args.dpo_enabled:
                config['stages']['dpo'] = {
                    'enabled': True,
                    'epochs': args.dpo_epochs,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'beta': args.dpo_beta,
                    'data_path': args.data or './data/dpo'
                }
        
        return config
    
    def _build_pipeline_steps(self, args: argparse.Namespace) -> list:
        """构建流水线步骤配置"""
        # 从配置文件加载
        if args.pipeline_config and os.path.exists(args.pipeline_config):
            with open(args.pipeline_config, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('steps', [])
        
        # 从命令行参数构建
        if args.pipeline_steps:
            steps = []
            for i, step_type in enumerate(args.pipeline_steps):
                steps.append({
                    'name': f'step_{i+1}_{step_type}',
                    'type': step_type,
                    'params': {
                        'epochs': args.epochs,
                        'batch_size': args.batch_size,
                        'learning_rate': args.learning_rate
                    },
                    'on_fail': 'rollback'
                })
            return steps
        
        # 默认步骤
        return [
            {'name': 'data_processing', 'type': 'data_processing', 'on_fail': 'stop'},
            {'name': 'training', 'type': 'finetune', 'params': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate
            }, 'on_fail': 'rollback'},
            {'name': 'evaluation', 'type': 'evaluation', 'on_fail': 'continue'}
        ]
    
    def _build_training_config(self, args: argparse.Namespace) -> Dict[str, Any]:
        """构建训练配置"""
        config = {
            'model': {
                'name': args.model,
                'type': 'standard'
            },
            'training': {
                'num_epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate
            },
            'data': {
                'train_path': args.data or './data/train',
                'val_path': None,
                'test_path': None
            },
            'output_dir': args.output_dir,
            'monitoring': {
                'logging_steps': 100,
                'save_steps': args.save_interval,
                'eval_steps': 500
            }
        }
        
        # 从配置文件加载
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                config = self._merge_config(file_config, config)
        
        # 根据训练模式添加特定配置
        if args.training_mode == 'multimodal':
            config['multimodal'] = {
                'enabled': True,
                'modality_types': args.modality_types
            }
        elif args.training_mode == 'distillation':
            config['distillation'] = {
                'enabled': True,
                'teacher_model_path': args.teacher_model
            }
        elif args.training_mode == 'scenario':
            config['scenario'] = {
                'enabled': True,
                'type': args.scenario_type
            }
        
        return config
    
    def _merge_config(self, file_config: Dict[str, Any], cli_config: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置"""
        merged = file_config.copy()
        
        def merge_dict(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    merge_dict(target[key], value)
                else:
                    target[key] = value
            return target
        
        return merge_dict(merged, cli_config)
    
    def _monitor_training_progress(self, service, session_id: str, tenant_id: str, 
                                   args: argparse.Namespace) -> int:
        """监控训练进度"""
        print(f"\n监控训练进度 (会话: {session_id})...")
        print("   按 Ctrl+C 可停止监控（训练将继续在后台运行）")
        print()
        
        last_progress = -1
        
        try:
            while not self._stop_event.is_set():
                progress_info = service.get_progress(session_id, tenant_id)
                
                status = progress_info.get('status', 'unknown')
                progress = progress_info.get('progress', 0)
                current_stage = progress_info.get('current_stage', '-')
                
                # 只在进度变化时更新显示
                if progress != last_progress:
                    bar_length = 30
                    filled = int(bar_length * progress / 100)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"\r  [{bar}] {progress:5.1f}% | 状态: {status} | 阶段: {current_stage}", end="")
                    last_progress = progress
                
                if status in ['completed', 'failed', 'stopped', 'error']:
                    print()
                    if status == 'completed':
                        print("\n训练完成!")
                        return 0
                    else:
                        print(f"\n训练结束: {status}")
                        return 1
                
                time.sleep(args.progress_interval)
                
        except KeyboardInterrupt:
            print("\n\n停止监控（训练继续在后台运行）")
            print(f"   会话ID: {session_id}")
            return 0
        
        return 0
    
    def _monitor_pipeline_progress(self, service, execution_id: str, tenant_id: str,
                                   args: argparse.Namespace) -> int:
        """监控流水线进度"""
        print(f"\n监控流水线进度 (执行: {execution_id})...")
        print()
        
        try:
            while not self._stop_event.is_set():
                status_info = service.get_execution_status(execution_id, tenant_id)
                
                if not status_info:
                    print("无法获取执行状态")
                    time.sleep(args.progress_interval)
                    continue
                
                status = status_info.get('status', 'unknown')
                progress = status_info.get('progress', 0)
                current_step = status_info.get('current_step', 0)
                total_steps = status_info.get('total_steps', 0)
                
                bar_length = 30
                filled = int(bar_length * progress / 100) if progress else 0
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"\r  [{bar}] {progress:5.1f}% | 步骤: {current_step}/{total_steps} | 状态: {status}", end="")
                
                if status in ['completed', 'failed', 'cancelled']:
                    print()
                    if status == 'completed':
                        print("\n流水线执行完成!")
                        return 0
                    else:
                        print(f"\n流水线结束: {status}")
                        return 1
                
                time.sleep(args.progress_interval)
                
        except KeyboardInterrupt:
            print("\n\n停止监控")
            return 0
        
        return 0
    
    def cleanup(self, args: argparse.Namespace) -> None:
        """Cleanup after training"""
        self._stop_event.set()
