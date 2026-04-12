"""命令管理器

负责处理CLI命令的注册、解析和执行。
"""

import argparse
import sys
import os
from typing import Dict, List, Type, Optional
from datetime import datetime

from .base_command import BaseCommand
from .serve_command import ServeCommand
from .info_command import InfoCommand
from .train_command import TrainCommand
from .test_command import TestCommand


class CommandManager:
    """CLI命令管理器"""
    
    VERSION = "1.0.0"
    
    def __init__(self, description: str = "VectorSphere 智能训练平台"):
        self.description = description
        self.commands: Dict[str, BaseCommand] = {}
        self.parser: Optional[argparse.ArgumentParser] = None
        self.subparsers = None
        
        # 注册默认命令
        self._register_default_commands()
    
    def _register_default_commands(self):
        """注册默认命令"""
        default_commands = [
            ServeCommand(),
            InfoCommand(), 
            TrainCommand(),
            TestCommand()
        ]
        
        for command in default_commands:
            self.register_command(command)
    
    def register_command(self, command: BaseCommand):
        """注册命令"""
        if not isinstance(command, BaseCommand):
            raise TypeError("命令必须继承自BaseCommand")
            
        self.commands[command.name] = command
    
    def unregister_command(self, command_name: str):
        """注销命令"""
        if command_name in self.commands:
            del self.commands[command_name]
    
    def get_command(self, command_name: str) -> Optional[BaseCommand]:
        """根据名称获取命令"""
        return self.commands.get(command_name)
    
    def list_commands(self) -> List[str]:
        """列出所有已注册的命令名称"""
        return list(self.commands.keys())
    
    def setup_parser(self) -> argparse.ArgumentParser:
        """设置参数解析器，包含所有已注册的命令"""
        self.parser = argparse.ArgumentParser(
            description=self.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_epilog()
        )
        
        # 添加全局参数
        self.parser.add_argument(
            '--version', action='version', 
            version=f'VectorSphere {self.VERSION}'
        )
        self.parser.add_argument(
            '--quiet', '-q', action='store_true',
            help='安静模式，减少输出'
        )
        
        # 为命令创建子解析器
        self.subparsers = self.parser.add_subparsers(
            dest='command', 
            help='可用命令',
            metavar='COMMAND'
        )
        
        # 将每个命令添加为子解析器
        for command_name, command in self.commands.items():
            cmd_parser = self.subparsers.add_parser(
                command_name,
                help=command.help,
                description=command.help,
                formatter_class=argparse.RawDescriptionHelpFormatter
            )
            
            # 让命令添加其特定参数
            command.add_arguments(cmd_parser)
        
        return self.parser
    
    def _get_epilog(self) -> str:
        """获取帮助信息的结尾部分"""
        return """
📝 使用示例:
  python app.py serve --host 0.0.0.0 --port 8080
  python app.py train --model llama2 --training-mode three-stage --sft-enabled
  python app.py test --category services --verbose
  python app.py info --check-health

💡 获取命令帮助:
  python app.py <command> --help
  
📚 文档: https://vectorsphere.io/docs
"""
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """解析命令行参数"""
        if self.parser is None:
            self.setup_parser()
            
        return self.parser.parse_args(args)
    
    def execute_command(self, args: argparse.Namespace) -> int:
        """执行指定的命令"""
        if not args.command:
            # 未指定命令，显示帮助
            self.show_usage()
            return 0
        
        command = self.get_command(args.command)
        if not command:
            print(f"❌ 未知命令: {args.command}")
            self.show_usage()
            return 1
        
        # 验证参数
        if not command.validate_args(args):
            return 1
        
        # 设置
        if not command.setup(args):
            return 1
        
        try:
            # 执行命令
            result = command.execute(args)
            return result
        except KeyboardInterrupt:
            print("\n操作已取消")
            return 1
        except Exception as e:
            print(f"执行失败: {e}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            # 清理
            command.cleanup(args)
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """解析参数并执行命令"""
        try:
            parsed_args = self.parse_args(args)
            return self.execute_command(parsed_args)
        except KeyboardInterrupt:
            print("\n👋 操作已取消")
            return 1
        except SystemExit as e:
            return e.code if isinstance(e.code, int) else 0
        except Exception as e:
            print(f"❌ 执行失败: {e}")
            return 1
    
    def show_usage(self):
        """显示使用说明"""
        self._show_banner()
        
        print(f"📋 可用命令:")
        print("-" * 50)
        
        for command_name, command in self.commands.items():
            print(f"  {command_name:<12} {command.help}")
        
        print()
        print(f"📝 使用示例:")
        print("-" * 50)
        print("  # 启动服务")
        print("  python app.py serve --host 0.0.0.0 --port 8080")
        print()
        print("  # 启动带WebSocket的服务")
        print("  python app.py serve --websocket --debug")
        print()
        print("  # 标准训练")
        print("  python app.py train --model gpt2 --epochs 10")
        print()
        print("  # 三阶段训练")
        print("  python app.py train --model llama2 --training-mode three-stage \\")
        print("      --sft-enabled --sft-epochs 5 --dpo-enabled --dpo-epochs 3")
        print()
        print("  # 流水线训练")
        print("  python app.py train --model bert --training-mode pipeline \\")
        print("      --pipeline-steps pretrain finetune evaluation")
        print()
        print("  # 运行测试")
        print("  python app.py test --category services --verbose")
        print()
        print("  # 查看平台信息")
        print("  python app.py info --format json --verbose")
        print()
        print("  # 健康检查")
        print("  python app.py info --check-health")
        print()
        print(f"💡 获取命令帮助:")
        print("-" * 50)
        print("  python app.py <command> --help")
        print()
    
    def _show_banner(self):
        """显示横幅"""
        print()
        print("=" * 60)
        print("🚀 VectorSphere 智能训练平台")
        print("=" * 60)
        print(f"📅 版本: {self.VERSION}")
        print(f"📅 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def add_custom_command(self, command_class: Type[BaseCommand], *args, **kwargs):
        """添加自定义命令类"""
        command_instance = command_class(*args, **kwargs)
        self.register_command(command_instance)
    
    def get_commands_info(self) -> Dict[str, Dict]:
        """获取所有命令的信息"""
        return {
            name: {
                'name': cmd.name,
                'help': cmd.help
            }
            for name, cmd in self.commands.items()
        }


def create_command_manager() -> CommandManager:
    """创建命令管理器实例"""
    return CommandManager()


def main(args: Optional[List[str]] = None) -> int:
    """主入口函数"""
    manager = create_command_manager()
    return manager.run(args)


if __name__ == '__main__':
    sys.exit(main())
