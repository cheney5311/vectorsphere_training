#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VectorSphere 启动流程管理器

实现启动前的usage显示、参数验证、环境检查等功能
"""

import sys
import os
import platform
import subprocess
import argparse
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import importlib.util

# 定义默认的PortManager类
class DefaultPortManager:
    def get_port_info(self, port: int) -> Dict[str, Any]:
        return {'available': True, 'processes': []}
    
    def get_available_port(self, start_port: int = 5000, max_attempts: int = 100) -> Optional[int]:
        return start_port

def default_check_and_cleanup_port(port: int, host: str = '0.0.0.0') -> Dict[str, Any]:
    return {'success': True, 'message': f'端口 {port} 检查跳过'}

# 尝试导入实际的端口管理功能
try:
    from .port_utils import PortManager as RealPortManager, check_and_cleanup_port as real_check_and_cleanup_port
    # 使用实际的实现
    PortManager = RealPortManager
    check_and_cleanup_port = real_check_and_cleanup_port
except ImportError:
    # 使用默认实现
    PortManager = DefaultPortManager
    check_and_cleanup_port = default_check_and_cleanup_port

# 定义颜色常量类
class ColorConstants:
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ''
    
    class Back:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ''
    
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ''

# 初始化颜色常量
Fore = ColorConstants.Fore
Back = ColorConstants.Back
Style = ColorConstants.Style

# 尝试导入colorama并更新颜色常量
try:
    import colorama
    from colorama import Fore as ColoramaFore, Back as ColoramaBack, Style as ColoramaStyle
    colorama.init(autoreset=True)
    
    # 创建新的颜色常量类
    class UpdatedFore:
        RED = ColoramaFore.RED
        GREEN = ColoramaFore.GREEN
        YELLOW = ColoramaFore.YELLOW
        BLUE = ColoramaFore.BLUE
        MAGENTA = ColoramaFore.MAGENTA
        CYAN = ColoramaFore.CYAN
        WHITE = ColoramaFore.WHITE
        RESET = ColoramaFore.RESET
    
    class UpdatedBack:
        RED = ColoramaBack.RED
        GREEN = ColoramaBack.GREEN
        YELLOW = ColoramaBack.YELLOW
        BLUE = ColoramaBack.BLUE
        MAGENTA = ColoramaBack.MAGENTA
        CYAN = ColoramaBack.CYAN
        WHITE = ColoramaBack.WHITE
        RESET = ColoramaBack.RESET
    
    class UpdatedStyle:
        BRIGHT = ColoramaStyle.BRIGHT
        DIM = ColoramaStyle.DIM
        NORMAL = ColoramaStyle.NORMAL
        RESET_ALL = ColoramaStyle.RESET_ALL
    
    # 更新颜色常量引用
    Fore = UpdatedFore
    Back = UpdatedBack
    Style = UpdatedStyle
        
except ImportError:
    pass

class StartupManager:
    """启动流程管理器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.startup_configs = self._load_startup_configs()
        self.environment_checks = self._load_environment_checks()
    
    def _load_startup_configs(self) -> Dict[str, Dict[str, Any]]:
        """加载启动配置"""
        return {
            'serve': {
                'description': '启动Web服务器',
                'requires_confirmation': True,
                'default_params': {
                    'host': '127.0.0.1',
                    'port': 5000,
                    'debug': False,
                    'websocket': False
                },
                'usage_text': f'''{Fore.CYAN}🚀 VectorSphere Web服务启动{Style.RESET_ALL}

{Fore.YELLOW}用法:{Style.RESET_ALL}
  python app.py serve [选项]

{Fore.YELLOW}选项:{Style.RESET_ALL}
  --host HOST        服务器主机地址 (默认: 127.0.0.1)
  --port PORT        服务器端口 (默认: 5000)
  --debug            启用调试模式
  --websocket        启用WebSocket支持
  --config FILE      指定配置文件路径

{Fore.YELLOW}示例:{Style.RESET_ALL}
  {Fore.GREEN}python app.py serve --host 0.0.0.0 --port 8080{Style.RESET_ALL}
  {Fore.GREEN}python app.py serve --websocket --debug{Style.RESET_ALL}
'''
            },
            'info': {
                'description': '显示服务信息',
                'requires_confirmation': False,
                'default_params': {},
                'usage_text': f'''{Fore.CYAN}📊 VectorSphere 服务信息{Style.RESET_ALL}

{Fore.YELLOW}用法:{Style.RESET_ALL}
  python app.py info

{Fore.YELLOW}功能:{Style.RESET_ALL}
  • 显示系统功能特性
  • 显示访问地址和端口
  • 显示使用示例
'''
            },
            'train': {
                'description': '命令行训练模型',
                'requires_confirmation': True,
                'default_params': {
                    'model': None,
                    'data': None,
                    'config': None
                },
                'usage_text': f'''{Fore.CYAN}🤖 VectorSphere 模型训练{Style.RESET_ALL}

{Fore.YELLOW}用法:{Style.RESET_ALL}
  python app.py train --model MODEL_NAME [选项]

{Fore.YELLOW}选项:{Style.RESET_ALL}
  --model MODEL      模型名称 (必需)
  --data PATH        训练数据路径
  --config FILE      配置文件路径

{Fore.YELLOW}示例:{Style.RESET_ALL}
  {Fore.GREEN}python app.py train --model bert_classifier --data ./data/train.csv{Style.RESET_ALL}
'''
            },
            'test': {
                'description': '运行基本功能测试',
                'requires_confirmation': False,
                'default_params': {},
                'usage_text': f'''{Fore.CYAN}🧪 VectorSphere 功能测试{Style.RESET_ALL}

{Fore.YELLOW}用法:{Style.RESET_ALL}
  python app.py test [选项]

{Fore.YELLOW}选项:{Style.RESET_ALL}
  --config FILE      指定配置文件路径

{Fore.YELLOW}功能:{Style.RESET_ALL}
  • 数据库连接测试
  • 向量存储测试
  • 语义搜索测试
  • API接口测试
  • WebSocket连接测试
'''
            }
        }
    
    def _load_environment_checks(self) -> List[Dict[str, Any]]:
        """加载环境检查配置"""
        return [
            {
                'name': 'python_version',
                'type': 'version_check',
                'method': 'check_python_version',
                'critical': True,
                'min_version': '3.8.0',
                'error_message': 'Python 3.8+ 是必需的'
            },
            {
                'name': 'dependencies',
                'type': 'package_check',
                'method': 'check_dependencies',
                'critical': True,
                'required_packages': ['flask', 'torch', 'transformers'],
                'error_message': '缺少必需的依赖包'
            },
            {
                'name': 'port_availability',
                'type': 'port_check',
                'method': 'check_port_available',
                'critical': True,
                'ports': [5000, 8080],
                'auto_cleanup': True,
                'error_message': '指定端口已被占用'
            }
        ]
    
    def show_usage(self, command: Optional[str] = None, args: Optional[dict] = None):
        """显示使用说明"""
        if command and command in self.startup_configs:
            # 显示详细的命令使用说明
            self._show_general_usage(command, args)
        else:
            # 显示通用帮助信息
            self._show_general_usage()
    
    def _show_general_usage(self, command=None, args=None):
        """显示通用使用说明"""
        print("\n" + "=" * 70)
        print("VectorSphere 向量数据库服务 - 启动向导")
        print("=" * 70)
        
        # 显示当前命令信息
        if command:
            print(f"📋 执行命令: {command}")
            if command == 'serve' and args:
                host = args.get('host', '127.0.0.1')
                port = args.get('port', 5000)
                debug = args.get('debug', False)
                websocket = args.get('websocket', False)
                
                print("\n启动配置:")
                print(f"  • 服务地址: {host}:{port}")
                print(f"  • 调试模式: {'启用' if debug else '禁用'}")
                print(f"  • WebSocket: {'启用' if websocket else '禁用'}")
                
                print("\n服务地址:")
                print(f"  • 主页面: http://{host}:{port}")
                print(f"  • API文档: http://{host}:{port}/api/docs")
                print(f"  • 健康检查: http://{host}:{port}/health")
                print(f"  • 管理界面: http://{host}:{port}/admin")
        else:
            print(f"""{Fore.YELLOW}可用命令:{Style.RESET_ALL}""")
            
            for cmd, config in self.startup_configs.items():
                print(f"  {Fore.GREEN}{cmd:<8}{Style.RESET_ALL} - {config['description']}")
        
        print("\n核心功能:")
        print(" 向量检索: 高性能相似度搜索")
        print(" 语义理解: 智能文本分析")
        print(" 数据管理: 多模态数据存储")
        print(" API接口: RESTful风格接口")
        print(" 实时通信: WebSocket支持")
        print("  监控面板: 性能指标可视化")
        
        print("\n使用指南:")
        print("  1. 首次启动会自动初始化数据库和索引")
        print("  2. 支持热重载，代码修改后自动重启")
        print("  3. 按 Ctrl+C 可安全停止服务")
        print("  4. 日志文件保存在 logs/ 目录下")
        print("  5. 配置文件位于 config/ 目录")
        
        print("\n常用命令示例:")
        print("  # 基础启动")
        print("  python app.py serve")
        print("  \n  # 自定义端口和主机")
        print("  python app.py serve --host 0.0.0.0 --port 8080")
        print("  \n  # 启用调试和WebSocket")
        print("  python app.py serve --debug --websocket")
        print("  \n  # 跳过此向导")
        print("  python app.py serve --skip-usage")
        
        print("\n注意事项:")
        print("  • 确保端口未被占用")
        print("  • 建议在虚拟环境中运行")
        print("  • 生产环境请关闭调试模式")
        print("  • 定期备份重要数据")
        
        if not command:
            print(f"""
{Fore.YELLOW}使用方法:{Style.RESET_ALL}
  python app.py <命令> [选项]
  python app.py <命令> --help  # 查看命令详细帮助

{Fore.YELLOW}示例:{Style.RESET_ALL}
  {Fore.GREEN}python app.py serve{Style.RESET_ALL}                    # 启动Web服务
  {Fore.GREEN}python app.py serve --port 8080{Style.RESET_ALL}        # 指定端口启动
  {Fore.GREEN}python app.py info{Style.RESET_ALL}                     # 查看服务信息
  {Fore.GREEN}python app.py train --model bert{Style.RESET_ALL}        # 训练模型
  {Fore.GREEN}python app.py test{Style.RESET_ALL}                     # 运行测试
""")
        
        print("=" * 70)
    
    def _ask_confirmation(self, message: str) -> bool:
        """询问用户确认"""
        try:
            response = input(f"\n{Fore.YELLOW}❓ {message} (y/N): {Style.RESET_ALL}").strip().lower()
            return response in ['y', 'yes', '是', 'Y']
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Fore.RED}❌ 用户取消操作{Style.RESET_ALL}")
            return False
    
    def validate_environment(self, port: int = 5000) -> bool:
        """验证环境
        
        Args:
            port: 要检查的端口号
            
        Returns:
            bool: 环境是否有效
        """
        print(f"\n{Fore.CYAN}🔍 环境检查中...{Style.RESET_ALL}")
        
        # 检查Python版本
        print(f"  检查Python版本...", end=" ")
        python_check = self.check_python_version({'min_version': '3.8.0'})
        if python_check['success']:
            print(f"{Fore.GREEN}✅ 通过{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ 失败{Style.RESET_ALL}")
            print(f"    {python_check['message']}")
            return False
        
        # 检查端口可用性
        print(f"  检查端口 {port} 可用性...", end=" ")
        port_check = self.check_port_available(port)
        if port_check['success']:
            print(f"{Fore.GREEN}✅ 通过{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}❌ 失败{Style.RESET_ALL}")
            print(f"    {port_check['message']}")
            return False
        
        print(f"{Fore.GREEN}✅ 环境检查通过{Style.RESET_ALL}")
        return True
    
    def confirm_startup(self, args: argparse.Namespace) -> bool:
        """确认启动参数
        
        Args:
            args: 解析后的命令行参数
            
        Returns:
            bool: 用户是否确认启动
        """
        print(f"\n{Fore.CYAN}📋 启动参数确认:{Style.RESET_ALL}")
        print(f"  主机: {Fore.GREEN}{args.host}{Style.RESET_ALL}")
        print(f"  端口: {Fore.GREEN}{args.port}{Style.RESET_ALL}")
        print(f"  调试模式: {Fore.GREEN}{'是' if args.debug else '否'}{Style.RESET_ALL}")
        print(f"  WebSocket: {Fore.GREEN}{'是' if args.websocket else '否'}{Style.RESET_ALL}")
        
        if hasattr(args, 'config') and args.config:
            print(f"  配置文件: {Fore.GREEN}{args.config}{Style.RESET_ALL}")
        
        return self._ask_confirmation("确认使用以上参数启动服务?")
    
    # 环境检查方法实现
    def check_python_version(self, check_config: Dict[str, Any]) -> Dict[str, Any]:
        """检查Python版本"""
        try:
            current_version = platform.python_version()
            min_version = check_config['min_version']
            
            # 简单的版本比较
            current_parts = [int(x) for x in current_version.split('.')]
            min_parts = [int(x) for x in min_version.split('.')]
            
            if current_parts >= min_parts:
                return {
                    'success': True,
                    'message': f'Python {current_version}'
                }
            else:
                return {
                    'success': False,
                    'message': f'当前版本 {current_version}, 需要 {min_version}+'
                }
        except Exception as e:
            return {
                'success': False,
                'message': f'版本检查失败: {str(e)}'
            }
    
    def check_dependencies(self, check_config: Dict[str, Any]) -> Dict[str, Any]:
        """检查依赖包"""
        try:
            required_packages = check_config.get('required_packages', [])
            missing_packages = []
            
            for package in required_packages:
                try:
                    importlib.import_module(package)
                except ImportError:
                    missing_packages.append(package)
            
            if not missing_packages:
                return {
                    'success': True,
                    'message': f'所有依赖包已安装 ({len(required_packages)} 个)'
                }
            else:
                return {
                    'success': False,
                    'message': f'缺少依赖包: {", ".join(missing_packages)}'
                }
        except Exception as e:
            return {
                'success': False,
                'message': f'依赖检查失败: {str(e)}'
            }
    
    def check_port_available(self, port: int) -> Dict[str, Any]:
        """检查端口可用性"""
        try:
            # 使用现有的端口检查和清理功能
            result = check_and_cleanup_port(port)
            return result
        except Exception as e:
            return {
                'success': False,
                'message': f'端口检查失败: {str(e)}'
            }
    
    def cleanup_port(self, port: int, host: str = '0.0.0.0') -> Dict[str, Any]:
        """清理占用端口的进程
        
        Args:
            port: 端口号
            host: 主机地址
            
        Returns:
            dict: 清理结果
        """
        return check_and_cleanup_port(port, host)
    
    def get_port_info(self, port: int) -> Dict[str, Any]:
        """获取端口详细信息
        
        Args:
            port: 端口号
            
        Returns:
            dict: 端口信息
        """
        port_manager = PortManager()
        return port_manager.get_port_info(port)
    
    def find_available_port(self, start_port: int = 5000, max_attempts: int = 100) -> Optional[int]:
        """查找可用端口
        
        Args:
            start_port: 起始端口号
            max_attempts: 最大尝试次数
            
        Returns:
            int: 可用端口号，如果没有找到返回None
        """
        port_manager = PortManager()
        return port_manager.get_available_port(start_port, max_attempts)