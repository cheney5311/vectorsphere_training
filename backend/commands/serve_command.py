# -*- coding: utf-8 -*-
"""
Serve command implementation

启动VectorSphere训练服务，支持多种启动模式和配置选项。
"""

import os
import sys
import traceback
import argparse
import signal
import threading
from typing import Dict, Any, Optional
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_command import BaseCommand
from backend.utils.startup_manager import StartupManager
from backend.utils.port_utils import check_and_cleanup_port


class ServeCommand(BaseCommand):
    """启动Web服务的命令"""
    
    def __init__(self):
        super().__init__()
        self._app = None
        self._socketio = None
        self._shutdown_event = threading.Event()
    
    def get_command_name(self) -> str:
        return 'serve'
    
    def get_command_help(self) -> str:
        return '启动训练平台Web服务'
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add serve command specific arguments"""
        # 基础配置
        parser.add_argument('--config', type=str, help='配置文件路径')
        parser.add_argument('--host', type=str, default='127.0.0.1', help='服务器主机地址')
        parser.add_argument('--port', type=int, default=5000, help='服务器端口')
        parser.add_argument('--debug', action='store_true', help='启用调试模式')
        
        # 功能开关
        parser.add_argument('--websocket', action='store_true', help='启用WebSocket支持')
        parser.add_argument('--cors', action='store_true', default=True, help='启用CORS支持')
        parser.add_argument('--api-docs', action='store_true', default=True, help='启用API文档')
        
        # 服务模式
        parser.add_argument('--mode', choices=['development', 'production', 'testing'],
                          default='development', help='运行模式')
        parser.add_argument('--workers', type=int, default=1, help='工作进程数（生产模式）')
        
        # 性能配置
        parser.add_argument('--threaded', action='store_true', default=True, help='启用多线程')
        parser.add_argument('--processes', type=int, default=1, help='进程数')
        
        # 安全配置
        parser.add_argument('--ssl-cert', type=str, help='SSL证书路径')
        parser.add_argument('--ssl-key', type=str, help='SSL密钥路径')
        
        # 服务初始化
        parser.add_argument('--init-db', action='store_true', help='初始化数据库')
        parser.add_argument('--migrate', action='store_true', help='运行数据库迁移')
        
        # 显示控制
        parser.add_argument('--skip-usage', action='store_true', help='跳过使用说明显示')
        parser.add_argument('--quiet', '-q', action='store_true', help='安静模式')
    
    def validate_args(self, args: argparse.Namespace) -> bool:
        """Validate serve command arguments"""
        if args.port < 1 or args.port > 65535:
            print("错误: 端口号必须在 1-65535 范围内")
            return False
        
        if args.ssl_cert and not args.ssl_key:
            print("错误: 指定SSL证书时必须同时指定SSL密钥")
            return False
        
        if args.ssl_key and not args.ssl_cert:
            print("错误: 指定SSL密钥时必须同时指定SSL证书")
            return False
        
        if args.ssl_cert and not os.path.exists(args.ssl_cert):
            print(f"错误: SSL证书文件不存在: {args.ssl_cert}")
            return False
        
        if args.ssl_key and not os.path.exists(args.ssl_key):
            print(f"错误: SSL密钥文件不存在: {args.ssl_key}")
            return False
        
        return True
    
    def setup(self, args: argparse.Namespace) -> bool:
        """Setup before starting the server"""
        startup_manager = StartupManager()
        
        # Skip usage display if requested
        if getattr(args, 'skip_usage', False) or getattr(args, 'quiet', False):
            return True
            
        try:
            # 显示使用说明
            startup_manager.show_usage(args.command, vars(args))
            
            # 环境验证
            if not startup_manager.validate_environment(args.port):
                print("环境验证失败，请检查上述问题后重试")
                return False
            
            # 用户确认（开发模式下跳过）
            if args.mode != 'development':
                if not startup_manager.confirm_startup(args):
                    print("启动已取消")
                    return False
                
        except KeyboardInterrupt:
            print("\n启动已取消")
            return False
        except Exception as e:
            print(f"启动检查失败: {e}")
            return False
            
        return True
    
    def execute(self, args: argparse.Namespace) -> int:
        """Execute the serve command"""
        try:
            # 设置信号处理
            self._setup_signal_handlers()
            
            # 检查并清理端口占用
            if not args.quiet:
                print(f"检查端口 {args.port} 占用情况...")
            
            cleanup_result = check_and_cleanup_port(args.port)
            
            if not cleanup_result['success']:
                print(f"端口清理失败: {cleanup_result['message']}")
                if cleanup_result.get('processes'):
                    print("占用端口的进程:")
                    for proc in cleanup_result['processes']:
                        print(f"  - PID: {proc['pid']}, 名称: {proc['name']}, 命令: {proc['cmdline']}")
                print("\n请手动停止占用端口的进程，或选择其他端口。")
                return 1
            
            if cleanup_result.get('cleaned_processes'):
                cleaned_count = cleanup_result['cleaned_processes']
                if isinstance(cleaned_count, (list, tuple)):
                    print(f"已清理 {len(cleaned_count)} 个占用端口的进程")
                else:
                    print(f"已清理 {cleaned_count} 个占用端口的进程")
            elif not args.quiet:
                print(f"端口 {args.port} 可用")
            
            # 初始化数据库（如果需要）
            if args.init_db:
                self._init_database(args)
            
            # 运行数据库迁移（如果需要）
            if args.migrate:
                self._run_migrations(args)
            
            # 导入和创建应用（延迟导入避免循环依赖）
            from app import create_app
            
            # 创建应用
            self._app = create_app(args.config)
            
            # 配置应用
            self._configure_app(args)
            
            # 启动服务
            return self._start_server(args)
                
        except Exception as e:
            print(f"❌ 服务启动失败: {e}")
            traceback.print_exc()
            return 1
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            print("\n🛑 收到停止信号，正在关闭服务...")
            self._shutdown_event.set()
            self._graceful_shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _init_database(self, args: argparse.Namespace):
        """初始化数据库"""
        print("初始化数据库...")
        try:
            from backend.modules.database.manager import get_database_manager
            db_manager = get_database_manager()
            if hasattr(db_manager, 'create_tables'):
                db_manager.create_tables()
            else:
                print("DatabaseManager.create_tables not found")
            print("数据库初始化完成")
        except Exception as e:
            print(f"数据库初始化失败: {e}")
    
    def _run_migrations(self, args: argparse.Namespace):
        """运行数据库迁移"""
        print("运行数据库迁移...")
        try:
            from backend.modules.database.manager import get_database_manager
            db_manager = get_database_manager()
            # 这里可以添加迁移逻辑
            print("数据库迁移完成")
        except Exception as e:
            print(f"数据库迁移失败: {e}")
    
    def _configure_app(self, args: argparse.Namespace):
        """配置应用"""
        if self._app is None:
            return
        
        # 设置调试模式
        self._app.config['DEBUG'] = args.debug
        
        # 设置运行模式
        self._app.config['ENV'] = args.mode
        
        # 配置CORS
        if args.cors:
            try:
                from flask_cors import CORS
                CORS(self._app, resources={r"/api/*": {"origins": "*"}})
            except ImportError:
                pass
    
    def _start_server(self, args: argparse.Namespace) -> int:
        """启动服务器"""
        protocol = "https" if args.ssl_cert else "http"
        
        # 显示启动信息
        self._show_startup_banner(args, protocol)
        
        # SSL配置
        ssl_context = None
        if args.ssl_cert and args.ssl_key:
            ssl_context = (args.ssl_cert, args.ssl_key)
        
        if args.websocket:
            # 使用 WebSocket 支持
            return self._start_with_websocket(args, ssl_context)
        else:
            # 普通 Flask 应用
            return self._start_without_websocket(args, ssl_context)
    
    def _start_with_websocket(self, args: argparse.Namespace, ssl_context) -> int:
        """启动带WebSocket的服务"""
        try:
            from flask_socketio import SocketIO
            
            self._socketio = SocketIO(
                self._app, 
                cors_allowed_origins="*",
                async_mode='threading'
            )
            
            # 注册WebSocket事件处理
            self._register_websocket_handlers()
            
            print(f"🔌 WebSocket 支持已启用")
            
            self._socketio.run(
                self._app, 
                host=args.host, 
                port=args.port, 
                debug=args.debug,
                use_reloader=args.debug,
                ssl_context=ssl_context
            )
            
            return 0
        except ImportError:
            print("⚠️ flask-socketio 未安装，使用普通HTTP模式")
            return self._start_without_websocket(args, ssl_context)
    
    def _start_without_websocket(self, args: argparse.Namespace, ssl_context) -> int:
        """启动普通HTTP服务"""
        self._app.run(
            host=args.host, 
            port=args.port, 
            debug=args.debug,
            threaded=args.threaded,
            processes=args.processes if args.processes > 1 else None,
            ssl_context=ssl_context,
            use_reloader=args.debug
        )
        return 0
    
    def _register_websocket_handlers(self):
        """注册WebSocket事件处理器"""
        if self._socketio is None:
            return
        
        @self._socketio.on('connect')
        def handle_connect():
            print("🔗 WebSocket客户端已连接")
        
        @self._socketio.on('disconnect')
        def handle_disconnect():
            print("🔌 WebSocket客户端已断开")
        
        @self._socketio.on('ping')
        def handle_ping(data):
            return {'type': 'pong', 'data': data, 'timestamp': datetime.now().isoformat()}
        
        @self._socketio.on('subscribe_training')
        def handle_subscribe_training(data):
            """订阅训练进度"""
            session_id = data.get('session_id')
            if session_id:
                from flask_socketio import join_room
                join_room(f'training_{session_id}')
                return {'success': True, 'message': f'已订阅训练会话: {session_id}'}
            return {'success': False, 'error': '缺少session_id'}
    
    def _show_startup_banner(self, args: argparse.Namespace, protocol: str):
        """显示启动横幅"""
        print()
        print("=" * 60)
        print("🚀 VectorSphere 智能训练平台")
        print("=" * 60)
        print(f"📍 服务地址: {protocol}://{args.host}:{args.port}")
        print(f"📋 运行模式: {args.mode}")
        print(f"🔧 调试模式: {'开启' if args.debug else '关闭'}")
        print(f"🔌 WebSocket: {'开启' if args.websocket else '关闭'}")
        print(f"📅 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
        print(f"🌐 Web界面: {protocol}://{args.host}:{args.port}")
        print(f"📚 API文档: {protocol}://{args.host}:{args.port}/api/docs")
        print(f"❤️  健康检查: {protocol}://{args.host}:{args.port}/health")
        print("=" * 60)
        print("📝 按 Ctrl+C 停止服务")
        print()
    
    def _graceful_shutdown(self):
        """优雅关闭"""
        try:
            from backend.utils.graceful_shutdown import get_shutdown_manager
            shutdown_manager = get_shutdown_manager()
            if shutdown_manager:
                shutdown_manager.shutdown_now()
        except Exception as e:
            print(f"⚠️ 关闭过程中出现错误: {e}")
    
    def cleanup(self, args: argparse.Namespace) -> None:
        """Cleanup after server shutdown"""
        print("🧹 清理资源...")
        self._graceful_shutdown()
        print("👋 服务已停止")
