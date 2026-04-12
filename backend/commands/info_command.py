# -*- coding: utf-8 -*-
"""
Info command implementation

显示平台详细信息，包括服务状态、功能模块、系统配置等。
"""

import os
import sys
import argparse
import json
import socket
from datetime import datetime
from typing import Dict, Any, List, Optional

from .base_command import BaseCommand


class InfoCommand(BaseCommand):
    """Command to display service information"""
    
    def __init__(self):
        super().__init__()
        self._services_info = {}
        self._modules_info = {}
    
    def get_command_name(self) -> str:
        return 'info'
    
    def get_command_help(self) -> str:
        return '显示平台详细信息'
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add info command specific arguments"""
        parser.add_argument('--format', choices=['text', 'json'], 
                          default='text', help='输出格式')
        parser.add_argument('--verbose', '-v', action='store_true', 
                          help='显示详细信息')
        parser.add_argument('--section', choices=[
            'all', 'services', 'modules', 'system', 'api', 'database'
        ], default='all', help='显示特定部分信息')
        parser.add_argument('--check-health', action='store_true',
                          help='检查服务健康状态')
    
    def execute(self, args: argparse.Namespace) -> int:
        """Execute the info command"""
        try:
            # 收集信息
            self._collect_services_info()
            self._collect_modules_info()
            
            if args.check_health:
                return self._check_health(args)
            
            if args.format == 'json':
                self._show_json_info(args.verbose, args.section)
            else:
                self._show_text_info(args.verbose, args.section)
            return 0
        except Exception as e:
            print(f"❌ 获取服务信息失败: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    def _collect_services_info(self):
        """收集服务层信息"""
        services = {
            'training': {
                'name': '训练服务',
                'module': 'backend.services.training_service',
                'description': '模型训练管理',
                'features': ['标准训练', '分布式训练', '三阶段训练', '场景化训练']
            },
            'three_stage_training': {
                'name': '三阶段训练服务',
                'module': 'backend.services.three_stage_training_service',
                'description': 'PT-SFT-DPO三阶段训练',
                'features': ['预训练', '监督微调', '偏好优化']
            },
            'pipeline': {
                'name': '流水线服务',
                'module': 'backend.services.pipeline_service',
                'description': '训练流水线管理',
                'features': ['流水线创建', '执行控制', '模板管理']
            },
            'hyperparameter_optimization': {
                'name': '超参数优化服务',
                'module': 'backend.services.hyperparameter_optimization_service',
                'description': '超参数自动优化',
                'features': ['随机搜索', '网格搜索', '贝叶斯优化']
            },
            'model_deployment': {
                'name': '模型部署服务',
                'module': 'backend.services.model_deployment_service',
                'description': '模型部署和服务化',
                'features': ['容器部署', '滚动更新', '金丝雀发布', '蓝绿部署']
            },
            'model_evaluation': {
                'name': '模型评估服务',
                'module': 'backend.services.model_evaluation_service',
                'description': '模型性能评估',
                'features': ['自动评估', '模型对比', '指标计算']
            },
            'model_optimization': {
                'name': '模型优化服务',
                'module': 'backend.services.model_optimization_service',
                'description': '模型压缩和优化',
                'features': ['量化', '剪枝', '知识蒸馏', '推理优化']
            },
            'model_selection': {
                'name': '模型选择服务',
                'module': 'backend.services.model_selection_service',
                'description': '智能模型推荐',
                'features': ['模型推荐', '配置生成', '模型搜索']
            },
            'intelligent_decision': {
                'name': '智能决策服务',
                'module': 'backend.services.intelligent_decision_service',
                'description': 'AI驱动的智能决策',
                'features': ['贝叶斯优化', '多臂老虎机', '强化学习', 'LangGraph代理']
            },
            'monitoring_operations': {
                'name': '监控运维服务',
                'module': 'backend.services.monitoring_operations_service',
                'description': '系统监控和自动化运维',
                'features': ['性能监控', '告警管理', '自动化任务', '报告生成']
            },
            'workflow': {
                'name': '工作流服务',
                'module': 'backend.services.workflow_service',
                'description': '工作流管理',
                'features': ['工作流创建', '执行控制', '模板管理']
            },
            'billing': {
                'name': '计费服务',
                'module': 'backend.services.billing_service',
                'description': '资源计费管理',
                'features': ['使用量统计', '账单生成', '配额管理']
            },
            'auth': {
                'name': '认证服务',
                'module': 'backend.services.auth_service',
                'description': '用户认证授权',
                'features': ['用户认证', 'JWT令牌', '权限管理']
            },
            'tenant': {
                'name': '租户服务',
                'module': 'backend.services.tenant_service',
                'description': '多租户管理',
                'features': ['租户隔离', '配额管理', '成员管理']
            }
        }
        
        # 检查服务可用性
        for key, info in services.items():
            try:
                module_path = info['module']
                __import__(module_path)
                info['available'] = True
            except ImportError:
                info['available'] = False
        
        self._services_info = services
    
    def _collect_modules_info(self):
        """收集功能模块信息"""
        modules = {
            'training': {
                'name': '训练模块',
                'path': 'backend.modules.training',
                'description': '模型训练核心模块',
                'submodules': ['standard', 'distributed', 'multimodal', 'three_stage', 'scenario']
            },
            'database': {
                'name': '数据库模块',
                'path': 'backend.modules.database',
                'description': '数据库连接和ORM管理',
                'submodules': ['manager', 'models', 'migrations']
            },
            'auth': {
                'name': '认证模块',
                'path': 'backend.modules.auth',
                'description': '用户认证和授权',
                'submodules': ['services', 'models', 'middleware']
            },
            'embeddings': {
                'name': '嵌入模块',
                'path': 'backend.modules.embeddings',
                'description': '向量嵌入和语义搜索',
                'submodules': ['manager', 'models']
            },
            'deployment': {
                'name': '部署模块',
                'path': 'backend.modules.deployment',
                'description': '模型部署和容器管理',
                'submodules': ['container_manager', 'service_wrapper']
            },
            'monitoring': {
                'name': '监控模块',
                'path': 'backend.modules.monitoring',
                'description': '系统监控和指标收集',
                'submodules': ['metrics', 'alerts', 'dashboard']
            },
            'optimization': {
                'name': '优化模块',
                'path': 'backend.modules.optimization',
                'description': '模型优化和压缩',
                'submodules': ['quantization', 'pruning', 'distillation']
            },
            'distributed': {
                'name': '分布式模块',
                'path': 'backend.modules.distributed',
                'description': '分布式训练支持',
                'submodules': ['coordinator', 'worker', 'communication']
            },
            'scheduler': {
                'name': '调度模块',
                'path': 'backend.modules.scheduler',
                'description': '任务调度管理',
                'submodules': ['task_queue', 'executor', 'cron']
            },
            'agent': {
                'name': 'Agent模块',
                'path': 'backend.modules.agent',
                'description': 'AI代理和自动化',
                'submodules': ['executor', 'tools', 'memory']
            },
            'checkpoint': {
                'name': '检查点模块',
                'path': 'backend.modules.checkpoint',
                'description': '模型检查点管理',
                'submodules': ['saver', 'loader', 'uploader']
            },
            'dataset': {
                'name': '数据集模块',
                'path': 'backend.modules.dataset',
                'description': '数据集管理和处理',
                'submodules': ['loader', 'processor', 'validator']
            }
        }
        
        # 检查模块可用性
        for key, info in modules.items():
            try:
                __import__(info['path'], fromlist=[''])
                info['available'] = True
            except ImportError:
                info['available'] = False
        
        self._modules_info = modules
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        # 获取系统名称
        system_name = sys.platform
        if system_name.startswith('linux'):
            system_name = 'Linux'
        elif system_name.startswith('win'):
            system_name = 'Windows'
        elif system_name.startswith('darwin'):
            system_name = 'Darwin'
            
        # 获取其他信息
        release = ''
        version = ''
        machine = ''
        processor = ''
        
        if hasattr(os, 'uname'):
            uname = os.uname()
            release = uname.release
            version = uname.version
            machine = uname.machine
            processor = uname.machine
        else:
            # Fallback or Windows
            machine = os.environ.get('PROCESSOR_ARCHITEW6432', os.environ.get('PROCESSOR_ARCHITECTURE', ''))
            processor = os.environ.get('PROCESSOR_IDENTIFIER', '')
            
        return {
            'platform': system_name,
            'platform_release': release,
            'platform_version': version,
            'architecture': machine,
            'processor': processor,
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'python_implementation': sys.implementation.name,
            'hostname': socket.gethostname(),
            'current_time': datetime.now().isoformat()
        }
    
    def _get_api_endpoints(self) -> List[Dict[str, str]]:
        """获取API端点信息"""
        return [
            {'path': '/api/v1/auth', 'description': '认证API', 'methods': 'POST, GET'},
            {'path': '/api/v1/training', 'description': '训练管理API', 'methods': 'GET, POST, PUT, DELETE'},
            {'path': '/api/v1/training/three-stage', 'description': '三阶段训练API', 'methods': 'GET, POST, PUT, DELETE'},
            {'path': '/api/v1/training/pipeline', 'description': '流水线API', 'methods': 'GET, POST, PUT, DELETE'},
            {'path': '/api/v1/training/hyperparameter', 'description': '超参数优化API', 'methods': 'GET, POST'},
            {'path': '/api/v1/training/model-deployment', 'description': '模型部署API', 'methods': 'GET, POST, PUT, DELETE'},
            {'path': '/api/v1/training/model-evaluation', 'description': '模型评估API', 'methods': 'GET, POST'},
            {'path': '/api/v1/training/model-optimization', 'description': '模型优化API', 'methods': 'GET, POST'},
            {'path': '/api/v1/training/model-selection', 'description': '模型选择API', 'methods': 'GET, POST'},
            {'path': '/api/v1/training/intelligent-decision', 'description': '智能决策API', 'methods': 'GET, POST'},
            {'path': '/api/v1/training/monitoring', 'description': '监控运维API', 'methods': 'GET, POST, PUT, DELETE'},
            {'path': '/api/v1/workflow', 'description': '工作流API', 'methods': 'GET, POST, PUT, DELETE'},
            {'path': '/api/v1/tenant', 'description': '租户管理API', 'methods': 'GET, POST, PUT, DELETE'},
            {'path': '/api/v1/billing', 'description': '计费API', 'methods': 'GET, POST'},
            {'path': '/api/v1/datasets', 'description': '数据集API', 'methods': 'GET, POST, PUT, DELETE'},
            {'path': '/api/v1/models', 'description': '模型管理API', 'methods': 'GET, POST, PUT, DELETE'},
            {'path': '/health', 'description': '健康检查', 'methods': 'GET'},
            {'path': '/ws', 'description': 'WebSocket', 'methods': 'WS'}
        ]
    
    def _check_health(self, args: argparse.Namespace) -> int:
        """检查服务健康状态"""
        print("🏥 检查服务健康状态...\n")
        
        health_checks = []
        
        # 检查数据库
        print("📊 数据库连接...", end=" ")
        try:
            from backend.modules.database.manager import get_database_manager
            db_manager = get_database_manager()
            if db_manager.health_check():
                print("✅ 正常")
                health_checks.append(('数据库', True, None))
            else:
                print("❌ 异常")
                health_checks.append(('数据库', False, '健康检查失败'))
        except Exception as e:
            print(f"❌ 错误: {e}")
            health_checks.append(('数据库', False, str(e)))
        
        # 检查训练服务
        print("🎓 训练服务...", end=" ")
        try:
            from backend.services.three_stage_training_service import get_three_stage_training_service
            service = get_three_stage_training_service(use_memory_storage=True)
            print("✅ 正常")
            health_checks.append(('训练服务', True, None))
        except Exception as e:
            print(f"❌ 错误: {e}")
            health_checks.append(('训练服务', False, str(e)))
        
        # 检查流水线服务
        print("🔧 流水线服务...", end=" ")
        try:
            from backend.services.pipeline_service import get_pipeline_service
            service = get_pipeline_service(use_memory_storage=True)
            print("✅ 正常")
            health_checks.append(('流水线服务', True, None))
        except Exception as e:
            print(f"❌ 错误: {e}")
            health_checks.append(('流水线服务', False, str(e)))
        
        # 检查监控服务
        print("📈 监控服务...", end=" ")
        try:
            from backend.services.monitoring_operations_service import get_monitoring_operations_service
            service = get_monitoring_operations_service(use_memory_storage=True)
            print("✅ 正常")
            health_checks.append(('监控服务', True, None))
        except Exception as e:
            print(f"❌ 错误: {e}")
            health_checks.append(('监控服务', False, str(e)))
        
        # 检查智能决策服务
        print("🧠 智能决策服务...", end=" ")
        try:
            from backend.services.intelligent_decision_service import get_intelligent_decision_service
            service = get_intelligent_decision_service(use_memory_storage=True)
            print("✅ 正常")
            health_checks.append(('智能决策服务', True, None))
        except Exception as e:
            print(f"❌ 错误: {e}")
            health_checks.append(('智能决策服务', False, str(e)))
        
        # 汇总
        print("\n" + "=" * 50)
        passed = sum(1 for _, status, _ in health_checks if status)
        total = len(health_checks)
        print(f"📊 健康检查结果: {passed}/{total} 通过")
        
        if passed == total:
            print("🎉 所有服务运行正常!")
            return 0
        else:
            print("\n❌ 以下服务存在问题:")
            for name, status, error in health_checks:
                if not status:
                    print(f"  • {name}: {error}")
            return 1
    
    def _show_text_info(self, verbose: bool = False, section: str = 'all'):
        """Show service information in text format"""
        print("=" * 60)
        print("🚀 VectorSphere 智能训练平台")
        print("=" * 60)
        print(f"📅 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        if section in ['all', 'system']:
            self._show_system_info_text(verbose)
        
        if section in ['all', 'services']:
            self._show_services_info_text(verbose)
        
        if section in ['all', 'modules']:
            self._show_modules_info_text(verbose)
        
        if section in ['all', 'api']:
            self._show_api_info_text(verbose)
        
        if verbose and section == 'all':
            self._show_usage_examples()
    
    def _show_system_info_text(self, verbose: bool):
        """显示系统信息"""
        print("📊 系统信息:")
        print("-" * 40)
        sys_info = self._get_system_info()
        print(f"  • 操作系统: {sys_info['platform']} {sys_info['platform_release']}")
        print(f"  • 架构: {sys_info['architecture']}")
        print(f"  • Python: {sys_info['python_version']}")
        print(f"  • 主机名: {sys_info['hostname']}")
        print()
    
    def _show_services_info_text(self, verbose: bool):
        """显示服务信息"""
        print("🔧 核心服务:")
        print("-" * 40)
        
        available_count = sum(1 for s in self._services_info.values() if s.get('available'))
        print(f"  可用服务: {available_count}/{len(self._services_info)}")
        print()
        
        for key, info in self._services_info.items():
            status = "✅" if info.get('available') else "❌"
            print(f"  {status} {info['name']}")
            if verbose:
                print(f"     描述: {info['description']}")
                print(f"     功能: {', '.join(info['features'])}")
        print()
    
    def _show_modules_info_text(self, verbose: bool):
        """显示模块信息"""
        print("📦 功能模块:")
        print("-" * 40)
        
        available_count = sum(1 for m in self._modules_info.values() if m.get('available'))
        print(f"  可用模块: {available_count}/{len(self._modules_info)}")
        print()
        
        for key, info in self._modules_info.items():
            status = "✅" if info.get('available') else "❌"
            print(f"  {status} {info['name']}")
            if verbose:
                print(f"     描述: {info['description']}")
                print(f"     子模块: {', '.join(info['submodules'])}")
        print()
    
    def _show_api_info_text(self, verbose: bool):
        """显示API信息"""
        print("🌐 API端点:")
        print("-" * 40)
        
        endpoints = self._get_api_endpoints()
        for ep in endpoints:
            print(f"  • {ep['path']}")
            if verbose:
                print(f"    {ep['description']} [{ep['methods']}]")
        print()
    
    def _show_usage_examples(self):
        """显示使用示例"""
        print("📝 使用示例:")
        print("-" * 40)
        print("  # 启动服务")
        print("  python app.py serve --host 0.0.0.0 --port 8080")
        print()
        print("  # 启动带WebSocket的服务")
        print("  python app.py serve --websocket --debug")
        print()
        print("  # 训练模型")
        print("  python app.py train --model llama2 --epochs 10 --training-mode three-stage")
        print()
        print("  # 运行测试")
        print("  python app.py test --category training --verbose")
        print()
        print("  # 查看JSON格式信息")
        print("  python app.py info --format json --verbose")
        print()
    
    def _show_json_info(self, verbose: bool = False, section: str = 'all'):
        """Show service information in JSON format"""
        info = {
            "name": "VectorSphere",
            "version": "1.0.0",
            "description": "AI模型训练和管理平台",
            "timestamp": datetime.now().isoformat()
        }
        
        if section in ['all', 'system']:
            info['system'] = self._get_system_info()
        
        if section in ['all', 'services']:
            info['services'] = self._services_info
        
        if section in ['all', 'modules']:
            info['modules'] = self._modules_info
        
        if section in ['all', 'api']:
            info['api_endpoints'] = self._get_api_endpoints()
        
        if verbose:
            info['features'] = [
                "向量数据存储与检索",
                "智能语义搜索",
                "多模态数据支持",
                "RESTful API 接口",
                "WebSocket 实时通信",
                "可视化管理界面",
                "三阶段训练 (PT-SFT-DPO)",
                "分布式训练支持",
                "超参数自动优化",
                "模型部署和服务化",
                "智能决策引擎",
                "多租户支持"
            ]
            
            info['training_modes'] = [
                "standard - 标准训练",
                "distributed - 分布式训练",
                "multimodal - 多模态训练",
                "distillation - 知识蒸馏",
                "three-stage - 三阶段训练",
                "scenario - 场景化训练"
            ]
        
        print(json.dumps(info, indent=2, ensure_ascii=False))
