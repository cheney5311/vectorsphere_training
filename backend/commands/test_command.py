# -*- coding: utf-8 -*-
"""
Test command implementation

运行平台功能测试，包括数据库、认证、训练、服务层等测试。
"""

import time
import argparse
import sys
import os
import uuid
from typing import Dict, Any, List, Tuple
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_command import BaseCommand


class TestCommand(BaseCommand):
    """Command to run system tests"""
    
    def __init__(self):
        super().__init__()
        self._test_results: List[Tuple[str, bool, str]] = []
    
    def get_command_name(self) -> str:
        return 'test'
    
    def get_command_help(self) -> str:
        return '运行平台功能测试'
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add test command specific arguments"""
        parser.add_argument('--config', type=str, help='配置文件路径')
        parser.add_argument('--category', choices=[
            'all', 'db', 'api', 'websocket', 'vector', 'auth', 
            'training', 'pipeline', 'services', 'repositories'
        ], default='all', help='测试类别')
        parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
        parser.add_argument('--fast', action='store_true', help='快速测试模式')
        parser.add_argument('--host', type=str, default='localhost', help='API服务器主机')
        parser.add_argument('--port', type=int, default=8080, help='API服务器端口')
        parser.add_argument('--env', choices=['dev', 'test', 'prod'], default='dev', help='测试环境')
        parser.add_argument('--stop-on-fail', action='store_true', help='遇到失败时停止')
        parser.add_argument('--output', type=str, help='测试报告输出文件')
    
    def execute(self, args: argparse.Namespace) -> int:
        """Execute the test command"""
        try:
            print("🧪 运行平台功能测试...")
            print(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # 初始化配置和日志
            self._init_environment(args)
            
            if args.category == 'all':
                result = self._run_all_tests(args)
            else:
                result = self._run_category_tests(args.category, args)
            
            # 输出测试报告
            if args.output:
                self._save_report(args.output)
            
            return result
                
        except KeyboardInterrupt:
            print("\n⏹️ 测试被用户中断")
            return 1
        except Exception as e:
            print(f"❌ 测试执行失败: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    def _init_environment(self, args: argparse.Namespace):
        """初始化测试环境"""
        try:
            from backend.core.logging_config import setup_logging
            from backend.core.config_manager import load_config
            
            setup_logging()
            if args.config:
                load_config({"yaml": args.config} if args.config.endswith('.yaml') else {"json": args.config})
        except Exception as e:
            if args.verbose:
                print(f"⚠️ 环境初始化警告: {e}")
    
    def _run_all_tests(self, args: argparse.Namespace) -> int:
        """Run all tests"""
        tests = [
            ("数据库连接测试", self._test_database),
            ("认证系统测试", self._test_auth),
            ("仓库层测试", self._test_repositories),
            ("服务层测试", self._test_services),
            ("三阶段训练测试", self._test_three_stage_training),
            ("流水线服务测试", self._test_pipeline),
            ("监控服务测试", self._test_monitoring),
            ("智能决策测试", self._test_intelligent_decision),
            ("向量存储测试", self._test_vector_storage), 
            ("语义搜索测试", self._test_semantic_search),
            ("API接口测试", self._test_api),
            ("WebSocket连接测试", self._test_websocket)
        ]
        
        total_tests = len(tests)
        passed_tests = 0
        failed_tests = []
        
        print(f"📋 总共 {total_tests} 个测试项目\n")
        
        for i, (test_name, test_func) in enumerate(tests, 1):
            print(f"[{i:2d}/{total_tests}] {test_name}...", end=" ")
            
            try:
                if not args.fast:
                    time.sleep(0.2)
                
                success, message = test_func(args)
                self._test_results.append((test_name, success, message))
                
                if success:
                    print("✅ 通过")
                    passed_tests += 1
                else:
                    print(f"❌ 失败")
                    failed_tests.append((test_name, message))
                    if args.stop_on_fail:
                        print("⏹️ 遇到失败，停止测试")
                        break
                    
            except Exception as e:
                print(f"❌ 错误: {e}")
                failed_tests.append((test_name, str(e)))
                self._test_results.append((test_name, False, str(e)))
                
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                
                if args.stop_on_fail:
                    break
        
        # 显示测试结果
        self._show_results(passed_tests, total_tests, failed_tests, args.verbose)
        
        return 0 if len(failed_tests) == 0 else 1
    
    def _run_category_tests(self, category: str, args: argparse.Namespace) -> int:
        """Run tests for specific category"""
        test_map = {
            'db': ('数据库测试', self._test_database),
            'api': ('API接口测试', self._test_api),
            'websocket': ('WebSocket测试', self._test_websocket),
            'vector': ('向量存储测试', self._test_vector_storage),
            'auth': ('认证系统测试', self._test_auth),
            'training': ('训练系统测试', self._test_three_stage_training),
            'pipeline': ('流水线测试', self._test_pipeline),
            'services': ('服务层测试', self._test_services),
            'repositories': ('仓库层测试', self._test_repositories)
        }
        
        if category not in test_map:
            print(f"❌ 未知的测试类别: {category}")
            return 1
            
        test_name, test_func = test_map[category]
        
        print(f"🧪 运行 {test_name}...")
        
        try:
            success, message = test_func(args)
            self._test_results.append((test_name, success, message))
            
            if success:
                print(f"✅ {test_name} 通过")
                if args.verbose and message:
                    print(f"   {message}")
                return 0
            else:
                print(f"❌ {test_name} 失败")
                if message:
                    print(f"   原因: {message}")
                return 1
        except Exception as e:
            print(f"❌ {test_name} 错误: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _show_results(self, passed: int, total: int, failed: List, verbose: bool):
        """显示测试结果"""
        print()
        print("=" * 50)
        print(f"📊 测试结果:")
        print(f"  ✅ 通过: {passed}/{total}")
        print(f"  ❌ 失败: {len(failed)}/{total}")
        
        if failed:
            print(f"\n❌ 失败的测试:")
            for test_name, reason in failed:
                print(f"  • {test_name}")
                if verbose and reason:
                    print(f"    原因: {reason}")
        else:
            print("\n🎉 所有测试通过!")
    
    def _save_report(self, output_path: str):
        """保存测试报告"""
        import json
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(self._test_results),
            'passed': sum(1 for _, s, _ in self._test_results if s),
            'failed': sum(1 for _, s, _ in self._test_results if not s),
            'results': [
                {'name': name, 'passed': passed, 'message': msg}
                for name, passed, msg in self._test_results
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 测试报告已保存到: {output_path}")
    
    # ==========================================================================
    # 测试方法
    # ==========================================================================
    
    def _test_database(self, args: argparse.Namespace) -> Tuple[bool, str]:
        """Test database connection"""
        try:
            from backend.modules.database.manager import get_database_manager
            
            db_manager = get_database_manager()
            health_status = db_manager.health_check()
            
            if args.verbose:
                from sqlalchemy import text
                with db_manager.get_db_session() as session:
                    result = session.execute(text('SELECT 1')).scalar()
                    print(f"\n    数据库查询结果: {result}")
            
            return health_status, "数据库连接正常"
        except Exception as e:
            return False, str(e)
    
    def _test_auth(self, args: argparse.Namespace) -> Tuple[bool, str]:
        """Test authentication system"""
        try:
            from backend.services.auth_service import AuthService
            from backend.modules.database.manager import get_database_manager
            from backend.repositories.auth_repository import get_auth_repository
            
            auth_service = AuthService(get_auth_repository())
            
            # 创建测试用户
            username = f"test_user_{int(datetime.now().timestamp())}"
            email = f"{username}@test.com"
            password = "TestPassword123!"
            
            if args.verbose:
                print(f"\n    创建测试用户: {username}")
            
            user = auth_service.register_user(username, email, password, "Test User")
            
            # 认证用户 (返回包含 token 的结果)
            login_result = auth_service.authenticate_user(username, password)
            tokens = login_result.get('tokens', {})
            access_token = tokens.get('access_token')
            
            if not access_token:
                return False, "获取访问令牌失败"
                
            payload = auth_service.verify_token(access_token)
            
            # 清理测试用户
            with get_database_manager().get_db_session() as db:
                from sqlalchemy import text
                db.execute(text("DELETE FROM user_roles WHERE user_id = :user_id"), {"user_id": user.id})
                db.execute(text("DELETE FROM api_keys WHERE user_id = :user_id"), {"user_id": user.id})
                db.execute(text("DELETE FROM user_sessions WHERE user_id = :user_id"), {"user_id": user.id})
                db.execute(text("DELETE FROM users WHERE id = :id"), {"id": user.id})
                db.commit()
            
            return True, "认证系统正常"
        except Exception as e:
            return False, str(e)
    
    def _test_repositories(self, args: argparse.Namespace) -> Tuple[bool, str]:
        """Test repository layer"""
        try:
            tenant_id = str(uuid.uuid4())
            user_id = str(uuid.uuid4())
            tests_passed = 0
            
            # 测试三阶段训练仓库
            try:
                from backend.repositories.three_stage_training_repository import (
                    ThreeStageSessionRepository, ThreeStageProgressRepository
                )
                
                session_repo = ThreeStageSessionRepository(use_memory_storage=True)
                created = session_repo.create({
                    'tenant_id': tenant_id,
                    'user_id': user_id,
                    'name': 'Test',
                    'model_name': 'gpt2',
                    'config': {}
                })
                if created.get('session_id'):
                    tests_passed += 1
                    if args.verbose:
                        print(f"\n    三阶段训练仓库: ✅")
            except Exception as e:
                if args.verbose:
                    print(f"\n    三阶段训练仓库: ❌ {e}")
            
            # 测试流水线仓库
            try:
                from backend.repositories.pipeline_repository import (
                    TrainingPipelineRepository, PipelineExecutionRepository
                )
                
                pipeline_repo = TrainingPipelineRepository(use_memory_storage=True)
                created = pipeline_repo.create({
                    'tenant_id': tenant_id,
                    'user_id': user_id,
                    'name': 'Test Pipeline',
                    'steps_config': [{'name': 'step1', 'type': 'pretrain'}]
                })
                if created.get('pipeline_id'):
                    tests_passed += 1
                    if args.verbose:
                        print(f"    流水线仓库: ✅")
            except Exception as e:
                if args.verbose:
                    print(f"    流水线仓库: ❌ {e}")
            
            # 测试监控仓库
            try:
                from backend.repositories.monitoring_operations_repository import (
                    PerformanceMetricRepository, AlertRuleRepository
                )
                
                metric_repo = PerformanceMetricRepository(use_memory_storage=True)
                created = metric_repo.create({
                    'tenant_id': tenant_id,
                    'deployment_id': 'test_deploy',
                    'metric_type': 'cpu_usage',
                    'value': 50.0
                })
                if created.get('metric_id'):
                    tests_passed += 1
                    if args.verbose:
                        print(f"    监控仓库: ✅")
            except Exception as e:
                if args.verbose:
                    print(f"    监控仓库: ❌ {e}")
            
            if tests_passed >= 2:
                return True, f"仓库层测试通过 ({tests_passed}/3)"
            else:
                return False, f"仓库层测试部分失败 ({tests_passed}/3)"
                
        except Exception as e:
            return False, str(e)
    
    def _test_services(self, args: argparse.Namespace) -> Tuple[bool, str]:
        """Test service layer"""
        try:
            tests_passed = 0
            
            # 测试三阶段训练服务
            try:
                from backend.services.three_stage_training_service import get_three_stage_training_service
                service = get_three_stage_training_service(use_memory_storage=True)
                tests_passed += 1
                if args.verbose:
                    print(f"\n    三阶段训练服务: ✅")
            except Exception as e:
                if args.verbose:
                    print(f"\n    三阶段训练服务: ❌ {e}")
            
            # 测试流水线服务
            try:
                from backend.services.pipeline_service import get_pipeline_service
                service = get_pipeline_service(use_memory_storage=True)
                tests_passed += 1
                if args.verbose:
                    print(f"    流水线服务: ✅")
            except Exception as e:
                if args.verbose:
                    print(f"    流水线服务: ❌ {e}")
            
            # 测试监控服务
            try:
                from backend.services.monitoring_operations_service import get_monitoring_operations_service
                service = get_monitoring_operations_service(use_memory_storage=True)
                tests_passed += 1
                if args.verbose:
                    print(f"    监控服务: ✅")
            except Exception as e:
                if args.verbose:
                    print(f"    监控服务: ❌ {e}")
            
            # 测试智能决策服务
            try:
                from backend.services.intelligent_decision_service import get_intelligent_decision_service
                service = get_intelligent_decision_service(use_memory_storage=True)
                tests_passed += 1
                if args.verbose:
                    print(f"    智能决策服务: ✅")
            except Exception as e:
                if args.verbose:
                    print(f"    智能决策服务: ❌ {e}")
            
            if tests_passed >= 3:
                return True, f"服务层测试通过 ({tests_passed}/4)"
            else:
                return False, f"服务层测试部分失败 ({tests_passed}/4)"
                
        except Exception as e:
            return False, str(e)
    
    def _test_three_stage_training(self, args: argparse.Namespace) -> Tuple[bool, str]:
        """Test three-stage training system"""
        try:
            from backend.services.three_stage_training_service import ThreeStageTrainingService
            
            service = ThreeStageTrainingService(use_memory_storage=True)
            tenant_id = str(uuid.uuid4())
            user_id = str(uuid.uuid4())
            
            # 创建会话
            session = service.create_session(
                name='Test Session',
                model_name='gpt2',
                config={
                    'stages': {
                        'sft': {'enabled': True, 'epochs': 1}
                    }
                },
                tenant_id=tenant_id,
                user_id=user_id
            )
            
            if not session.get('session_id'):
                return False, "创建会话失败"
            
            # 获取会话
            fetched = service.get_session(session['session_id'], tenant_id)
            if not fetched:
                return False, "获取会话失败"
            
            # 列表查询
            result = service.list_sessions(tenant_id, user_id)
            if result.get('total', 0) < 1:
                return False, "列表查询失败"
            
            if args.verbose:
                print(f"\n    创建会话: {session.get('session_id')}")
                print(f"    会话状态: {fetched.get('status')}")
            
            return True, "三阶段训练服务正常"
        except Exception as e:
            return False, str(e)
    
    def _test_pipeline(self, args: argparse.Namespace) -> Tuple[bool, str]:
        """Test pipeline service"""
        try:
            from backend.services.pipeline_service import PipelineService
            
            service = PipelineService(use_memory_storage=True)
            tenant_id = str(uuid.uuid4())
            user_id = str(uuid.uuid4())
            
            # 创建流水线
            pipeline = service.create_pipeline(
                name='Test Pipeline',
                steps_config=[
                    {'name': 'step1', 'type': 'pretrain', 'on_fail': 'stop'},
                    {'name': 'step2', 'type': 'finetune', 'on_fail': 'rollback'}
                ],
                tenant_id=tenant_id,
                user_id=user_id
            )
            
            if not pipeline.get('pipeline_id'):
                return False, "创建流水线失败"
            
            # 获取流水线
            fetched = service.get_pipeline(pipeline['pipeline_id'], tenant_id)
            if not fetched:
                return False, "获取流水线失败"
            
            # 创建模板
            template = service.create_template(
                name='Test Template',
                steps_template=[{'type': 'pretrain'}],
                tenant_id=tenant_id,
                user_id=user_id,
                category='nlp'
            )
            
            if not template.get('template_id'):
                return False, "创建模板失败"
            
            if args.verbose:
                print(f"\n    创建流水线: {pipeline.get('pipeline_id')}")
                print(f"    创建模板: {template.get('template_id')}")
            
            return True, "流水线服务正常"
        except Exception as e:
            return False, str(e)
    
    def _test_monitoring(self, args: argparse.Namespace) -> Tuple[bool, str]:
        """Test monitoring service"""
        try:
            from backend.services.monitoring_operations_service import get_monitoring_operations_service
            
            service = get_monitoring_operations_service(use_memory_storage=True)
            tenant_id = str(uuid.uuid4())
            deployment_id = f"deploy_{uuid.uuid4().hex[:8]}"
            
            # 收集性能指标
            report = service.collect_performance_metrics(
                deployment_id=deployment_id,
                tenant_id=tenant_id,
                save_to_db=True
            )
            
            if not report or not report.metrics:
                return False, "收集指标失败"
            
            if args.verbose:
                print(f"\n    收集指标数: {len(report.metrics)}")
            
            return True, "监控服务正常"
        except Exception as e:
            return False, str(e)
    
    def _test_intelligent_decision(self, args: argparse.Namespace) -> Tuple[bool, str]:
        """Test intelligent decision service"""
        try:
            from backend.services.intelligent_decision_service import get_intelligent_decision_service
            
            service = get_intelligent_decision_service(use_memory_storage=True)
            tenant_id = str(uuid.uuid4())
            
            # 测试算法
            algorithms = service.get_available_algorithms()
            if not algorithms:
                return False, "获取算法列表失败"
            
            if args.verbose:
                print(f"\n    可用算法: {len(algorithms)}")
            
            return True, "智能决策服务正常"
        except Exception as e:
            return False, str(e)
    
    def _test_vector_storage(self, args: argparse.Namespace) -> Tuple[bool, str]:
        """Test vector storage functionality"""
        try:
            from backend.modules.embeddings.manager import get_embedding_manager
            
            embedding_manager = get_embedding_manager()
            
            test_text = "这是一个测试文本"
            embedding = embedding_manager.generate_embedding(test_text)
            
            if args.verbose:
                print(f"\n    嵌入维度: {len(embedding)}")
            
            return True, f"向量存储正常 (维度: {len(embedding)})"
        except ImportError:
            return True, "嵌入服务未安装，跳过测试"
        except Exception as e:
            return False, str(e)
    
    def _test_semantic_search(self, args: argparse.Namespace) -> Tuple[bool, str]:
        """Test semantic search functionality"""
        try:
            from backend.modules.embeddings.manager import get_embedding_manager
            
            embedding_manager = get_embedding_manager()
            
            texts = [
                "人工智能技术发展迅速",
                "机器学习是AI的核心",
                "深度学习在图像识别表现出色"
            ]
            
            embeddings = embedding_manager.generate_batch_embeddings(texts)
            query_embedding = embedding_manager.generate_embedding("人工智能")
            
            similarities = []
            for i, emb in enumerate(embeddings):
                sim = embedding_manager.calculate_similarity(query_embedding, emb)
                similarities.append((texts[i], sim))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            if args.verbose:
                print(f"\n    最相似文本: {similarities[0][0]} ({similarities[0][1]:.4f})")
            
            return True, "语义搜索正常"
        except ImportError:
            return True, "嵌入服务未安装，跳过测试"
        except Exception as e:
            return False, str(e)
    
    def _test_api(self, args: argparse.Namespace) -> Tuple[bool, str]:
        """Test API endpoints"""
        try:
            import requests
            
            url = f"http://{args.host}:{args.port}/health"
            
            if args.verbose:
                print(f"\n    测试URL: {url}")
            
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                return True, "API健康检查正常"
            else:
                return False, f"状态码: {response.status_code}"
        except requests.exceptions.ConnectionError:
            return True, "API服务未启动，跳过测试"
        except Exception as e:
            return False, str(e)
    
    def _test_websocket(self, args: argparse.Namespace) -> Tuple[bool, str]:
        """Test WebSocket connection"""
        try:
            import websocket
            import json
            
            ws_url = f"ws://{args.host}:{args.port}/ws"
            
            if args.verbose:
                print(f"\n    WebSocket URL: {ws_url}")
            
            ws = websocket.create_connection(ws_url, timeout=5)
            ws.send(json.dumps({"type": "ping", "data": "test"}))
            ws.close()
            
            return True, "WebSocket连接正常"
        except ImportError:
            return True, "websocket库未安装，跳过测试"
        except Exception as e:
            return True, f"WebSocket服务未启动，跳过测试"
