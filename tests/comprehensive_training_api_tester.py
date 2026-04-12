#!/usr/bin/env python3
"""
修复后的训练API测试脚本

测试所有训练相关API接口，验证真实逻辑，修复发现的问题
"""

import os
import sys
import json
import time
import requests
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingAPITester:
    """训练API测试器"""
    
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.auth_token = None
        self.test_results = {}
        self.errors_found = []
        self.mock_data_issues = []
        
        # 测试数据
        self.test_user_id = "test_user_" + str(uuid.uuid4())[:8]
        self.test_session_id = None
        self.test_job_id = None
        
    def setup_auth(self):
        """设置认证"""
        try:
            # 尝试获取JWT token
            auth_data = {
                "username": "test_user",
                "password": "test_password"
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/auth/login",
                json=auth_data
            )
            
            if response.status_code == 200:
                data = response.json()
                self.auth_token = data.get('access_token')
                self.session.headers.update({
                    'Authorization': f'Bearer {self.auth_token}'
                })
                logger.info("认证设置成功")
            else:
                # 如果认证失败，生成有效的测试token
                self.auth_token = self._generate_test_token()
                self.session.headers.update({
                    'Authorization': f'Bearer {self.auth_token}'
                })
                logger.warning("使用生成的测试认证token")
                
        except Exception as e:
            logger.error(f"认证设置失败: {e}")
            # 生成有效的测试token
            self.auth_token = self._generate_test_token()
            self.session.headers.update({
                'Authorization': f'Bearer {self.auth_token}'
            })
    
    def _generate_test_token(self):
        """生成测试用的JWT token"""
        try:
            import jwt
            
            # 使用与app.py相同的JWT配置
            JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
            JWT_ALGORITHM = 'HS256'
            
            # 使用Flask-JWT-Extended兼容的格式
            payload = {
                'sub': self.test_user_id,  # subject字段，Flask-JWT-Extended的标准字段
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(hours=24),
                'type': 'access',  # Flask-JWT-Extended的token类型
                'fresh': False,  # Flask-JWT-Extended的fresh标记
                'jti': str(uuid.uuid4())  # JWT ID，Flask-JWT-Extended要求
            }
            
            token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
            logger.info(f"JWT认证token已生成，用户ID: {self.test_user_id}")
            return token
            
        except Exception as e:
            logger.error(f"生成JWT token失败: {e}")
            # 如果JWT生成失败，返回简单的测试token
            return 'test_token_123'
    
    def test_api_endpoint(self, method: str, endpoint: str, data: Dict = None, 
                         expected_status: int = 200) -> Dict:
        """测试API端点"""
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == 'GET':
                response = self.session.get(url)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data)
            elif method.upper() == 'PUT':
                response = self.session.put(url, json=data)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")
            
            result = {
                'endpoint': endpoint,
                'method': method,
                'status_code': response.status_code,
                'success': response.status_code == expected_status,
                'response': response.text if response.text else None,  # 不截断响应
                'error': None
            }
            
            if not result['success']:
                self.errors_found.append(f"{method} {endpoint}: 状态码 {response.status_code}, 期望 {expected_status}")
            
            return result
            
        except Exception as e:
            error_msg = f"{method} {endpoint}: {str(e)}"
            self.errors_found.append(error_msg)
            return {
                'endpoint': endpoint,
                'method': method,
                'status_code': None,
                'success': False,
                'response': None,
                'error': str(e)
            }
    
    def test_training_api(self):
        """测试训练API"""
        logger.info("开始测试训练API...")
        
        # 首先创建一个训练会话
        create_session_data = {
            "name": "测试训练会话",
            "description": "API测试用的训练会话",
            "config": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001
            }
        }
        
        create_result = self.test_api_endpoint(
            'POST', 
            '/api/v1/training/sessions',
            data=create_session_data,
            expected_status=201
        )
        
        # 从创建结果中获取session_id
        if create_result['success'] and create_result['response']:
            try:
                response_data = json.loads(create_result['response'])
                logger.info(f"创建会话响应数据: {response_data}")
                
                # 尝试多种可能的字段名
                self.test_session_id = (
                    response_data.get('session_id') or 
                    response_data.get('id') or 
                    response_data.get('data', {}).get('session_id') or
                    response_data.get('data', {}).get('id')
                )
                
                if self.test_session_id:
                    logger.info(f"创建训练会话成功，session_id: {self.test_session_id}")
                else:
                    logger.error(f"无法从响应中提取session_id: {response_data}")
                    # 尝试从响应的根级别直接获取id字段
                    if isinstance(response_data, dict) and 'id' in response_data:
                        self.test_session_id = response_data['id']
                        logger.info(f"从根级别获取到session_id: {self.test_session_id}")
                    else:
                        self.test_session_id = None
            except Exception as e:
                logger.error(f"解析创建会话响应失败: {e}")
                logger.error(f"原始响应: {create_result['response']}")
                self.test_session_id = None
        else:
            logger.warning(f"创建训练会话失败: {create_result}")
            self.test_session_id = None
        
        # 测试训练API端点
        test_cases = [
            # 创建训练会话（已测试）
            {
                'name': '创建训练会话',
                'result': create_result
            },
            # 获取训练会话
            {
                'name': '获取训练会话',
                'result': self.test_api_endpoint('GET', f'/api/v1/training/sessions/{self.test_session_id}')
            },
            # 获取训练会话列表
            {
                'name': '获取训练会话列表',
                'result': self.test_api_endpoint('GET', '/api/v1/training/sessions')
            },
            # 开始训练会话
            {
                'name': '开始训练会话',
                'result': self.test_api_endpoint('POST', f'/api/v1/training/sessions/{self.test_session_id}/start')
            },
            # 获取训练进度
            {
                'name': '获取训练进度',
                'result': self.test_api_endpoint('GET', f'/api/v1/training/sessions/{self.test_session_id}/progress')
            },
            # 更新训练进度
            {
                'name': '更新训练进度',
                'result': self.test_api_endpoint('PUT', f'/api/v1/training/sessions/{self.test_session_id}/progress', {
                    "progress": 50.0,
                    "current_step": 100,
                    "total_steps": 200,
                    "train_loss": 0.5,
                    "eval_loss": 0.6,
                    "learning_rate": 1e-5
                })
            }
        ]
        
        # 记录测试结果
        self.test_results['training_api.py'] = {
            'total_tests': len(test_cases),
            'passed_tests': sum(1 for case in test_cases if case['result']['success']),
            'failed_tests': sum(1 for case in test_cases if not case['result']['success']),
            'test_cases': test_cases
        }
        
        logger.info(f"训练API测试完成: {self.test_results['training_api.py']['passed_tests']}/{self.test_results['training_api.py']['total_tests']} 通过")
    
    def test_training_execution_api(self):
        """测试训练执行API"""
        logger.info("开始测试训练执行API...")
        
        # 确保使用之前创建的真实session_id，而不是生成新的
        if not self.test_session_id:
            logger.warning("没有可用的session_id，跳过训练执行API测试")
            self.test_results['training_execution_api.py'] = {
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0,
                'test_cases': [],
                'error': '没有可用的session_id'
            }
            return
        
        test_cases = [
            # 启动训练执行
            {
                'name': '启动训练执行',
                'result': self.test_api_endpoint('POST', f'/api/v1/training/execution/sessions/{self.test_session_id}/start', {
                    "training_config": {
                        "epochs": 5,
                        "batch_size": 16,
                        "learning_rate": 0.001
                    }
                })
            },
            # 获取训练状态
            {
                'name': '获取训练状态',
                'result': self.test_api_endpoint('GET', f'/api/v1/training/execution/sessions/{self.test_session_id}/status')
            },
            # 暂停训练
            {
                'name': '暂停训练',
                'result': self.test_api_endpoint('POST', f'/api/v1/training/execution/sessions/{self.test_session_id}/pause')
            },
            # 恢复训练
            {
                'name': '恢复训练',
                'result': self.test_api_endpoint('POST', f'/api/v1/training/execution/sessions/{self.test_session_id}/resume')
            },
            # 停止训练
            {
                'name': '停止训练',
                'result': self.test_api_endpoint('POST', f'/api/v1/training/execution/sessions/{self.test_session_id}/stop')
            }
        ]
        
        # 记录测试结果
        self.test_results['training_execution_api.py'] = {
            'total_tests': len(test_cases),
            'passed_tests': sum(1 for case in test_cases if case['result']['success']),
            'failed_tests': sum(1 for case in test_cases if not case['result']['success']),
            'test_cases': test_cases
        }
        
        logger.info(f"训练执行API测试完成: {self.test_results['training_execution_api.py']['passed_tests']}/{self.test_results['training_execution_api.py']['total_tests']} 通过")
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("开始运行训练API测试...")
        
        # 设置认证
        self.setup_auth()
        
        # 运行测试
        self.test_training_api()
        self.test_training_execution_api()
        
        # 生成报告
        self.generate_report()
    
    def generate_report(self):
        """生成测试报告"""
        total_tests = sum(result.get('total_tests', 0) for result in self.test_results.values() if isinstance(result, dict))
        total_passed = sum(result.get('passed_tests', 0) for result in self.test_results.values() if isinstance(result, dict))
        total_failed = sum(result.get('failed_tests', 0) for result in self.test_results.values() if isinstance(result, dict))
        
        report = {
            'summary': {
                'total_apis_tested': len(self.test_results),
                'total_tests': total_tests,
                'passed_tests': total_passed,
                'failed_tests': total_failed,
                'success_rate': f"{(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "0%"
            },
            'api_results': self.test_results,
            'errors_found': self.errors_found,
            'mock_data_issues': self.mock_data_issues,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存报告到文件
        with open('comprehensive_training_api_test_report_fixed.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 打印摘要
        logger.info("=" * 60)
        logger.info("训练API测试报告")
        logger.info("=" * 60)
        logger.info(f"测试的API数量: {report['summary']['total_apis_tested']}")
        logger.info(f"总测试数: {report['summary']['total_tests']}")
        logger.info(f"通过测试数: {report['summary']['passed_tests']}")
        logger.info(f"失败测试数: {report['summary']['failed_tests']}")
        logger.info(f"成功率: {report['summary']['success_rate']}")
        logger.info(f"发现的错误数: {len(self.errors_found)}")
        logger.info("=" * 60)
        
        if self.errors_found:
            logger.info("发现的错误:")
            for error in self.errors_found:
                logger.error(f"  - {error}")

def main():
    """主函数"""
    logger.info("启动训练API测试...")
    
    tester = TrainingAPITester()
    tester.run_all_tests()
    
    logger.info("训练API测试完成!")

if __name__ == "__main__":
    main()