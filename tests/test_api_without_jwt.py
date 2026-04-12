#!/usr/bin/env python3
"""
测试API，临时禁用JWT认证
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from backend.api.training.training_jobs_api import training_jobs_bp
from backend.api.training.training_history_api import training_history_bp
from backend.api.training.model_evaluation_api import model_evaluation_bp
from backend.api.training.model_deployment_api import model_deployment_bp

# 创建测试应用
app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'test-secret-key'

# 临时禁用JWT认证装饰器
def mock_jwt_required():
    def decorator(f):
        return f
    return decorator

def mock_get_jwt_identity():
    return "test_user_123"

# 替换JWT函数
import backend.api.training.training_jobs_api as training_jobs_module
import backend.api.training.training_history_api as training_history_module
import backend.api.training.model_evaluation_api as model_evaluation_module
import backend.api.training.model_deployment_api as model_deployment_module

training_jobs_module.jwt_required = mock_jwt_required
training_jobs_module.get_jwt_identity = mock_get_jwt_identity
training_history_module.jwt_required = mock_jwt_required
training_history_module.get_jwt_identity = mock_get_jwt_identity
model_evaluation_module.jwt_required = mock_jwt_required
model_evaluation_module.get_jwt_identity = mock_get_jwt_identity
model_deployment_module.jwt_required = mock_jwt_required
model_deployment_module.get_jwt_identity = mock_get_jwt_identity

# 注册蓝图
app.register_blueprint(training_jobs_bp)
app.register_blueprint(training_history_bp)
app.register_blueprint(model_evaluation_bp)
app.register_blueprint(model_deployment_bp)

@app.route('/health')
def health():
    return {'status': 'healthy', 'service': 'Test API Server'}

if __name__ == '__main__':
    print("启动测试API服务器...")
    app.run(host='0.0.0.0', port=5002, debug=True)