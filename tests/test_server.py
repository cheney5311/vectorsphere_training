#!/usr/bin/env python3
"""简单的测试服务器

用于验证基本的Flask功能和工件安全API。
"""

import os
import sys
from flask import Flask, jsonify

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_app():
    """创建测试Flask应用"""
    app = Flask(__name__)
    
    # 基本配置
    app.config['SECRET_KEY'] = 'test-secret-key'
    app.config['TESTING'] = True
    
    @app.route('/health')
    def health():
        return jsonify({
            'status': 'healthy',
            'service': 'VectorSphere Test Server',
            'message': '服务运行正常'
        })
    
    @app.route('/api/test')
    def test_api():
        return jsonify({
            'message': 'API测试成功',
            'timestamp': '2024-01-01T00:00:00Z'
        })
    
    # 尝试导入工件安全服务
    try:
        from backend.services.artifact_security import ArtifactSecurityService
        from backend.services.access_control import AccessControlManager
        
        @app.route('/api/security/test')
        def test_security():
            try:
                # 创建工件安全服务配置
                security_config = {
                    'storage_path': '/tmp/vectorsphere/artifacts',
                    'max_file_size': 100 * 1024 * 1024,  # 100MB
                    'allowed_extensions': ['.txt', '.pdf', '.jpg', '.png'],
                    'access_control': {
                        'default_role': 'user',
                        'admin_role': 'admin',
                        'enable_audit': True
                    }
                }
                security_service = ArtifactSecurityService(security_config)
                
                return jsonify({
                    'message': '工件安全服务测试成功',
                    'security_service': 'available',
                    'access_control': 'available'
                })
            except Exception as e:
                return jsonify({
                    'message': '工件安全服务测试失败',
                    'error': str(e)
                }), 500
                
    except ImportError as e:
        @app.route('/api/security/test')
        def test_security_fallback():
            return jsonify({
                'message': '工件安全服务不可用',
                'error': f'导入失败: {str(e)}'
            }), 503
    
    return app

if __name__ == '__main__':
    app = create_test_app()
    print("启动VectorSphere测试服务器...")
    print("健康检查: http://localhost:5000/health")
    print("API测试: http://localhost:5000/api/test")
    print("安全测试: http://localhost:5000/api/security/test")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )