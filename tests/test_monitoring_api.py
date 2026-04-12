"""测试统一监控API"""

import json
from flask import Flask
from backend.core.monitoring.api import monitoring_bp


def test_monitoring_api():
    """测试统一监控API"""
    print("开始测试统一监控API...")
    
    # 创建Flask应用并注册监控蓝图
    app = Flask(__name__)
    app.register_blueprint(monitoring_bp)
    
    # 测试客户端
    with app.test_client() as client:
        # 测试健康检查接口
        response = client.get('/api/monitoring/health')
        print(f"健康检查状态码: {response.status_code}")
        if response.status_code == 200:
            data = json.loads(response.data)
            print(f"健康检查结果: {data}")
        
        # 测试系统概览接口
        response = client.get('/api/monitoring/system-overview')
        print(f"系统概览状态码: {response.status_code}")
        if response.status_code == 200:
            data = json.loads(response.data)
            print(f"系统概览结果: {data}")
        
        # 测试当前指标接口
        response = client.get('/api/monitoring/metrics')
        print(f"当前指标状态码: {response.status_code}")
        if response.status_code == 200:
            data = json.loads(response.data)
            print(f"当前指标结果: {data}")
    
    print("API测试完成!")


if __name__ == "__main__":
    test_monitoring_api()