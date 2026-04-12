#!/usr/bin/env python3
"""测试仪表盘API接口"""

import sys
import os
import requests
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置
BASE_URL = "http://localhost:5000"
API_PREFIX = "/api/v1/user/training"

def test_api_endpoints():
    """测试所有仪表盘API端点"""
    print("开始测试仪表盘API接口...")
    
    # 1. 测试获取用户训练概览
    print("\n1. 测试获取用户训练概览...")
    try:
        response = requests.get(f"{BASE_URL}{API_PREFIX}/overview")
        print(f"   状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   响应数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"   错误: {response.text}")
    except Exception as e:
        print(f"   请求失败: {e}")
    
    # 2. 测试获取最近训练会话
    print("\n2. 测试获取最近训练会话...")
    try:
        response = requests.get(f"{BASE_URL}{API_PREFIX}/recent-sessions?limit=3")
        print(f"   状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   响应数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"   错误: {response.text}")
    except Exception as e:
        print(f"   请求失败: {e}")
    
    # 3. 测试获取训练会话列表
    print("\n3. 测试获取训练会话列表...")
    try:
        response = requests.get(f"{BASE_URL}{API_PREFIX}/sessions?page=1&limit=5")
        print(f"   状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   响应数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"   错误: {response.text}")
    except Exception as e:
        print(f"   请求失败: {e}")
    
    # 4. 测试获取训练统计信息
    print("\n4. 测试获取训练统计信息...")
    try:
        response = requests.get(f"{BASE_URL}{API_PREFIX}/statistics")
        print(f"   状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   响应数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"   错误: {response.text}")
    except Exception as e:
        print(f"   请求失败: {e}")

def generate_api_documentation():
    """生成API文档"""
    print("\n生成API文档...")
    
    api_docs = {
        "title": "VectorSphere 训练平台仪表盘API文档",
        "version": "1.0.0",
        "description": "为前端仪表盘提供训练数据的API接口",
        "endpoints": [
            {
                "path": "/api/v1/user/training/overview",
                "method": "GET",
                "description": "获取用户训练概览信息",
                "parameters": [],
                "response": {
                    "activeSessions": "integer",
                    "completedSessions": "integer", 
                    "totalModels": "integer",
                    "avgAccuracy": "float"
                }
            },
            {
                "path": "/api/v1/user/training/recent-sessions",
                "method": "GET",
                "description": "获取用户最近的训练会话",
                "parameters": [
                    {"name": "limit", "type": "integer", "required": False, "description": "限制数量 (默认: 5)"}
                ],
                "response": {
                    "recentSessions": [
                        {
                            "id": "string",
                            "model_name": "string", 
                            "status": "string",
                            "progress": "integer",
                            "started_at": "string",
                            "completed_at": "string"
                        }
                    ]
                }
            },
            {
                "path": "/api/v1/user/training/sessions",
                "method": "GET",
                "description": "获取用户所有训练会话（支持分页）",
                "parameters": [
                    {"name": "page", "type": "integer", "required": False, "description": "页码 (默认: 1)"},
                    {"name": "limit", "type": "integer", "required": False, "description": "每页数量 (默认: 10)"},
                    {"name": "status", "type": "string", "required": False, "description": "状态过滤"}
                ],
                "response": {
                    "sessions": [
                        {
                            "id": "string",
                            "name": "string",
                            "modelType": "string",
                            "status": "string",
                            "progress": "integer",
                            "startTime": "string",
                            "endTime": "string"
                        }
                    ],
                    "total": "integer",
                    "page": "integer", 
                    "limit": "integer"
                }
            },
            {
                "path": "/api/v1/user/training/statistics",
                "method": "GET",
                "description": "获取用户训练统计信息",
                "parameters": [],
                "response": {
                    "totalTasks": "integer",
                    "completedTasks": "integer",
                    "runningTasks": "integer", 
                    "failedTasks": "integer",
                    "successRate": "float",
                    "avgTrainingTime": "float"
                }
            }
        ]
    }
    
    # 保存API文档
    with open("dashboard_api_docs.json", "w", encoding="utf-8") as f:
        json.dump(api_docs, f, indent=2, ensure_ascii=False)
    
    print("API文档已保存到 dashboard_api_docs.json")
    return api_docs

if __name__ == "__main__":
    # 生成API文档
    docs = generate_api_documentation()
    
    # 如果需要测试API，取消下面的注释
    # test_api_endpoints()
    
    print("\n任务完成！")