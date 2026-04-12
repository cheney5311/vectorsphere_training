#!/usr/bin/env python3
"""
调试启动训练API的错误
"""

import requests
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_test_token import generate_test_token

def debug_start_training():
    """调试启动训练API"""
    base_url = "http://localhost:5001"
    session_id = "c1b5cf84-d765-453a-b2ae-19d162c6500d"
    user_id = "test_user_debug"
    
    # 生成JWT token (注意：generate_test_token不接受参数，使用硬编码的用户ID)
    token = generate_test_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # 测试启动训练
    url = f"{base_url}/api/v1/training/sessions/{session_id}/start"
    print(f"请求URL: {url}")
    print(f"请求头: {headers}")
    
    response = requests.post(url, headers=headers)
    print(f"响应状态码: {response.status_code}")
    print(f"响应头: {dict(response.headers)}")
    
    try:
        response_data = response.json()
        print(f"响应内容: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
    except:
        print(f"响应文本: {response.text}")

if __name__ == "__main__":
    debug_start_training()