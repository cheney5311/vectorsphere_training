#!/usr/bin/env python3
"""工件安全功能测试脚本

测试访问控制、文件验证、安全扫描等功能
"""

import os
import sys
import json
import tempfile
import requests
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.services.artifact_security import ArtifactSecurityService, SecurityLevel, ArtifactType
from backend.services.access_control import AccessControlManager

def test_access_control():
    """测试访问控制功能"""
    print("🔐 测试访问控制功能...")
    
    # 创建访问控制管理器
    config = {
        'default_role': 'user',
        'admin_role': 'admin',
        'enable_audit': True
    }
    
    access_manager = AccessControlManager(config)
    
    # 测试权限检查
    result = access_manager.check_permission(
        user_id='test_user',
        resource='artifact:upload',
        action='create'
    )
    
    print(f"  ✓ 权限检查结果: {result.allowed}")
    print(f"  ✓ 权限检查原因: {result.reason}")
    
    return True

def test_artifact_security_service():
    """测试工件安全服务"""
    print("🛡️ 测试工件安全服务...")
    
    try:
        # 创建配置
        security_config = {
            'storage_path': '/tmp/vectorsphere/test_artifacts',
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'allowed_extensions': ['.txt', '.pdf', '.jpg', '.png', '.py'],
            'access_control': {
                'default_role': 'user',
                'admin_role': 'admin',
                'enable_audit': True
            }
        }
        
        # 创建服务实例
        security_service = ArtifactSecurityService(security_config)
        
        # 为访问控制管理器添加基本策略
        try:
            from backend.modules.security.models import AccessPolicy
            from datetime import datetime
            
            # 添加允许上传的策略
            upload_policy = AccessPolicy(
                id="allow_upload",
                name="允许上传",
                description="允许用户上传文件",
                effect="allow",
                principals=["test_user"],
                resources=["artifact:*"],
                actions=["upload"],
                conditions={},
                priority=100,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            security_service.access_control.access_policies[upload_policy.id] = upload_policy
            
            # 为测试用户分配角色
            from backend.modules.security.models import Role
            security_service.access_control.assign_role("test_user", Role.DEVELOPER)
            
        except Exception as e:
            print(f"  ⚠️ 添加访问控制策略失败: {e}")
    except Exception as e:
        print(f"  ⚠️ 创建安全服务失败: {e}")
        return False
    
    # 创建测试文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("这是一个测试文件，用于验证工件安全功能。")
        test_file_path = f.name
    
    try:
        # 测试文件验证
        is_valid, error_msg, metadata = security_service.validate_file_upload(
            file_path=test_file_path,
            user_id='test_user',
            artifact_type=ArtifactType.TRAINING_DATA,
            security_level=SecurityLevel.INTERNAL
        )
        
        print(f"  ✓ 文件验证结果: {is_valid}")
        if not is_valid:
            print(f"  ✗ 验证错误: {error_msg}")
        else:
            print(f"  ✓ 文件元数据: {metadata.original_name if metadata else 'None'}")
        
        # 测试安全策略
        policies = security_service.get_security_policies()
        print(f"  ✓ 安全策略数量: {len(policies)}")
        
        return is_valid
        
    finally:
        # 清理测试文件
        os.unlink(test_file_path)

def test_api_endpoints():
    """测试API端点"""
    print("🌐 测试API端点...")
    
    base_url = "http://localhost:5000"
    
    # 测试健康检查
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ 健康检查: {data['status']}")
        else:
            print(f"  ✗ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ 健康检查异常: {e}")
        return False
    
    # 测试安全API
    try:
        response = requests.get(f"{base_url}/api/security/test", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ 安全API: {data['message']}")
            print(f"  ✓ 安全服务状态: {data['security_service']}")
            print(f"  ✓ 访问控制状态: {data['access_control']}")
        else:
            print(f"  ✗ 安全API失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ 安全API异常: {e}")
        return False
    
    return True

def test_file_operations():
    """测试文件操作"""
    print("📁 测试文件操作...")
    
    # 创建测试目录
    test_dir = Path('/tmp/vectorsphere/test_artifacts')
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建测试文件
    test_files = [
        ('test_document.txt', '这是一个测试文档'),
        ('test_script.py', 'print("Hello, VectorSphere!")'),
        ('test_config.json', '{"test": true, "version": "1.0"}')
    ]
    
    created_files = []
    
    try:
        for filename, content in test_files:
            file_path = test_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            created_files.append(file_path)
            print(f"  ✓ 创建测试文件: {filename}")
        
        # 验证文件存在
        for file_path in created_files:
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"  ✓ 文件验证: {file_path.name} ({size} bytes)")
            else:
                print(f"  ✗ 文件不存在: {file_path.name}")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ 文件操作异常: {e}")
        return False
    
    finally:
        # 清理测试文件
        for file_path in created_files:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                print(f"  ⚠️ 清理文件失败: {file_path.name} - {e}")

def main():
    """主测试函数"""
    print("🚀 开始VectorSphere工件安全功能测试")
    print("=" * 50)
    
    tests = [
        ("访问控制", test_access_control),
        ("工件安全服务", test_artifact_security_service),
        ("API端点", test_api_endpoints),
        ("文件操作", test_file_operations)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}测试:")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {status}")
        except Exception as e:
            print(f"  ❌ 异常: {e}")
            results.append((test_name, False))
    
    # 输出测试总结
    print("\n" + "=" * 50)
    print("📊 测试总结:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {test_name}")
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！VectorSphere工件安全功能正常运行。")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查相关功能。")
        return 1

if __name__ == '__main__':
    sys.exit(main())