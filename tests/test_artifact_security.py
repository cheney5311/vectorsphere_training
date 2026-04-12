#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""安全工件与文件策略测试套件

测试安全工件管理、文件访问控制和权限管理功能。
"""

import os
import json
import tempfile
import shutil
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# 设置测试环境
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.services.artifact_security import (
    ArtifactSecurityService, SecurityPolicy, SecurityLevel, 
    ArtifactType, FileType, FileMetadata, SecurityScanResult
)
from backend.services.artifact_management import (
    ArtifactManagementService, Artifact, ArtifactVersion, 
    ArtifactStatus, ArtifactDependency
)
from backend.services.file_access_control import (
    FileAccessControlService, AccessType, PermissionResult,
    FilePermission, AccessRequest, AccessLog
)


class TestArtifactSecurityService(unittest.TestCase):
    """测试工件安全服务"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'artifact_security': {
                'storage_path': self.temp_dir,
                'max_file_size': 100 * 1024 * 1024,  # 100MB
                'allowed_extensions': ['.txt', '.json', '.py', '.md'],
                'virus_scan_enabled': False,  # 测试时禁用
                'encryption_enabled': False   # 测试时禁用
            }
        }
        self.service = ArtifactSecurityService(self.config)
        
        # 为测试用户分配角色
        from backend.modules.security.models import Role
        self.service.access_control.assign_role("test_user", Role.DEVELOPER)
        self.service.access_control.assign_role("user123", Role.DEVELOPER)
        self.service.access_control.assign_role("admin_user", Role.ADMIN)
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_file_upload(self):
        """测试文件上传验证"""
        # 创建测试文件
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test content')
        
        # 测试有效文件
        is_valid, error_msg, metadata = self.service.validate_file_upload(
            test_file, 'user123', ArtifactType.CONFIG_FILE, SecurityLevel.INTERNAL
        )
        self.assertTrue(is_valid)
        self.assertEqual(error_msg, "验证通过")
        self.assertIsNotNone(metadata)
        
        # 测试无效扩展名
        invalid_file = os.path.join(self.temp_dir, 'test.exe')
        with open(invalid_file, 'w') as f:
            f.write('test content')
        
        is_valid, error_msg, metadata = self.service.validate_file_upload(
            invalid_file, 'user123', ArtifactType.CONFIG_FILE, SecurityLevel.INTERNAL
        )
        self.assertFalse(is_valid)
        self.assertIn('不支持的文件类型', error_msg)
    
    def test_store_file_securely(self):
        """测试安全文件存储"""
        # 创建测试文件
        test_file = os.path.join(self.temp_dir, 'test.txt')
        test_content = 'test content for secure storage'
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # 先验证文件
        is_valid, error_msg, metadata = self.service.validate_file_upload(
            test_file, 'user123', ArtifactType.CONFIG_FILE, SecurityLevel.CONFIDENTIAL
        )
        self.assertTrue(is_valid)
        
        # 存储文件
        success, stored_path = self.service.store_secure_file(metadata, test_file)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(stored_path))
        
        # 验证文件元数据
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.original_name, 'test.txt')
        self.assertEqual(metadata.security_level, SecurityLevel.CONFIDENTIAL)
        self.assertEqual(metadata.owner_id, 'user123')
    
    def test_access_file(self):
        """测试文件访问"""
        # 先存储文件
        test_file = os.path.join(self.temp_dir, 'test.txt')
        test_content = 'test content for access'
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # 验证并存储文件
        is_valid, error_msg, metadata = self.service.validate_file_upload(
            test_file, 'user123', ArtifactType.CONFIG_FILE, SecurityLevel.INTERNAL
        )
        self.assertTrue(is_valid)
        
        success, stored_path = self.service.store_secure_file(metadata, test_file)
        self.assertTrue(success)
        
        # 访问文件
        success, error_msg, file_path = self.service.access_file(metadata.id, 'user123')
        self.assertTrue(success)
        self.assertEqual(error_msg, "访问允许")
        self.assertTrue(os.path.exists(file_path))
        
        # 读取文件内容验证
        with open(file_path, 'r') as f:
            content = f.read()
        self.assertEqual(content, test_content)
    
    def test_security_policies(self):
        """测试安全策略管理"""
        # 创建安全策略
        policy = SecurityPolicy(
            id='test_policy',
            name='测试策略',
            description='用于测试的安全策略',
            security_level=SecurityLevel.CONFIDENTIAL,
            allowed_file_types=['.txt', '.json'],
            max_file_size=50 * 1024 * 1024,
            encryption_required=True,
            virus_scan_required=True,
            access_control_enabled=True,
            audit_enabled=True,
            retention_days=90,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # 创建策略
        success = self.service.create_security_policy(policy)
        self.assertTrue(success)
        
        # 获取策略
        policies = self.service.get_security_policies()
        self.assertGreater(len(policies), 0)
        
        # 查找创建的策略
        created_policy = next((p for p in policies if p.id == 'test_policy'), None)
        self.assertIsNotNone(created_policy)
        self.assertEqual(created_policy.name, '测试策略')
        
        # 更新策略
        update_data = {'name': '更新的测试策略', 'max_file_size': 100 * 1024 * 1024}
        success = self.service.update_security_policy('test_policy', update_data)
        self.assertTrue(success)
        
        # 验证更新
        policies = self.service.get_security_policies()
        updated_policy = next((p for p in policies if p.id == 'test_policy'), None)
        self.assertEqual(updated_policy.name, '更新的测试策略')
        self.assertEqual(updated_policy.max_file_size, 100 * 1024 * 1024)


class TestArtifactManagementService(unittest.TestCase):
    """测试工件管理服务"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'artifact_management': {
                'storage_path': self.temp_dir
            },
            'artifact_security': {
                'storage_path': self.temp_dir,
                'max_file_size': 100 * 1024 * 1024,
                'allowed_extensions': ['.txt', '.json', '.py', '.md'],
                'virus_scan_enabled': False,
                'encryption_enabled': False
            }
        }
        
        # 创建安全服务
        self.security_service = ArtifactSecurityService(self.config)
        self.service = ArtifactManagementService(self.config, self.security_service)
        
        # 为测试用户分配角色
        from backend.modules.security.models import Role
        self.security_service.access_control.assign_role("test_user", Role.DEVELOPER)
        self.security_service.access_control.assign_role("user123", Role.DEVELOPER)
        self.security_service.access_control.assign_role("admin_user", Role.ADMIN)
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_artifact(self):
        """测试工件创建"""
        success, error_msg, artifact = self.service.create_artifact(
            name='测试工件',
            description='这是一个测试工件',
            artifact_type=ArtifactType.MODEL_FILE,
            security_level=SecurityLevel.INTERNAL,
            owner_id='user123',
            tags=['test', 'model'],
            metadata={'framework': 'pytorch'}
        )
        
        self.assertTrue(success)
        self.assertEqual(error_msg, "创建成功")
        self.assertIsNotNone(artifact)
        self.assertEqual(artifact.name, '测试工件')
        self.assertEqual(artifact.artifact_type, ArtifactType.MODEL_FILE)
        self.assertEqual(artifact.security_level, SecurityLevel.INTERNAL)
        self.assertEqual(artifact.owner_id, 'user123')
        self.assertIn('test', artifact.tags)
        self.assertEqual(artifact.metadata['framework'], 'pytorch')
    
    def test_upload_artifact_version(self):
        """测试工件版本上传"""
        # 先创建工件
        success, error_msg, artifact = self.service.create_artifact(
            name='测试工件',
            description='这是一个测试工件',
            artifact_type=ArtifactType.MODEL_FILE,
            security_level=SecurityLevel.INTERNAL,
            owner_id='user123'
        )
        self.assertTrue(success)
        
        # 创建测试文件
        test_file = os.path.join(self.temp_dir, 'model.json')
        with open(test_file, 'w') as f:
            f.write('{"model": "test model data"}')
        
        # 上传版本
        success, error_msg, version = self.service.upload_artifact_version(
            artifact_id=artifact.id,
            file_path=test_file,
            version='1.0.0',
            changelog='初始版本',
            user_id='user123'
        )
        
        if not success:
            print(f"Upload failed with error: {error_msg}")
        self.assertTrue(success)
        self.assertEqual(error_msg, "上传成功")
        self.assertIsNotNone(version)
        self.assertEqual(version.version, '1.0.0')
        self.assertEqual(version.changelog, '初始版本')
        self.assertEqual(version.created_by, 'user123')
        self.assertIsInstance(version.tags, list)
    
    def test_download_artifact_version(self):
        """测试下载工件版本"""
        # 创建工件并上传版本
        success, error_msg, artifact = self.service.create_artifact(
            name='测试工件',
            description='这是一个测试工件',
            artifact_type=ArtifactType.MODEL_FILE,
            security_level=SecurityLevel.INTERNAL,
            owner_id='user123'
        )
        self.assertTrue(success)
        
        test_file = os.path.join(self.temp_dir, 'model.txt')
        test_content = 'model content v1.0'
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        success, error_msg, version = self.service.upload_artifact_version(
            artifact_id=artifact.id,
            file_path=test_file,
            version='1.0.0',
            changelog='初始版本',
            user_id='user123'
        )
        self.assertTrue(success)
        
        # 下载版本
        success, error_msg, download_path = self.service.download_artifact_version(
            artifact.id, '1.0.0', 'user123'
        )
        
        self.assertTrue(success)
        self.assertEqual(error_msg, "下载成功")
        self.assertTrue(os.path.exists(download_path))
        
        # 验证文件内容
        with open(download_path, 'r') as f:
            content = f.read()
        self.assertEqual(content, test_content)
    
    def test_artifact_dependencies(self):
        """测试工件依赖管理"""
        # 创建两个工件
        success1, _, artifact1 = self.service.create_artifact(
            name='工件1', description='', artifact_type=ArtifactType.MODEL_FILE,
            security_level=SecurityLevel.INTERNAL, owner_id='user123'
        )
        success2, _, artifact2 = self.service.create_artifact(
            name='工件2', description='', artifact_type=ArtifactType.CONFIG_FILE,
            security_level=SecurityLevel.INTERNAL, owner_id='user123'
        )
        self.assertTrue(success1 and success2)
        
        # 添加依赖
        success, error_msg = self.service.add_dependency(
            artifact1.id, artifact2.id, 'required', '>=1.0.0', 'user123'
        )
        self.assertTrue(success)
        self.assertEqual(error_msg, "添加依赖成功")
        
        # 获取依赖
        dependencies = self.service.get_artifact_dependencies(artifact1.id)
        self.assertEqual(len(dependencies), 1)
        self.assertEqual(dependencies[0].target_artifact_id, artifact2.id)
        self.assertEqual(dependencies[0].dependency_type, 'required')
        
        # 获取被依赖
        dependents = self.service.get_artifact_dependents(artifact2.id)
        self.assertEqual(len(dependents), 1)
        self.assertEqual(dependents[0].source_artifact_id, artifact1.id)
        
        # 移除依赖
        success, error_msg = self.service.remove_dependency(
            artifact1.id, artifact2.id, 'user123'
        )
        self.assertTrue(success)
        
        # 验证依赖已移除
        dependencies = self.service.get_artifact_dependencies(artifact1.id)
        self.assertEqual(len(dependencies), 0)


class TestFileAccessControlService(unittest.TestCase):
    """测试文件访问控制服务"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'file_access_control': {
                'storage_path': self.temp_dir,
                'max_logs': 1000
            },
            'access_control': {
                'storage_path': self.temp_dir
            }
        }
        
        # Mock AccessControlManager
        with patch('backend.services.file_access_control.AccessControlManager'):
            self.service = FileAccessControlService(self.config)
            
        # 设置mock方法
        self.service.access_control.get_user_roles = Mock(return_value=['user'])
        self.service.access_control.get_user_permissions = Mock(return_value=[])
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_check_file_permission_explicit(self):
        """测试显式权限检查"""
        # 授予权限
        success = self.service.grant_file_permission(
            user_id='user123',
            file_path='/test/file.txt',
            access_types=[AccessType.READ, AccessType.WRITE],
            granted_by='admin'
        )
        self.assertTrue(success)
        
        # 检查读权限
        has_permission, reason = self.service.check_file_permission(
            'user123', '/test/file.txt', AccessType.READ
        )
        self.assertTrue(has_permission)
        self.assertEqual(reason, '显式权限')
        
        # 检查写权限
        has_permission, reason = self.service.check_file_permission(
            'user123', '/test/file.txt', AccessType.WRITE
        )
        self.assertTrue(has_permission)
        
        # 检查删除权限（未授予）
        has_permission, reason = self.service.check_file_permission(
            'user123', '/test/file.txt', AccessType.DELETE
        )
        self.assertFalse(has_permission)
        self.assertEqual(reason, '访问类型不匹配')
    
    def test_check_file_permission_expired(self):
        """测试过期权限检查"""
        # 授予过期权限
        expired_time = datetime.now() - timedelta(hours=1)
        success = self.service.grant_file_permission(
            user_id='user123',
            file_path='/test/file.txt',
            access_types=[AccessType.READ],
            granted_by='admin',
            expires_at=expired_time
        )
        self.assertTrue(success)
        
        # 检查权限（应该被拒绝）
        has_permission, reason = self.service.check_file_permission(
            'user123', '/test/file.txt', AccessType.READ
        )
        self.assertFalse(has_permission)
        self.assertEqual(reason, '权限已过期')
    
    def test_check_file_permission_with_conditions(self):
        """测试条件权限检查"""
        # 授予带时间条件的权限
        conditions = {
            'time_range': {
                'start': '09:00',
                'end': '18:00'
            },
            'allowed_ips': ['192.168.1.100', '10.0.0.1']
        }
        
        success = self.service.grant_file_permission(
            user_id='user123',
            file_path='/test/file.txt',
            access_types=[AccessType.READ],
            granted_by='admin',
            conditions=conditions
        )
        self.assertTrue(success)
        
        # 在允许的IP地址访问
        with patch('backend.services.file_access_control.datetime') as mock_datetime:
            # 设置当前时间为工作时间
            mock_datetime.now.return_value = datetime(2023, 1, 1, 10, 0, 0)
            mock_datetime.strptime = datetime.strptime
            
            has_permission, reason = self.service.check_file_permission(
                'user123', '/test/file.txt', AccessType.READ, ip_address='192.168.1.100'
            )
            self.assertTrue(has_permission)
        
        # 在不允许的IP地址访问
        with patch('backend.services.file_access_control.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 10, 0, 0)
            mock_datetime.strptime = datetime.strptime
            
            has_permission, reason = self.service.check_file_permission(
                'user123', '/test/file.txt', AccessType.READ, ip_address='192.168.1.200'
            )
            self.assertFalse(has_permission)
            self.assertEqual(reason, '不满足访问条件')
    
    def test_file_owner_permission(self):
        """测试文件所有者权限"""
        # 测试所有者权限
        has_permission, reason = self.service.check_file_permission(
            'user123', '/users/user123/file.txt', AccessType.READ
        )
        self.assertTrue(has_permission)
        self.assertEqual(reason, '文件所有者')
    
        # 测试非所有者 - 使用不同的路径
        has_permission, reason = self.service.check_file_permission(
            'user456', '/restricted/secret/file.txt', AccessType.READ
        )
        # 这个应该根据默认权限来判断，不是文件所有者
        # restricted 级别的文件，普通用户没有默认权限，应该被拒绝
        self.assertFalse(has_permission)
    
    def test_permission_management(self):
        """测试权限管理"""
        # 授予权限
        success = self.service.grant_file_permission(
            user_id='user123',
            file_path='/test/file.txt',
            access_types=[AccessType.READ, AccessType.WRITE],
            granted_by='admin'
        )
        self.assertTrue(success)
        
        # 列出用户权限
        permissions = self.service.list_user_permissions('user123')
        self.assertEqual(len(permissions), 1)
        self.assertEqual(permissions[0].file_path, '/test/file.txt')
        
        # 列出文件权限
        permissions = self.service.list_file_permissions('/test/file.txt')
        self.assertEqual(len(permissions), 1)
        self.assertEqual(permissions[0].user_id, 'user123')
        
        # 撤销权限
        success = self.service.revoke_file_permission('user123', '/test/file.txt')
        self.assertTrue(success)
        
        # 验证权限已撤销
        permissions = self.service.list_user_permissions('user123')
        self.assertEqual(len(permissions), 0)
    
    def test_access_logging(self):
        """测试访问日志"""
        # 执行一些权限检查以生成日志
        self.service.check_file_permission('user123', '/test/file1.txt', AccessType.READ)
        self.service.check_file_permission('user123', '/test/file2.txt', AccessType.WRITE)
        self.service.check_file_permission('user456', '/test/file1.txt', AccessType.READ)
        
        # 获取所有日志
        logs = self.service.get_access_logs()
        self.assertGreaterEqual(len(logs), 3)
        
        # 按用户过滤
        user_logs = self.service.get_access_logs(user_id='user123')
        self.assertGreaterEqual(len(user_logs), 2)
        
        # 按文件过滤
        file_logs = self.service.get_access_logs(file_path='/test/file1.txt')
        self.assertGreaterEqual(len(file_logs), 2)
    
    def test_cleanup_expired_permissions(self):
        """测试清理过期权限"""
        # 添加一些权限，包括过期的
        expired_time = datetime.now() - timedelta(hours=1)
        future_time = datetime.now() + timedelta(hours=1)
        
        # 过期权限
        self.service.grant_file_permission(
            'user123', '/test/expired.txt', [AccessType.READ], 'admin', expired_time
        )
        
        # 有效权限
        self.service.grant_file_permission(
            'user123', '/test/valid.txt', [AccessType.READ], 'admin', future_time
        )
        
        # 永久权限
        self.service.grant_file_permission(
            'user123', '/test/permanent.txt', [AccessType.READ], 'admin'
        )
        
        # 清理过期权限
        cleaned_count = self.service.cleanup_expired_permissions()
        self.assertEqual(cleaned_count, 1)
        
        # 验证只有有效权限保留
        permissions = self.service.list_user_permissions('user123')
        self.assertEqual(len(permissions), 2)
        
        file_paths = [p.file_path for p in permissions]
        self.assertIn('/test/valid.txt', file_paths)
        self.assertIn('/test/permanent.txt', file_paths)
        self.assertNotIn('/test/expired.txt', file_paths)
    
    def test_permission_statistics(self):
        """测试权限统计"""
        # 添加一些权限和日志
        self.service.grant_file_permission(
            'user123', '/test/file1.txt', [AccessType.READ], 'admin'
        )
        self.service.grant_file_permission(
            'user456', '/test/file2.txt', [AccessType.READ, AccessType.WRITE], 'admin'
        )
        
        # 执行一些访问
        self.service.check_file_permission('user123', '/test/file1.txt', AccessType.READ)
        self.service.check_file_permission('user456', '/test/file3.txt', AccessType.DELETE)
        
        # 获取统计信息
        stats = self.service.get_permission_statistics()
        
        self.assertIn('permissions', stats)
        self.assertIn('access_logs', stats)
        
        # 验证权限统计
        perm_stats = stats['permissions']
        self.assertGreaterEqual(perm_stats['total'], 2)
        self.assertGreaterEqual(perm_stats['active'], 2)
        
        # 验证访问日志统计
        log_stats = stats['access_logs']
        self.assertGreaterEqual(log_stats['total'], 2)


if __name__ == '__main__':
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    unittest.main(verbosity=2)