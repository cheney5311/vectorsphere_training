#!/usr/bin/env python3
"""更新服务导入路径的脚本"""

import os
import re

# 定义要替换的导入路径
IMPORT_REPLACEMENTS = [
    # 训练服务
    (r'from backend\.modules\.training\.services\.training_service import', 'from backend.services.training_service import'),
    (r'from backend\.modules\.training\.services\.training_history_service import', 'from backend.services.training_history_service import'),
    (r'from backend\.modules\.training\.services\.enhanced_training_service import', 'from backend.services.enhanced_training_service import'),
    (r'from backend\.modules\.training\.services\.training_execution_service import', 'from backend.services.training_execution_service import'),
    (r'from backend\.modules\.training\.services\.training_statistics_service import', 'from backend.services.training_statistics_service import'),
    (r'from backend\.modules\.training\.services\.model_selection_service import', 'from backend.services.model_selection_service import'),
    (r'from backend\.modules\.training\.services\.model_evaluation_service import', 'from backend.services.model_evaluation_service import'),
    (r'from backend\.modules\.training\.services\.model_optimization_service import', 'from backend.services.model_optimization_service import'),
    (r'from backend\.modules\.training\.services\.model_deployment_service import', 'from backend.services.model_deployment_service import'),
    (r'from backend\.modules\.training\.services\.hyperparameter_optimization_service import', 'from backend.services.hyperparameter_optimization_service import'),
    (r'from backend\.modules\.training\.services\.intelligent_decision_service import', 'from backend.services.intelligent_decision_service import'),
    (r'from backend\.modules\.training\.services\.monitoring_operations_service import', 'from backend.services.monitoring_operations_service import'),
    
    # 数据集服务
    (r'from backend\.modules\.dataset\.services\.dataset_service import', 'from backend.services.dataset_service import'),
    (r'from backend\.modules\.dataset\.services\.data_discovery_service import', 'from backend.services.data_discovery_service import'),
    (r'from backend\.modules\.dataset\.services\.data_preprocessing_service import', 'from backend.services.data_preprocessing_service import'),
    (r'from backend\.modules\.dataset\.services\.data_quality_service import', 'from backend.services.data_quality_service import'),
    
    # 模型服务
    (r'from backend\.modules\.model\.services\.model_service import', 'from backend.services.model_service import'),
    
    # 认证服务
    (r'from backend\.modules\.auth\.services\.auth_service import', 'from backend.services.auth_service_module import'),
    (r'from backend\.modules\.auth\.services\.permission_service import', 'from backend.services.permission_service import'),
    
    # 智能体服务
    (r'from backend\.modules\.agent\.services\.agent_service import', 'from backend.services.agent_service import'),
    
    # 监控服务
    (r'from backend\.modules\.monitoring\.service import', 'from backend.services.monitoring_service import'),
    
    # 优化服务
    (r'from backend\.modules\.optimization\.services\.performance_analyzer import', 'from backend.services.performance_analyzer import'),
    (r'from backend\.modules\.optimization\.services\.resource_monitor import', 'from backend.services.resource_monitor import'),
    (r'from backend\.modules\.optimization\.services\.resource_optimizer import', 'from backend.services.resource_optimizer import'),
    
    # 性能服务
    (r'from backend\.modules\.performance\.services\.performance_monitor import', 'from backend.services.performance_monitor import'),
    (r'from backend\.modules\.performance\.services\.async_processor import', 'from backend.services.async_processor import'),
    (r'from backend\.modules\.performance\.services\.db_pool import', 'from backend.services.db_pool import'),
    
    # 调度服务
    (r'from backend\.modules\.scheduler\.services\.scheduler import', 'from backend.services.scheduler import'),
    
    # 安全服务
    (r'from backend\.modules\.security\.services\.access_control import', 'from backend.services.access_control import'),
    (r'from backend\.modules\.security\.services\.audit_logger import', 'from backend.services.audit_logger import'),
    (r'from backend\.modules\.security\.services\.auth_manager import', 'from backend.services.auth_manager import'),
    (r'from backend\.modules\.security\.services\.compliance_checker import', 'from backend.services.compliance_checker import'),
    (r'from backend\.modules\.security\.services\.encryption_service import', 'from backend.services.encryption_service import'),
    
    # 租户服务
    (r'from backend\.modules\.tenants\.services\.billing_service import', 'from backend.services.billing_service import'),
]


def update_file_imports(file_path):
    """更新单个文件的导入路径"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 应用所有替换
        for old_import, new_import in IMPORT_REPLACEMENTS:
            content = re.sub(old_import, new_import, content)
        
        # 如果内容有变化，则写回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated imports in {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False


def update_directory_imports(directory):
    """递归更新目录中所有Python文件的导入路径"""
    updated_files = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if update_file_imports(file_path):
                    updated_files += 1
    
    print(f"Updated imports in {updated_files} files")


if __name__ == "__main__":
    # 更新API目录中的导入路径
    api_directory = "/root/seetaSearch/VectorSphere-intelligent-platform/backend/api"
    if os.path.exists(api_directory):
        update_directory_imports(api_directory)
        print("API导入路径更新完成")
    
    # 更新其他可能使用服务的目录
    other_directories = [
        "/root/seetaSearch/VectorSphere-intelligent-platform/backend/modules",
        "/root/seetaSearch/VectorSphere-intelligent-platform/backend/core"
    ]
    
    for directory in other_directories:
        if os.path.exists(directory):
            update_directory_imports(directory)
    
    print("所有导入路径更新完成")