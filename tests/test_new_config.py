"""测试新重构的配置模块"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_new_config():
    """测试新配置模块"""
    print("开始测试新重构的配置模块...")
    
    try:
        # 导入新模块
        from backend_new.config import (
            DatabaseConfig,
            RedisConfig,
            JWTConfig,
            SecurityConfig,
            EmailConfig,
            StorageConfig,
            LoggingConfig,
            MonitoringConfig,
            DistributedConfig,
            TrainingConfig,
            APIConfig,
            TenantConfig,
            Config,
            get_config,
            reload_config,
            get_database_url,
            get_redis_url,
            is_development,
            is_production,
            is_testing,
            ConfigSource,
            ConfigValidationError,
            ConfigMetadata,
            ConfigObserver,
            OptimizedConfigManager,
            DynamicTrainingConfig,
            TrainingConfigObserver,
            get_config_manager,
            load_training_config,
            update_training_config
        )
        
        print("✓ 模块导入成功")
        
        # 测试基础配置类
        db_config = DatabaseConfig()
        print(f"✓ 数据库配置创建成功: {db_config.db_type}")
        
        redis_config = RedisConfig()
        print(f"✓ Redis配置创建成功: {redis_config.host}:{redis_config.port}")
        
        jwt_config = JWTConfig()
        print(f"✓ JWT配置创建成功: {jwt_config.algorithm}")
        
        # 测试主配置类
        config = Config()
        print("✓ 主配置类创建成功")
        
        # 测试配置获取函数
        main_config = get_config()
        print(f"✓ 获取主配置成功: {main_config.app_name}")
        
        # 测试数据库URL获取
        db_url = get_database_url()
        print(f"✓ 数据库URL获取成功: {db_url}")
        
        # 测试Redis URL获取
        redis_url = get_redis_url()
        print(f"✓ Redis URL获取成功: {redis_url}")
        
        # 测试环境判断函数
        dev_env = is_development()
        prod_env = is_production()
        test_env = is_testing()
        print(f"✓ 环境判断函数测试成功: dev={dev_env}, prod={prod_env}, test={test_env}")
        
        # 测试优化配置管理器
        config_manager = get_config_manager()
        print("✓ 配置管理器获取成功")
        
        # 测试动态训练配置
        dynamic_config = DynamicTrainingConfig()
        print(f"✓ 动态训练配置创建成功: batch_size={dynamic_config.batch_size}")
        
        # 测试配置管理器功能
        config_manager.set_config("test_key", "test_value", ConfigSource.ENVIRONMENT)
        test_value = config_manager.get_config("test_key")
        print(f"✓ 配置管理器设置和获取测试成功: {test_value}")
        
        # 测试训练配置加载
        training_config = load_training_config()
        print(f"✓ 训练配置加载成功: model_name={training_config.model_name}")
        
        print("\n🎉 新配置模块测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ 新配置模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_new_config()
    if success:
        print("\n✅ 所有测试通过!")
    else:
        print("\n❌ 测试失败!")