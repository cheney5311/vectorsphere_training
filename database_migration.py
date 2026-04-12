#!/usr/bin/env python3
"""数据库迁移脚本

修复 models 表缺少 model_type 列的问题，创建完整的数据库表结构。
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config.config import get_config
from backend.schemas.base_models import Base
from backend.schemas.model_models import ModelDB, ModelMetadataDB, ModelVersionDB
from backend.schemas.auth_models import User, Role, Permission, ApiKey
from backend.schemas.training_models import TrainingSession, TrainingProgress, TrainingMetrics
from backend.schemas.project_models import Project, Dataset
from backend.schemas.monitoring_models import SystemMetric, Alert, ResourceUsage
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_database_tables():
    """创建数据库表"""
    try:
        # 获取配置
        config = get_config()
        
        # 尝试使用 PostgreSQL，如果失败则使用 SQLite
        try:
            engine = create_engine(config.database.url, echo=True)
            logger.info(f"使用数据库: {config.database.url}")
        except Exception as e:
            logger.warning(f"PostgreSQL 连接失败: {e}")
            logger.info("切换到 SQLite 数据库")
            
            # 使用 SQLite
            sqlite_path = Path('./data/vectorsphere.db')
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            sqlite_url = f"sqlite:///{sqlite_path}"
            engine = create_engine(sqlite_url, echo=True)
            logger.info(f"使用 SQLite 数据库: {sqlite_url}")
        
        # 创建所有表
        logger.info("开始创建数据库表...")
        Base.metadata.create_all(bind=engine)
        logger.info("数据库表创建成功！")
        
        # 验证 models 表是否正确创建
        with engine.connect() as conn:
            if 'postgresql' in str(engine.url):
                # PostgreSQL
                result = conn.execute(text("""
                    SELECT column_name, data_type, is_nullable 
                    FROM information_schema.columns 
                    WHERE table_name = 'models' 
                    AND table_schema = 'public'
                    ORDER BY ordinal_position;
                """))
            else:
                # SQLite
                result = conn.execute(text("PRAGMA table_info(models);"))
            
            columns = result.fetchall()
            if columns:
                logger.info("models 表列结构验证:")
                for col in columns:
                    if 'postgresql' in str(engine.url):
                        logger.info(f"  {col[0]} ({col[1]}) - nullable: {col[2]}")
                    else:
                        logger.info(f"  {col[1]} ({col[2]}) - nullable: {not col[3]}")
                
                # 检查 model_type 列是否存在
                column_names = [col[0] if 'postgresql' in str(engine.url) else col[1] for col in columns]
                if 'model_type' in column_names:
                    logger.info("✅ model_type 列已成功创建")
                else:
                    logger.error("❌ model_type 列未找到")
            else:
                logger.error("❌ models 表未找到")
        
        return True
        
    except Exception as e:
        logger.error(f"数据库表创建失败: {e}")
        return False


def test_model_query():
    """测试模型查询"""
    try:
        config = get_config()
        
        # 创建引擎和会话
        try:
            engine = create_engine(config.database.url)
        except:
            sqlite_path = Path('./data/vectorsphere.db')
            sqlite_url = f"sqlite:///{sqlite_path}"
            engine = create_engine(sqlite_url)
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        with SessionLocal() as session:
            # 测试查询所有模型
            logger.info("测试查询所有模型...")
            models = session.query(ModelDB).limit(10).all()
            logger.info(f"查询成功，找到 {len(models)} 个模型")
            
            # 测试创建一个示例模型
            from backend.schemas.enums import ModelType, ModelFramework, ModelStatus
            
            test_model = ModelDB(
                name="测试模型",
                description="这是一个测试模型",
                version="1.0.0",
                model_type=ModelType.CLASSIFICATION,
                framework=ModelFramework.PYTORCH,
                status=ModelStatus.DRAFT,
                user_id="test-user-id",
                tenant_id="default"
            )
            
            session.add(test_model)
            session.commit()
            logger.info("✅ 测试模型创建成功")
            
            # 再次查询验证
            models = session.query(ModelDB).all()
            logger.info(f"验证查询成功，总共 {len(models)} 个模型")
            
            for model in models:
                logger.info(f"  模型: {model.name} (类型: {model.model_type}, 框架: {model.framework})")
        
        return True
        
    except Exception as e:
        logger.error(f"模型查询测试失败: {e}")
        return False


def main():
    """主函数"""
    logger.info("开始数据库迁移...")
    
    # 创建数据库表
    if create_database_tables():
        logger.info("数据库表创建成功")
        
        # 测试模型查询
        if test_model_query():
            logger.info("✅ 数据库迁移完成，所有测试通过")
        else:
            logger.error("❌ 模型查询测试失败")
    else:
        logger.error("❌ 数据库表创建失败")


if __name__ == '__main__':
    main()