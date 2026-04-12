#!/usr/bin/env python3
"""测试 models 表查询的脚本"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("开始测试 models 表查询...")

try:
    from backend.schemas.model_models import ModelDB
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    print("导入模块成功")
    
    # 直接使用正确的数据库 URL
    database_url = "postgresql://postgres:password@localhost:5432/vectorsphere"
    print(f"数据库 URL: {database_url}")
    
    # 创建数据库引擎
    engine = create_engine(database_url, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # 测试查询
    with SessionLocal() as session:
        print("测试查询所有模型...")
        
        # 这是原始错误中的查询
        models = session.query(ModelDB).limit(1000).offset(0).all()
        print(f"✅ 查询成功！找到 {len(models)} 个模型")
        
        # 测试特定的 model_type 查询
        print("测试 model_type 列查询...")
        result = session.query(ModelDB.model_type).limit(5).all()
        print(f"✅ model_type 列查询成功！结果: {result}")
        
        # 测试完整的模型属性访问
        if models:
            print("测试模型属性访问...")
            for i, model in enumerate(models[:3]):  # 只测试前3个
                print(f"  模型 {i+1}:")
                print(f"    名称: {model.name}")
                print(f"    描述: {model.description}")
                print(f"    版本: {model.version}")
                print(f"    模型类型: {model.model_type}")
                print(f"    框架: {model.framework}")
                print(f"    状态: {model.status}")
        
        print("✅ 所有查询测试通过！")
        
except Exception as e:
    print(f"❌ 查询测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("✅ models 表查询修复验证完成！")