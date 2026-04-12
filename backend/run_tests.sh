#!/bin/bash

# 测试运行脚本
# 设置正确的Python路径以避免模块导入冲突

# 设置项目根目录
PROJECT_ROOT="/root/VectorSphere/VectorSphere-intelligent-platform"

# 设置Python路径
export PYTHONPATH="$PROJECT_ROOT"

# 切换到backend目录
cd "$PROJECT_ROOT/backend"

# 运行测试
echo "运行增强验证测试套件..."
echo "Python路径: $PYTHONPATH"
echo "当前目录: $(pwd)"
echo ""

# 运行所有测试
python -m pytest tests/test_enhanced_validation.py -v

echo ""
echo "测试完成！"