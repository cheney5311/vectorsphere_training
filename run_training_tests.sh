#!/bin/bash

# 训练API测试执行脚本

echo "========================================="
echo "  VectorSphere智能平台训练API测试脚本  "
echo "========================================="

# 进入项目目录
cd /root/seetaSearch/VectorSphere-intelligent-platform

# 运行模拟测试
echo ""
echo "1. 运行模拟API测试..."
python3 tests/comprehensive_training_api_test.py

if [ $? -eq 0 ]; then
    echo "模拟测试通过"
else
    echo "模拟测试失败"
    exit 1
fi

# 运行真实逻辑测试
echo ""
echo "2. 运行真实逻辑测试..."
python3 tests/real_training_api_test.py

if [ $? -eq 0 ]; then
    echo "真实逻辑测试通过"
else
    echo "真实逻辑测试失败"
    exit 1
fi

echo ""
echo "========================================="
echo "  所有测试完成！                         "
echo "========================================="
echo "测试报告已生成: TRAINING_API_TEST_REPORT.md"