#!/bin/bash

# VectorSphere平台启动脚本

set -e  # 遇到错误时退出

echo "VectorSphere智能平台启动脚本"
echo "=============================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3"
    exit 1
fi

# 检查pip
if ! command -v pip &> /dev/null; then
    echo "错误: 未找到pip"
    exit 1
fi

# 安装依赖
echo "安装Python依赖..."
pip install -r requirements.txt

# 初始化数据库
echo "初始化数据库..."
python3 init_database.py

# 启动应用
echo "启动VectorSphere平台..."
python3 app.py