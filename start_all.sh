#!/bin/bash

# VectorSphere平台完整启动脚本

set -e  # 遇到错误时退出

echo "VectorSphere智能平台完整启动脚本"
echo "=================================="

# 检查是否在Linux系统上运行
if [[ "$OSTYPE" != "linux"* ]]; then
    echo "警告: 此脚本设计用于Linux系统"
fi

# 加载环境变量
if [ -f .env ]; then
    export $(cat .env | xargs)
    echo "环境变量已加载"
else
    echo "警告: 未找到.env文件"
fi

# 检查Docker
if ! command -v docker &> /dev/null; then
    echo "错误: 未找到Docker"
    exit 1
fi

# 检查Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "错误: 未找到Docker Compose"
    exit 1
fi

# 启动基础服务 (PostgreSQL和Redis)
echo "启动基础服务 (PostgreSQL和Redis)..."
docker-compose up -d postgres redis

# 等待服务启动
echo "等待数据库服务启动..."
sleep 10

# 检查服务状态
echo "检查服务状态..."
docker-compose ps

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

# 启动应用
echo "启动VectorSphere平台..."
python3 app.py