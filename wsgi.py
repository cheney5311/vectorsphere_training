"""VectorSphere Intelligent Platform - WSGI入口

用于生产环境部署的WSGI应用入口
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ.setdefault('ENVIRONMENT', 'production')
os.environ.setdefault('FLASK_ENV', 'production')

from app import create_app

# 创建WSGI应用
application = create_app()

if __name__ == '__main__':
    # 用于调试
    application.run()