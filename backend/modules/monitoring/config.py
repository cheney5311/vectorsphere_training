"""监控模块配置管理"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """监控配置"""
    # 性能监控配置
    monitoring_interval: float = 1.0  # 监控间隔(秒)
    history_size: int = 3600  # 历史数据保留数量
    enable_gpu_monitoring: bool = True  # 是否启用GPU监控
    enable_anomaly_detection: bool = True  # 是否启用异常检测
    
    # 告警配置
    alert_check_interval: int = 60  # 告警检查间隔(秒)
    alert_retention_days: int = 30  # 告警保留天数
    enable_email_notifications: bool = False  # 是否启用邮件通知
    enable_slack_notifications: bool = False  # 是否启用Slack通知
    
    # 仪表板配置
    dashboard_refresh_interval: int = 30  # 仪表板刷新间隔(秒)
    dashboard_retention_days: int = 7  # 仪表板数据保留天数
    
    # 异常检测配置
    anomaly_window_size: int = 100  # 异常检测窗口大小
    anomaly_contamination: float = 0.1  # 异常比例
    anomaly_sensitivity: float = 0.8  # 异常检测敏感度
    
    # 数据库配置
    database_url: str = ""  # 数据库连接URL
    database_pool_size: int = 10  # 数据库连接池大小
    
    def __post_init__(self):
        """初始化后处理"""
        # 从环境变量获取配置
        self._load_from_env()
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        # 性能监控配置
        self.monitoring_interval = float(os.getenv('MONITORING_INTERVAL', self.monitoring_interval))
        self.history_size = int(os.getenv('HISTORY_SIZE', self.history_size))
        self.enable_gpu_monitoring = os.getenv('ENABLE_GPU_MONITORING', str(self.enable_gpu_monitoring)).lower() == 'true'
        self.enable_anomaly_detection = os.getenv('ENABLE_ANOMALY_DETECTION', str(self.enable_anomaly_detection)).lower() == 'true'
        
        # 告警配置
        self.alert_check_interval = int(os.getenv('ALERT_CHECK_INTERVAL', self.alert_check_interval))
        self.alert_retention_days = int(os.getenv('ALERT_RETENTION_DAYS', self.alert_retention_days))
        self.enable_email_notifications = os.getenv('ENABLE_EMAIL_NOTIFICATIONS', str(self.enable_email_notifications)).lower() == 'true'
        self.enable_slack_notifications = os.getenv('ENABLE_SLACK_NOTIFICATIONS', str(self.enable_slack_notifications)).lower() == 'true'
        
        # 仪表板配置
        self.dashboard_refresh_interval = int(os.getenv('DASHBOARD_REFRESH_INTERVAL', self.dashboard_refresh_interval))
        self.dashboard_retention_days = int(os.getenv('DASHBOARD_RETENTION_DAYS', self.dashboard_retention_days))
        
        # 异常检测配置
        self.anomaly_window_size = int(os.getenv('ANOMALY_WINDOW_SIZE', self.anomaly_window_size))
        self.anomaly_contamination = float(os.getenv('ANOMALY_CONTAMINATION', self.anomaly_contamination))
        self.anomaly_sensitivity = float(os.getenv('ANOMALY_SENSITIVITY', self.anomaly_sensitivity))
        
        # 数据库配置
        self.database_url = os.getenv('MONITORING_DATABASE_URL', self.database_url)
        self.database_pool_size = int(os.getenv('DATABASE_POOL_SIZE', self.database_pool_size))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MonitoringConfig':
        """从字典创建配置对象"""
        return cls(**data)


# 全局配置实例
_monitoring_config: Optional[MonitoringConfig] = None


def get_monitoring_config() -> MonitoringConfig:
    """获取监控配置"""
    global _monitoring_config
    if _monitoring_config is None:
        _monitoring_config = MonitoringConfig()
    return _monitoring_config


def set_monitoring_config(config: MonitoringConfig):
    """设置监控配置"""
    global _monitoring_config
    _monitoring_config = config


def load_config_from_file(filepath: str) -> MonitoringConfig:
    """从文件加载配置"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        config = MonitoringConfig.from_dict(data)
        set_monitoring_config(config)
        logger.info(f"监控配置已从 {filepath} 加载")
        return config
    except Exception as e:
        logger.error(f"加载监控配置失败: {e}")
        raise


def save_config_to_file(filepath: str, config: Optional[MonitoringConfig] = None):
    """保存配置到文件"""
    if config is None:
        config = get_monitoring_config()
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"监控配置已保存到 {filepath}")
    except Exception as e:
        logger.error(f"保存监控配置失败: {e}")
        raise


# 导出所有配置相关类和函数
__all__ = [
    'MonitoringConfig',
    'get_monitoring_config',
    'set_monitoring_config',
    'load_config_from_file',
    'save_config_to_file'
]