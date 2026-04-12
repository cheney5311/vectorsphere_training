"""默认告警规则定义（可被动态管理）

规则示例：
- high_cpu_usage: cpu_percent > 85% for 60s -> severity HIGH
- high_memory_usage: memory_percent > 90% for 30s -> severity MEDIUM
- low_disk_space: disk_percent > 95% for 300s -> severity HIGH
- training_loss_spike: training.loss increases > 2x over last epoch -> severity MEDIUM
"""
from backend.core.monitoring.models import AlertRule, AlertLevel, MetricType
from datetime import timedelta

DEFAULT_ALERT_RULES = [
    AlertRule(id='high_cpu_usage', name='CPU 使用率过高', metric_type=MetricType.SYSTEM, metric_name='cpu_percent', operator='>', threshold=85.0, duration=60, severity=AlertLevel.HIGH),
    AlertRule(id='high_memory_usage', name='内存使用率过高', metric_type=MetricType.SYSTEM, metric_name='memory_percent', operator='>', threshold=90.0, duration=30, severity=AlertLevel.MEDIUM),
    AlertRule(id='low_disk_space', name='磁盘不足', metric_type=MetricType.SYSTEM, metric_name='disk_percent', operator='>', threshold=95.0, duration=300, severity=AlertLevel.HIGH),
    AlertRule(id='high_gpu_utilization', name='GPU 使用率过高', metric_type=MetricType.GPU, metric_name='gpu_utilization', operator='>', threshold=90.0, duration=120, severity=AlertLevel.MEDIUM),
    AlertRule(id='training_loss_spike', name='训练损失突增', metric_type=MetricType.TRAINING, metric_name='loss', operator='>', threshold=2.0, duration=0, severity=AlertLevel.MEDIUM),
]
