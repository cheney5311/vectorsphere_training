"""测试统一监控服务"""

import time
import threading
from backend.core.monitoring.service import get_monitoring_service
from backend.core.monitoring.analyzer import get_performance_analyzer
from backend.core.monitoring.optimizer import get_resource_optimizer


def test_unified_monitoring():
    """测试统一监控服务"""
    print("开始测试统一监控服务...")
    
    # 获取监控服务实例
    monitor = get_monitoring_service()
    
    # 启动监控
    monitor.start_monitoring()
    print("监控服务已启动")
    
    # 等待一段时间收集指标
    time.sleep(5)
    
    # 获取当前指标
    metrics = monitor.get_current_metrics()
    print(f"当前指标: {metrics}")
    
    # 获取活跃告警
    alerts = monitor.get_active_alerts()
    print(f"活跃告警数量: {len(alerts)}")
    
    # 获取历史指标
    history = monitor.get_metrics_history()
    print(f"历史指标数量: {len(history.get('cpu_usage', []))}")
    
    # 测试性能分析器
    analyzer = get_performance_analyzer()
    if metrics['system']:
        report = analyzer.analyze(metrics['system'], metrics['gpu'])
        print(f"性能分析报告分数: {report.overall_score}")
    
    # 测试资源优化器
    optimizer = get_resource_optimizer()
    if metrics['system']:
        recommendations = optimizer.generate_recommendations(metrics['system'], metrics['gpu'])
        print(f"优化建议数量: {len(recommendations)}")
    
    # 停止监控
    monitor.stop_monitoring()
    print("监控服务已停止")
    
    print("测试完成!")


if __name__ == "__main__":
    test_unified_monitoring()