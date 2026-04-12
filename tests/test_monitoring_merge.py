#!/usr/bin/env python3
"""
测试监控模块合并后的功能
验证 backend.modules.training.monitoring.metrics_exporter 的功能已成功合并到 backend.modules.monitoring.metrics_exporter
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_monitoring_module_merge():
    """测试监控模块合并"""
    print("🔍 开始测试监控模块合并...")
    
    try:
        # 测试统一监控模块的导入
        from backend.modules.monitoring.metrics_exporter import (
            record_training_metrics,
            export_training_metrics, 
            get_training_metrics_summary,
            record_prometheus_metrics,
            metrics_bp
        )
        print("✅ 统一监控模块导入成功")
        
        # 测试原训练监控模块是否已删除
        try:
            from backend.modules.training.monitoring.metrics_exporter import record_training_metrics as old_func
            print("❌ 原训练监控模块仍然存在，删除失败")
            return False
        except ImportError:
            print("✅ 原训练监控模块已成功删除")
        
        # 测试新的训练指标记录功能
        result = record_training_metrics(
            session_id='test_merge_123',
            metrics={
                'loss': 0.3,
                'accuracy': 0.92,
                'learning_rate': 0.0001,
                'throughput': 150.5,
                'memory_mb': 2048
            },
            stage='sft',
            epoch=5,
            step=500
        )
        
        if result:
            print("✅ 训练指标记录功能正常")
        else:
            print("❌ 训练指标记录功能异常")
            return False
        
        # 测试导出功能
        export_data = export_training_metrics('test_merge_123', format='json')
        if export_data and 'session_id' in export_data:
            print("✅ 训练指标导出功能正常")
        else:
            print("❌ 训练指标导出功能异常")
            return False
        
        # 测试摘要功能
        summary_data = get_training_metrics_summary('test_merge_123')
        if summary_data and 'session_id' in summary_data:
            print("✅ 训练指标摘要功能正常")
        else:
            print("❌ 训练指标摘要功能异常")
            return False
        
        # 测试兼容性函数
        record_prometheus_metrics(
            session_id='test_merge_123',
            stage='sft',
            epoch=5,
            loss=0.3,
            accuracy=0.92,
            lr=0.0001,
            throughput=150.5,
            memory_mb=2048
        )
        print("✅ Prometheus 兼容性函数正常")
        
        # 测试 Flask Blueprint
        if hasattr(metrics_bp, 'name') and metrics_bp.name == 'metrics':
            print("✅ Flask Blueprint 正常")
        else:
            print("❌ Flask Blueprint 异常")
            return False
        
        print("\n🎉 监控模块合并测试全部通过！")
        print("📋 合并总结:")
        print("   - ✅ 原 backend.modules.training.monitoring.metrics_exporter 已删除")
        print("   - ✅ 功能已合并到 backend.modules.monitoring.metrics_exporter")
        print("   - ✅ 保持了原有的 Prometheus 监控功能")
        print("   - ✅ 新增了完整的训练指标管理功能")
        print("   - ✅ 提供了兼容性接口")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_monitoring_module_merge()
    sys.exit(0 if success else 1)