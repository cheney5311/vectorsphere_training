"""GPU 资源管理（骨架实现）
- 提供 GPU 探测、显存/利用率采集、简单配额查询接口。
- 在无 nvidia-smi 环境下提供降级实现以保持兼容性。
"""
import logging
import subprocess
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def detect_gpus() -> List[Dict[str, Any]]:
    """检测可用 GPU 列表，返回每个 GPU 的基础信息。
    降级行为：如果未安装 nvidia-smi，则返回空列表。
    """
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,uuid', '--format=csv,noheader,nounits'], stderr=subprocess.DEVNULL)
        lines = out.decode().strip().splitlines()
        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_total_mb': int(parts[2]) if parts[2].isdigit() else None,
                    'uuid': parts[3]
                })
        return gpus
    except Exception:
        # 无 nvidia-smi 或不可用，返回空列表
        return []


def get_gpu_metrics() -> List[Dict[str, Any]]:
    """获取每个 GPU 的当前利用率与显存使用情况。
    返回示例：[{index:0, util:10, memory_used_mb:1234, memory_total_mb:16384}, ...]
    """
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], stderr=subprocess.DEVNULL)
        lines = out.decode().strip().splitlines()
        metrics = []
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                metrics.append({
                    'index': int(parts[0]),
                    'utilization_percent': int(parts[1]) if parts[1].isdigit() else None,
                    'memory_used_mb': int(parts[2]) if parts[2].isdigit() else None,
                    'memory_total_mb': int(parts[3]) if parts[3].isdigit() else None
                })
        return metrics
    except Exception:
        return []


def get_gpu_summary() -> Dict[str, Any]:
    """返回汇总信息，用于监控导出或调度决策"""
    gpus = detect_gpus()
    metrics = get_gpu_metrics()
    return {
        'gpu_count': len(gpus),
        'gpus': metrics or gpus
    }
