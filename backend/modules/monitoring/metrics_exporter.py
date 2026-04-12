"""训练指标导出器 - 统一的监控模块

提供训练指标的收集、导出和 Prometheus 监控功能。
合并自 backend.modules.training.monitoring.metrics_exporter
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from flask import Blueprint, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

# GPU Prometheus 指标
GPU_UTIL_GAUGE = Gauge("gpu_utilization_percent", "GPU utilization percent", ["node", "gpu_index"]) 
GPU_MEMORY_FREE_GAUGE = Gauge("gpu_memory_free_mb", "GPU memory free MB", ["node", "gpu_index"]) 
GPU_COUNT_GAUGE = Gauge("gpu_count", "Number of GPUs on node", ["node"])

# Allocation metrics
ALLOCATION_REQUESTS_COUNTER = Counter("allocation_requests_total", "Total allocation requests", ["result"])
ALLOCATION_FAILURES_COUNTER = Counter("allocation_failures_total", "Total allocation failures", ["reason"])
logger = logging.getLogger(__name__)

# --- 可选集成：PushGateway 与 InfluxDB 助手 ---

def _maybe_push_to_pushgateway(job: str = 'vectorsphere-training') -> None:
    """如环境启用，推送默认注册表到 PushGateway"""
    import os
    try:
        if os.getenv('ENABLE_PUSHGATEWAY', 'false').lower() != 'true':
            return
        from prometheus_client import REGISTRY, push_to_gateway
        url = os.getenv('PUSHGATEWAY_URL', 'http://localhost:9091')
        push_to_gateway(url, job=job, registry=REGISTRY)
        logger.debug(f"PushGateway 推送成功: {url}, job={job}")
    except Exception as e:
        logger.warning(f"PushGateway 推送失败: {e}")


def _maybe_write_influx(session_id: str, stage: Optional[str], metrics: Dict[str, Any]) -> None:
    """如环境启用，写入 InfluxDB v2（HTTP line protocol）"""
    import os
    try:
        if os.getenv('ENABLE_INFLUX', 'false').lower() != 'true':
            return
        influx_url = os.getenv('INFLUX_URL', '').rstrip('/')
        org = os.getenv('INFLUX_ORG', '')
        bucket = os.getenv('INFLUX_BUCKET', '')
        token = os.getenv('INFLUX_TOKEN', '')
        if not (influx_url and org and bucket and token):
            logger.warning("InfluxDB 环境变量不完整，跳过写入")
            return
        write_endpoint = f"{influx_url}/api/v2/write?org={org}&bucket={bucket}&precision=s"
        import requests
        lines = []
        for k, v in metrics.items():
            try:
                val = float(v)
            except Exception:
                continue
            measurement = 'training_metrics'
            tags = []
            tags.append(f"session_id={session_id}")
            if stage:
                tags.append(f"stage={stage}")
            line = f"{measurement},{','.join(tags)} {k}={val}"
            lines.append(line)
        if not lines:
            return
        resp = requests.post(write_endpoint, headers={"Authorization": f"Token {token}"}, data='\n'.join(lines), timeout=5)
        if resp.status_code >= 300:
            logger.warning(f"InfluxDB 写入失败: {resp.status_code} {resp.text}")
        else:
            logger.debug("InfluxDB 写入成功")
    except Exception as e:
        logger.warning(f"InfluxDB 写入异常: {e}")


def _maybe_write_timescale(session_id: str, stage: Optional[str], metrics: Dict[str, Any]) -> None:
    """如环境启用，写入 TimescaleDB（PostgreSQL）"""
    import os
    try:
        if os.getenv('ENABLE_TIMESCALE', 'false').lower() != 'true':
            return
        dsn = os.getenv('TS_DB_DSN')
        host = os.getenv('TS_DB_HOST')
        user = os.getenv('TS_DB_USER')
        password = os.getenv('TS_DB_PASSWORD')
        dbname = os.getenv('TS_DB_NAME')
        port = os.getenv('TS_DB_PORT', '5432')
        table = os.getenv('TS_TABLE', 'training_metrics')
        if not dsn and not (host and user and password and dbname):
            logger.warning("TimescaleDB 连接信息不完整，跳过写入")
            return
        # Import psycopg2 lazily to avoid segfaults in test environments without Postgres
        try:
            import psycopg2
        except Exception:
            logger.warning("psycopg2 not available; skipping TimescaleDB write")
            return
        conn_args = {}
        if dsn:
            conn_args['dsn'] = dsn
        else:
            conn_args.update(dict(host=host, user=user, password=password, dbname=dbname, port=port))
        conn = psycopg2.connect(**conn_args)
        conn.autocommit = True
        cur = conn.cursor()
        # 表与 hypertable 初始化（幂等）
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                time TIMESTAMPTZ NOT NULL,
                session_id TEXT NOT NULL,
                stage TEXT,
                metric_key TEXT NOT NULL,
                metric_value DOUBLE PRECISION NOT NULL
            );
        """)
        try:
            cur.execute(f"SELECT create_hypertable('{table}', 'time', if_not_exists => TRUE);")
        except Exception:
            pass
        # 保留策略与压缩策略（可选）
        try:
            retention_days = int(os.getenv('TS_RETENTION_DAYS', '0'))
            if retention_days > 0:
                cur.execute(f"SELECT add_retention_policy('{table}', INTERVAL '{retention_days} days');")
        except Exception:
            pass
        try:
            if os.getenv('TS_ENABLE_COMPRESSION', 'false').lower() == 'true':
                compress_interval = os.getenv('TS_COMPRESS_INTERVAL', '7 days')
                cur.execute(f"ALTER TABLE {table} SET (timescaledb.compress, timescaledb.compress_segmentby = 'session_id');")
                cur.execute(f"SELECT add_compression_policy('{table}', INTERVAL '{compress_interval}');")
        except Exception:
            pass
        now_sql = datetime.utcnow().isoformat()
        for k, v in metrics.items():
            try:
                val = float(v)
            except Exception:
                continue
            cur.execute(
                f"INSERT INTO {table} (time, session_id, stage, metric_key, metric_value) VALUES (%s, %s, %s, %s, %s)",
                (now_sql, session_id, stage, k, val)
            )
        cur.close()
        conn.close()
        logger.debug("TimescaleDB 写入成功")
    except Exception as e:
        logger.warning(f"TimescaleDB 写入异常: {e}")

# Prometheus 训练相关指标
TRAINING_EPOCH_COUNTER = Counter("training_epoch_total", "Total epochs processed", ["session", "stage"])
TRAINING_LOSS_GAUGE = Gauge("training_loss", "Current training loss", ["session", "stage"])
TRAINING_ACCURACY_GAUGE = Gauge("training_accuracy", "Current training accuracy", ["session", "stage"])
TRAINING_LR_GAUGE = Gauge("training_learning_rate", "Current learning rate", ["session", "stage"])
TRAINING_THROUGHPUT_GAUGE = Gauge("training_throughput", "Samples/sec", ["session", "stage"])
TRAINING_MEMORY_GAUGE = Gauge("training_memory_usage_mb", "Memory usage (MB)", ["session", "stage"])
STEP_DURATION_HIST = Histogram("training_step_duration_seconds", "Step duration seconds", ["session", "stage"])

metrics_bp = Blueprint("metrics", __name__, url_prefix="/metrics")

@metrics_bp.route("", methods=["GET"])
def metrics() -> Response:
    """Prometheus 指标端点"""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


def record_training_metrics(
    session_id: str,
    metrics: Dict[str, Any],
    stage: Optional[str] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None
) -> bool:
    """记录训练指标
    
    Args:
        session_id: 训练会话ID
        metrics: 指标数据
        stage: 训练阶段
        epoch: 训练轮次
        step: 训练步数
        
    Returns:
        bool: 是否成功记录
    """
    try:
        # 构建指标记录
        metric_record = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'epoch': epoch,
            'step': step,
            'metrics': metrics
        }
        
        # 记录到日志
        logger.info(f"训练指标记录: {session_id}, 阶段: {stage}, 轮次: {epoch}, 步数: {step}")
        logger.debug(f"指标详情: {metrics}")
        
        # 更新 Prometheus 指标
        if stage:
            labels = {"session": session_id, "stage": stage}
            try:
                if epoch is not None:
                    TRAINING_EPOCH_COUNTER.labels(**labels).inc()
                
                # 从 metrics 字典中提取常见指标
                loss = metrics.get('loss')
                accuracy = metrics.get('accuracy')
                lr = metrics.get('learning_rate') or metrics.get('lr')
                throughput = metrics.get('throughput')
                memory_mb = metrics.get('memory_mb') or metrics.get('memory_usage')
                
                if loss is not None:
                    TRAINING_LOSS_GAUGE.labels(**labels).set(float(loss))
                if accuracy is not None:
                    TRAINING_ACCURACY_GAUGE.labels(**labels).set(float(accuracy))
                if lr is not None:
                    TRAINING_LR_GAUGE.labels(**labels).set(float(lr))
                if throughput is not None:
                    TRAINING_THROUGHPUT_GAUGE.labels(**labels).set(float(throughput))
                if memory_mb is not None:
                    TRAINING_MEMORY_GAUGE.labels(**labels).set(float(memory_mb))
            except Exception as e:
                logger.warning(f"更新 Prometheus 指标失败: {e}")
        
        # 额外：如果 metrics 中包含 GPU 信息，导出到 Prometheus GPU 指标
        try:
            node = metrics.get('node')
            gpus = metrics.get('gpus', [])
            if node and isinstance(gpus, list):
                GPU_COUNT_GAUGE.labels(node=node).set(len(gpus))
                for g in gpus:
                    try:
                        idx = g.get('index')
                        GPU_UTIL_GAUGE.labels(node=node, gpu_index=str(idx)).set(float(g.get('utilization_percent') or g.get('utilization') or 0.0))
                        GPU_MEMORY_FREE_GAUGE.labels(node=node, gpu_index=str(idx)).set(float(g.get('memory_free') or g.get('memory_free_mb') or 0))
                    except Exception:
                        continue
        except Exception:
            pass

        # 可选：推送到 PushGateway 与 InfluxDB/TimescaleDB
        _maybe_push_to_pushgateway()
        _maybe_write_influx(session_id, stage, metrics)
        _maybe_write_timescale(session_id, stage, metrics)
        
        return True
        
    except Exception as e:
        logger.error(f"记录训练指标失败: {session_id}, 错误: {e}")
        return False


def export_training_metrics(
    session_id: str,
    format: str = 'json'
) -> Optional[Dict[str, Any]]:
    """导出训练指标
    
    Args:
        session_id: 训练会话ID
        format: 导出格式
        
    Returns:
        Optional[Dict[str, Any]]: 导出的指标数据
    """
    try:
        # 这里应该从存储中获取指标数据
        # 目前返回模拟数据
        metrics_data = {
            'session_id': session_id,
            'export_time': datetime.now().isoformat(),
            'format': format,
            'metrics': {
                'total_epochs': 10,
                'total_steps': 1000,
                'final_loss': 0.1,
                'best_accuracy': 0.95,
                'training_time': 3600  # 秒
            }
        }
        
        logger.info(f"导出训练指标: {session_id}, 格式: {format}")
        return metrics_data
        
    except Exception as e:
        logger.error(f"导出训练指标失败: {session_id}, 错误: {e}")
        return None


def get_training_metrics_summary(session_id: str) -> Optional[Dict[str, Any]]:
    """获取训练指标摘要
    
    Args:
        session_id: 训练会话ID
        
    Returns:
        Optional[Dict[str, Any]]: 指标摘要
    """
    try:
        # 这里应该从存储中获取并计算指标摘要
        # 目前返回模拟数据
        summary = {
            'session_id': session_id,
            'summary_time': datetime.now().isoformat(),
            'total_metrics_count': 100,
            'latest_metrics': {
                'loss': 0.1,
                'accuracy': 0.95,
                'learning_rate': 1e-5
            },
            'best_metrics': {
                'best_loss': 0.05,
                'best_accuracy': 0.98,
                'best_epoch': 8
            },
            'average_metrics': {
                'avg_loss': 0.2,
                'avg_accuracy': 0.85
            }
        }
        
        logger.info(f"获取训练指标摘要: {session_id}")
        return summary
        
    except Exception as e:
        logger.error(f"获取训练指标摘要失败: {session_id}, 错误: {e}")
        return None


# 兼容性函数 - 保持原有的 Prometheus 接口
def record_prometheus_metrics(session_id: str, stage: str, epoch: int, loss: Optional[float], accuracy: Optional[float], lr: Optional[float], throughput: Optional[float], memory_mb: Optional[float]):
    """记录 Prometheus 指标（兼容性函数）"""
    labels = {"session": session_id, "stage": stage}
    try:
        TRAINING_EPOCH_COUNTER.labels(**labels).inc()
        if loss is not None:
            TRAINING_LOSS_GAUGE.labels(**labels).set(float(loss))
        if accuracy is not None:
            TRAINING_ACCURACY_GAUGE.labels(**labels).set(float(accuracy))
        if lr is not None:
            TRAINING_LR_GAUGE.labels(**labels).set(float(lr))
        if throughput is not None:
            TRAINING_THROUGHPUT_GAUGE.labels(**labels).set(float(throughput))
        if memory_mb is not None:
            TRAINING_MEMORY_GAUGE.labels(**labels).set(float(memory_mb))
    except Exception:
        # 指标异常不影响训练流程
        pass