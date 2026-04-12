"""统一可观测性工具集（轻量骨架）

提供：
- Prometheus 统一 Metrics 注册封装（Counter/Gauge/Histogram）
- 简单的 metrics 注册器初始化函数（可在服务启动处调用）
- convenience decorator/context for timing

设计原则：
- 非侵入：若 prometheus_client 不可用，所有函数退化为 no-op，保证在 CI/开发环境不因缺少依赖而失败
- 统一命名与标签（service, component, instance）便于聚合
"""
from typing import Optional, Callable
import time
import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server
    _HAS_PROM = True
except Exception:
    _HAS_PROM = False
    # define no-op standins
    class CollectorRegistry: pass
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def labels(self, *a, **k): return self
        def inc(self, v=1): pass
    class Gauge(Counter):
        def set(self, v): pass
    class Histogram(Counter):
        def observe(self, v): pass
    def start_http_server(port, addr='0.0.0.0', registry=None):
        logger.debug('prometheus_client not available; start_http_server noop')


_DEFAULT_REGISTRY = None


def init_metrics(port: Optional[int] = None, service: str = 'vectorsphere', registry: Optional[CollectorRegistry] = None):
    """初始化全局 metrics registry 并启动 metrics HTTP server（best-effort）"""
    global _DEFAULT_REGISTRY
    if not _HAS_PROM:
        logger.info('prometheus_client not available; metrics disabled')
        return None
    if registry is None:
        registry = CollectorRegistry()
    _DEFAULT_REGISTRY = registry
    if port:
        try:
            start_http_server(port, registry=registry)
            logger.info(f'started prometheus metrics http server on port {port}')
        except Exception as e:
            logger.warning(f'failed to start prometheus http server: {e}')
    return registry


class _MetricProxy:
    """Proxy wrapper for prometheus metrics to allow calling inc()/set() without labels.
    If underlying metric is labeled, proxy will supply default label values from env or 'unknown'."""
    def __init__(self, metric, labelnames=None):
        self._m = metric
        self._labelnames = labelnames
        # default label values
        self._default_labels = {
            'service': os.getenv('SERVICE_NAME', 'vectorsphere-backend'),
            'component': os.getenv('COMPONENT', 'unknown')
        }

    def labels(self, **labels):
        return self._m.labels(**{**self._default_labels, **labels})

    def inc(self, v=1):
        try:
            if self._labelnames:
                self._m.labels(**self._default_labels).inc(v)
            else:
                self._m.inc(v)
        except Exception:
            pass

    def set(self, v):
        try:
            if self._labelnames:
                self._m.labels(**self._default_labels).set(v)
            else:
                self._m.set(v)
        except Exception:
            pass

    def observe(self, v):
        try:
            if self._labelnames:
                self._m.labels(**self._default_labels).observe(v)
            else:
                self._m.observe(v)
        except Exception:
            pass


def create_counter(name: str, documentation: str, labelnames: Optional[tuple] = None):
    if not _HAS_PROM:
        return Counter(name, documentation)
    try:
        # default labelnames when None
        if labelnames is None:
            # create unlabeled
            return Counter(name, documentation, registry=_DEFAULT_REGISTRY)
        m = Counter(name, documentation, labelnames=labelnames, registry=_DEFAULT_REGISTRY)
        return _MetricProxy(m, labelnames=labelnames)
    except Exception:
        try:
            return Counter(name, documentation, registry=_DEFAULT_REGISTRY)
        except Exception:
            return Counter(name, documentation)


def create_gauge(name: str, documentation: str, labelnames: Optional[tuple] = None):
    if not _HAS_PROM:
        return Gauge(name, documentation)
    try:
        if labelnames is None:
            return Gauge(name, documentation, registry=_DEFAULT_REGISTRY)
        m = Gauge(name, documentation, labelnames=labelnames, registry=_DEFAULT_REGISTRY)
        return _MetricProxy(m, labelnames=labelnames)
    except Exception:
        try:
            return Gauge(name, documentation, registry=_DEFAULT_REGISTRY)
        except Exception:
            return Gauge(name, documentation)


def timing(histogram: Histogram, labels: dict = None):
    """Context manager for timing a block and observing in histogram"""
    class _Timer:
        def __enter__(self):
            self._start = time.time()
            return self
        def __exit__(self, exc_type, exc, tb):
            elapsed = time.time() - self._start
            try:
                if labels:
                    histogram.labels(**labels).observe(elapsed)
                else:
                    histogram.observe(elapsed)
            except Exception:
                pass
    return _Timer()
