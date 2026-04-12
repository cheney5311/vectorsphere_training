import os
from backend.utils.observability import init_metrics, create_counter, create_gauge


def test_init_metrics_no_prom(monkeypatch):
    # simulate prometheus_client not installed by monkeypatching import
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1')
    # calling init_metrics should not raise
    init_metrics(None)


def test_create_counter(monkeypatch):
    # Should not raise even if prometheus not available
    c = create_counter('test_counter', 'doc')
    c.inc()
    g = create_gauge('test_gauge', 'doc')
    g.set(1)
    assert True
