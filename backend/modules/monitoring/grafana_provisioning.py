"""
Grafana 预置配置生成器
- 自动生成 Grafana 数据源与仪表盘预置文件
- 通过环境变量控制，默认不影响现有行为

环境变量：
- GRAFANA_PROVISION_DIR=/etc/grafana/provisioning
- GRAFANA_ENABLE_PROVISIONING=true
- GRAFANA_PROVISION_FORCE=true  # 可选，强制重写预置文件以触发 Grafana 重载

数据源（可选）：
- Prometheus:
  - GRAFANA_DS_PROMETHEUS_URL=http://prometheus:9090
- InfluxDB v2:
  - GRAFANA_DS_INFLUX_URL=http://influxdb:8086
  - GRAFANA_DS_INFLUX_ORG=your-org
  - GRAFANA_DS_INFLUX_TOKEN=your-token
  - GRAFANA_DS_INFLUX_BUCKET=your-bucket
- TimescaleDB(PostgreSQL):
  - GRAFANA_DS_PG_HOST=timescale
  - GRAFANA_DS_PG_DB=vectorsphere
  - GRAFANA_DS_PG_USER=postgres
  - GRAFANA_DS_PG_PASSWORD=your_password
  - GRAFANA_DS_PG_PORT=5432

"""
import os
import json
from typing import Dict, Any


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_yaml(path: str, content: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def _datasources_yaml(config: Dict[str, Any]) -> str:
    # 生成 Grafana datasources.yaml（简单字符串模板，避免额外依赖）
    # 为避免重复创建数据源，显式设置数据源 uid（稳定）
    prom_uid = os.getenv('GRAFANA_DS_PROM_UID', 'prometheus')
    influx_uid = os.getenv('GRAFANA_DS_INFLUX_UID', 'influxdb')
    pg_uid = os.getenv('GRAFANA_DS_PG_UID', 'timescaledb')

    lines = [
        "apiVersion: 1",
        "datasources:",
    ]
    if config.get('prometheus_url'):
        lines += [
            "  - name: Prometheus",
            "    type: prometheus",
            "    access: proxy",
            f"    url: {config['prometheus_url']}",
            "    isDefault: true",
            f"    uid: {prom_uid}",
        ]
    if config.get('influx_url') and config.get('influx_org') and config.get('influx_token'):
        lines += [
            "  - name: InfluxDB",
            "    type: influxdb",
            "    access: proxy",
            f"    url: {config['influx_url']}",
            "    jsonData:",
            f"      version: Flux",
            f"      organization: {config['influx_org']}",
            f"      defaultBucket: {config.get('influx_bucket', '')}",
            "    secureJsonData:",
            f"      token: {config['influx_token']}",
            f"    uid: {influx_uid}",
        ]
    if config.get('pg_host') and config.get('pg_db') and config.get('pg_user'):
        lines += [
            "  - name: TimescaleDB",
            "    type: postgres",
            "    access: proxy",
            f"    url: {config['pg_host']}:{config.get('pg_port', 5432)}",
            f"    database: {config['pg_db']}",
            f"    user: {config['pg_user']}",
            "    secureJsonData:",
            f"      password: {config.get('pg_password', '')}",
            "    jsonData:",
            "      postgresVersion: 1200",
            "      sslmode: disable",
            f"    uid: {pg_uid}",
        ]
    return "\n".join(lines) + "\n"


def _dashboard_training_overview_json() -> Dict[str, Any]:
    # 极简训练概览仪表盘（使用 Prometheus 指标）
    # 为避免重复导入，显式设置仪表盘 uid，并设置面板 id 稳定
    dashboard_uid = os.getenv('GRAFANA_DASHBOARD_UID_TRAINING', 'vectorsphere-training-overview')
    return {
        "uid": dashboard_uid,
        "annotations": {"list": []},
        "editable": True,
        "title": "VectorSphere Training Overview",
        "timezone": "",
        "panels": [
            {
                "id": 1,
                "type": "stat",
                "title": "Epoch Count",
                "datasource": {"type": "prometheus", "uid": os.getenv('GRAFANA_DS_PROM_UID', 'prometheus')},
                "targets": [{"expr": "sum(training_epoch_total)", "refId": "A"}],
                "gridPos": {"h": 6, "w": 8, "x": 0, "y": 0}
            },
            {
                "id": 2,
                "type": "graph",
                "title": "Loss by Session",
                "datasource": {"type": "prometheus", "uid": os.getenv('GRAFANA_DS_PROM_UID', 'prometheus')},
                "targets": [{"expr": "training_loss", "legendFormat": "{{session}}|{{stage}}", "refId": "A"}],
                "gridPos": {"h": 8, "w": 16, "x": 8, "y": 0}
            },
            {
                "id": 3,
                "type": "graph",
                "title": "Accuracy by Session",
                "datasource": {"type": "prometheus", "uid": os.getenv('GRAFANA_DS_PROM_UID', 'prometheus')},
                "targets": [{"expr": "training_accuracy", "legendFormat": "{{session}}|{{stage}}", "refId": "A"}],
                "gridPos": {"h": 8, "w": 16, "x": 0, "y": 8}
            }
        ],
        "schemaVersion": 36,
        "version": 1
    }


def setup_grafana_provisioning() -> None:
    if os.getenv('GRAFANA_ENABLE_PROVISIONING', 'false').lower() != 'true':
        return
    base_dir = os.getenv('GRAFANA_PROVISION_DIR', '/etc/grafana/provisioning')
    datasources_dir = os.path.join(base_dir, 'datasources')
    dashboards_dir = os.path.join(base_dir, 'dashboards')
    _ensure_dir(datasources_dir)
    _ensure_dir(dashboards_dir)

    # 组装配置
    cfg = {
        'prometheus_url': os.getenv('GRAFANA_DS_PROMETHEUS_URL'),
        'influx_url': os.getenv('GRAFANA_DS_INFLUX_URL'),
        'influx_org': os.getenv('GRAFANA_DS_INFLUX_ORG'),
        'influx_token': os.getenv('GRAFANA_DS_INFLUX_TOKEN'),
        'influx_bucket': os.getenv('GRAFANA_DS_INFLUX_BUCKET'),
        'pg_host': os.getenv('GRAFANA_DS_PG_HOST'),
        'pg_db': os.getenv('GRAFANA_DS_PG_DB'),
        'pg_user': os.getenv('GRAFANA_DS_PG_USER'),
        'pg_password': os.getenv('GRAFANA_DS_PG_PASSWORD'),
        'pg_port': int(os.getenv('GRAFANA_DS_PG_PORT', '5432')),
    }

    # 有状态更新策略：如内容未变化则跳过写入，避免重复导入；支持强制重写开关
    force_rewrite = os.getenv('GRAFANA_PROVISION_FORCE', 'false').lower() == 'true'
    datasources_path = os.path.join(datasources_dir, 'datasources.yaml')
    new_datasources_yaml = _datasources_yaml(cfg)
    try:
        if os.path.exists(datasources_path):
            with open(datasources_path, 'r', encoding='utf-8') as rf:
                old = rf.read()
            if force_rewrite or old != new_datasources_yaml:
                _write_yaml(datasources_path, new_datasources_yaml)
            else:
                pass
        else:
            _write_yaml(datasources_path, new_datasources_yaml)
    except Exception:
        _write_yaml(datasources_path, new_datasources_yaml)

    # 写仪表盘 JSON 与索引 YAML（带 uid，按需更新）；支持强制重写开关
    dashboard_json = _dashboard_training_overview_json()
    dashboard_path = os.path.join(dashboards_dir, 'training_overview.json')
    new_dashboard_str = json.dumps(dashboard_json, ensure_ascii=False, indent=2)
    try:
        if os.path.exists(dashboard_path):
            with open(dashboard_path, 'r', encoding='utf-8') as rf:
                old = rf.read()
            if force_rewrite or old != new_dashboard_str:
                with open(dashboard_path, 'w', encoding='utf-8') as f:
                    f.write(new_dashboard_str)
            else:
                pass
        else:
            with open(dashboard_path, 'w', encoding='utf-8') as f:
                f.write(new_dashboard_str)
    except Exception:
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(new_dashboard_str)

    dashboards_yaml = "\n".join([
        "apiVersion: 1",
        "providers:",
        "  - name: Default",
        "    orgId: 1",
        "    folder: VectorSphere",
        "    type: file",
        "    disableDeletion: false",
        "    editable: true",
        f"    options:",
        f"      path: {dashboards_dir}",
    ]) + "\n"
    dashboards_index_path = os.path.join(dashboards_dir, 'dashboards.yaml')
    try:
        if os.path.exists(dashboards_index_path):
            with open(dashboards_index_path, 'r', encoding='utf-8') as rf:
                old = rf.read()
            if force_rewrite or old != dashboards_yaml:
                _write_yaml(dashboards_index_path, dashboards_yaml)
            else:
                pass
        else:
            _write_yaml(dashboards_index_path, dashboards_yaml)
    except Exception:
        _write_yaml(dashboards_index_path, dashboards_yaml)
