"""轻量告警发送与调度适配器

功能：
- 提供统一接口 `send_alert(alert)`，可通过 webhook、email 或 Alertmanager API 发送告警
- 支持基于配置的告警策略（ALERT_WEBHOOK_URL, ALERTMANAGER_URL, ALERT_EMAIL_RECIPIENTS）
- 若依赖不可用则退化为日志记录
"""
from typing import Dict, Any, Optional
import os
import logging
import requests

logger = logging.getLogger(__name__)


def send_alert(alert: Dict[str, Any]) -> bool:
    """发送告警，alert 为字典，包含至少 name, severity, description, metric, value, timestamp
    返回 True 表示发送成功或记录成功（退化情况下）。"""
    try:
        # 1) 尝试发送到 Alertmanager（如果配置了 ALERTMANAGER_URL）
        am_url = os.getenv('ALERTMANAGER_URL')
        if am_url:
            try:
                payload = [{
                    'labels': {
                        'alertname': alert.get('name', 'vectorsphere_alert'),
                        'severity': alert.get('severity', 'critical'),
                        'service': os.getenv('SERVICE_NAME', 'vectorsphere-backend')
                    },
                    'annotations': {
                        'summary': alert.get('description', ''),
                        'metric': alert.get('metric', ''),
                        'value': str(alert.get('value', ''))
                    },
                    'startsAt': alert.get('timestamp')
                }]
                resp = requests.post(f"{am_url.rstrip('/')}/api/v1/alerts", json=payload, timeout=5)
                if resp.status_code >= 200 and resp.status_code < 300:
                    logger.info(f"Alertmanager accepted alert: {alert.get('name')}")
                    return True
                else:
                    logger.warning(f"Alertmanager rejected alert: {resp.status_code} {resp.text}")
            except Exception as e:
                logger.warning(f"Failed to send alert to Alertmanager: {e}")

        # 2) 尝试发送到自定义 webhook（兼容旧版配置 ALERT_WEBHOOK_URL）
        webhook = os.getenv('ALERT_WEBHOOK_URL')
        if webhook:
            try:
                resp = requests.post(webhook, json=alert, timeout=5)
                if resp.status_code >= 200 and resp.status_code < 300:
                    logger.info(f"Webhook accepted alert: {alert.get('name')}")
                    return True
                else:
                    logger.warning(f"Webhook rejected alert: {resp.status_code} {resp.text}")
            except Exception as e:
                logger.warning(f"Failed to send alert to webhook: {e}")

        # 3) 退化：发送电子邮件（如果配置了 ALERT_EMAIL_RECIPIENTS 和 SMTP 环境）
        recipients = os.getenv('ALERT_EMAIL_RECIPIENTS')
        if recipients:
            try:
                # 简单实现：调用外部邮件发送命令或 API；这里退化为日志记录
                logger.info(f"Alert email to {recipients}: {alert.get('name')} - {alert.get('description')}")
                return True
            except Exception as e:
                logger.warning(f"Failed to send alert email: {e}")

        # 最后回退到日志
        logger.warning(f"Alert raised (no sink configured): {alert}")
        return True

    except Exception as e:
        logger.error(f"Unexpected error while sending alert: {e}")
        return False
