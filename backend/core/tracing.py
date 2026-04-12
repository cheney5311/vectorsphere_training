"""
OpenTelemetry 分布式追踪初始化
"""
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.flask import FlaskInstrumentor


def init_tracing(app=None, service_name: Optional[str] = None) -> None:
    """
    初始化 OpenTelemetry 追踪。
    通过环境变量 ENABLE_TRACING 控制开关。
    """
    enable = os.getenv('ENABLE_TRACING', 'false').lower() == 'true'
    if not enable:
        return

    service = service_name or os.getenv('OTEL_SERVICE_NAME', 'vectorsphere-backend')
    endpoint = os.getenv('OTLP_ENDPOINT', 'http://localhost:4318/v1/traces')

    resource = Resource.create({"service.name": service})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    exporter = OTLPSpanExporter(endpoint=endpoint)
    span_processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(span_processor)

    # 自动对 Flask 进行追踪注入
    if app is not None:
        FlaskInstrumentor().instrument_app(app)
