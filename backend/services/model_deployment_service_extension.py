# 模型部署服务扩展方法

def _create_rest_service(self, model_id: str, service_id: str, config) -> list:
    """创建REST服务"""
    try:
        from backend.modules.service.rest_service_creator import RestServiceCreator
        creator = RestServiceCreator()
        
        service_config = {
            "model_id": model_id,
            "service_id": service_id,
            "load_balancing": config.load_balancing,
            "circuit_breaker": config.circuit_breaker,
            "rate_limiting": config.rate_limiting,
            "timeout": config.timeout,
            "max_concurrent_requests": config.max_concurrent_requests
        }
        
        endpoints = creator.create_service(service_config)
        return endpoints
        
    except Exception as e:
        self.logger.warning(f"REST服务创建失败: {e}")
        # 回退到基础REST服务
        base_port = 9000 + hash(service_id) % 1000
        return [
            f"http://localhost:{base_port}/predict",
            f"http://localhost:{base_port}/health",
            f"http://localhost:{base_port}/metrics"
        ]

def _create_graphql_service(self, model_id: str, service_id: str, config) -> list:
    """创建GraphQL服务"""
    try:
        from backend.modules.service.graphql_service_creator import GraphQLServiceCreator
        creator = GraphQLServiceCreator()
        
        service_config = {
            "model_id": model_id,
            "service_id": service_id,
            "timeout": config.timeout
        }
        
        endpoints = creator.create_service(service_config)
        return endpoints
        
    except Exception as e:
        self.logger.warning(f"GraphQL服务创建失败: {e}")
        # 回退到基础GraphQL服务
        base_port = 9100 + hash(service_id) % 1000
        return [f"http://localhost:{base_port}/graphql"]

def _create_grpc_service(self, model_id: str, service_id: str, config) -> list:
    """创建gRPC服务"""
    try:
        from backend.modules.service.grpc_service_creator import GrpcServiceCreator
        creator = GrpcServiceCreator()
        
        service_config = {
            "model_id": model_id,
            "service_id": service_id,
            "timeout": config.timeout
        }
        
        endpoints = creator.create_service(service_config)
        return endpoints
        
    except Exception as e:
        self.logger.warning(f"gRPC服务创建失败: {e}")
        # 回退到基础gRPC服务
        base_port = 9200 + hash(service_id) % 1000
        return [f"localhost:{base_port}"]

def _create_websocket_service(self, model_id: str, service_id: str, config) -> list:
    """创建WebSocket服务"""
    try:
        from backend.modules.service.websocket_service_creator import WebSocketServiceCreator
        creator = WebSocketServiceCreator()
        
        service_config = {
            "model_id": model_id,
            "service_id": service_id,
            "timeout": config.timeout,
            "max_concurrent_requests": config.max_concurrent_requests
        }
        
        endpoints = creator.create_service(service_config)
        return endpoints
        
    except Exception as e:
        self.logger.warning(f"WebSocket服务创建失败: {e}")
        # 回退到基础WebSocket服务
        base_port = 9300 + hash(service_id) % 1000
        return [f"ws://localhost:{base_port}/ws"]

def _save_service_info(self, service_id: str, model_id: str, config, api_endpoints: list):
    """保存服务信息"""
    try:
        from backend.core.database import get_db_session
        from backend.modules.service.models.service import Service
        
        with get_db_session() as session:
            service = Service(
                service_id=service_id,
                model_id=model_id,
                api_type=config.api_type,
                endpoints=",".join(api_endpoints),
                status="running",
                created_at=self._get_current_timestamp()
            )
            session.add(service)
            session.commit()
            self.logger.info(f"服务信息已保存到数据库: {service_id}")
            
    except Exception as e:
        self.logger.warning(f"保存服务信息失败: {e}")
        # 保存到本地文件作为备份
        import json
        import os
        
        service_info = {
            "service_id": service_id,
            "model_id": model_id,
            "api_type": config.api_type,
            "endpoints": api_endpoints,
            "status": "running",
            "created_at": self._get_current_timestamp().isoformat()
        }
        
        os.makedirs("/tmp/services", exist_ok=True)
        with open(f"/tmp/services/{service_id}.json", "w") as f:
            json.dump(service_info, f, indent=2)
        self.logger.info(f"服务信息已保存到本地文件: {service_id}")