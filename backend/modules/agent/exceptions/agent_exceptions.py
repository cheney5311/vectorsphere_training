"""智能体异常类

定义智能体相关的异常类。
"""


class AgentNotFoundError(Exception):
    """智能体未找到异常"""
    
    def __init__(self, message: str = None, agent_id: str = None):
        if agent_id and not message:
            self.message = f"智能体未找到: {agent_id}"
        else:
            self.message = message or "智能体未找到"
        self.agent_id = agent_id
        super().__init__(self.message)


class AgentValidationError(Exception):
    """智能体验证异常"""
    
    def __init__(self, message: str, validation_errors: list = None):
        self.message = message
        self.validation_errors = validation_errors or []
        super().__init__(self.message)


class AgentExecutionError(Exception):
    """智能体执行异常"""
    
    def __init__(self, message: str = None, agent_id: str = None, cause: Exception = None):
        if agent_id and not message:
            self.message = f"智能体执行失败: {agent_id}"
        else:
            self.message = message or "智能体执行失败"
        self.agent_id = agent_id
        self.cause = cause
        super().__init__(self.message)


class AgentConfigError(Exception):
    """智能体配置异常"""
    
    def __init__(self, message: str, config_errors: list = None):
        self.message = message
        self.config_errors = config_errors or []
        super().__init__(self.message)


class AgentPermissionError(Exception):
    """智能体权限异常"""
    
    def __init__(self, agent_id: str, user_id: str, action: str, message: str = None):
        self.agent_id = agent_id
        self.user_id = user_id
        self.action = action
        self.message = message or f"用户 {user_id} 无权限对智能体 {agent_id} 执行 {action} 操作"
        super().__init__(self.message)


class AgentTimeoutError(Exception):
    """智能体超时异常"""
    
    def __init__(self, agent_id: str, timeout: int, message: str = None):
        self.agent_id = agent_id
        self.timeout = timeout
        self.message = message or f"智能体 {agent_id} 执行超时: {timeout}秒"
        super().__init__(self.message)


class AgentResourceError(Exception):
    """智能体资源异常"""
    
    def __init__(self, agent_id: str, resource_type: str, message: str = None):
        self.agent_id = agent_id
        self.resource_type = resource_type
        self.message = message or f"智能体 {agent_id} 资源不足: {resource_type}"
        super().__init__(self.message)