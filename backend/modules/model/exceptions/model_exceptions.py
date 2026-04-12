"""模型异常类

定义模型相关的异常类。
"""


class ModelNotFoundError(Exception):
    """模型未找到异常"""
    
    def __init__(self, model_id: str, message: str = None):
        self.model_id = model_id
        self.message = message or f"模型未找到: {model_id}"
        super().__init__(self.message)


class ModelValidationError(Exception):
    """模型验证异常"""
    
    def __init__(self, message: str, validation_errors: list = None):
        self.message = message
        self.validation_errors = validation_errors or []
        super().__init__(self.message)


class ModelLoadError(Exception):
    """模型加载异常"""
    
    def __init__(self, model_id: str, message: str = None, cause: Exception = None):
        self.model_id = model_id
        self.message = message or f"模型加载失败: {model_id}"
        self.cause = cause
        super().__init__(self.message)


class ModelSaveError(Exception):
    """模型保存异常"""
    
    def __init__(self, model_id: str, message: str = None, cause: Exception = None):
        self.model_id = model_id
        self.message = message or f"模型保存失败: {model_id}"
        self.cause = cause
        super().__init__(self.message)


class ModelConfigError(Exception):
    """模型配置异常"""
    
    def __init__(self, message: str, config_errors: list = None):
        self.message = message
        self.config_errors = config_errors or []
        super().__init__(self.message)


class ModelVersionError(Exception):
    """模型版本异常"""
    
    def __init__(self, model_id: str, version: str, message: str = None):
        self.model_id = model_id
        self.version = version
        self.message = message or f"模型版本错误: {model_id} v{version}"
        super().__init__(self.message)


class ModelPermissionError(Exception):
    """模型权限异常"""
    
    def __init__(self, model_id: str, user_id: str, action: str, message: str = None):
        self.model_id = model_id
        self.user_id = user_id
        self.action = action
        self.message = message or f"用户 {user_id} 无权限对模型 {model_id} 执行 {action} 操作"
        super().__init__(self.message)