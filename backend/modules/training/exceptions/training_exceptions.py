"""训练异常类

定义训练过程中可能发生的异常。

异常层次结构:
├── TrainingException (基础训练异常)
│   ├── TrainingError (训练执行错误)
│   ├── ValidationError (验证错误)
│   ├── BusinessLogicError (业务逻辑错误)
│   ├── TrainingJobNotFoundException
│   ├── TrainingJobAlreadyExistsException
│   ├── TrainingJobNotRunningException
│   ├── TrainingConfigInvalidException
│   ├── TrainingResourceInsufficientException
│   └── TrainingScenarioNotSupportedException
"""


class TrainingException(Exception):
    """训练基础异常类"""
    
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


# ==================== 通用异常类 ====================

class TrainingError(TrainingException):
    """训练执行错误
    
    用于表示训练过程中的执行错误，如模型训练失败、数据加载失败等。
    """
    
    def __init__(self, message: str, error_code: str = "TRAINING_ERROR"):
        super().__init__(message, error_code)


class ValidationError(TrainingException):
    """验证错误
    
    用于表示参数验证、配置验证失败等情况。
    """
    
    def __init__(self, message: str, error_code: str = "VALIDATION_ERROR"):
        super().__init__(f"验证错误: {message}", error_code)


class BusinessLogicError(TrainingException):
    """业务逻辑错误
    
    用于表示业务逻辑层面的错误，如状态不一致、操作不允许等。
    """
    
    def __init__(self, message: str, error_code: str = "BUSINESS_LOGIC_ERROR"):
        super().__init__(message, error_code)


# ==================== 任务相关异常 ====================

class TrainingJobNotFoundException(TrainingException):
    """训练任务未找到异常"""
    
    def __init__(self, job_id: str):
        super().__init__(f"训练任务未找到: {job_id}", "JOB_NOT_FOUND")


class TrainingJobAlreadyExistsException(TrainingException):
    """训练任务已存在异常"""
    
    def __init__(self, job_id: str):
        super().__init__(f"训练任务已存在: {job_id}", "JOB_ALREADY_EXISTS")


class TrainingJobNotRunningException(TrainingException):
    """训练任务未运行异常"""
    
    def __init__(self, job_id: str):
        super().__init__(f"训练任务未运行: {job_id}", "JOB_NOT_RUNNING")


# ==================== 配置和资源相关异常 ====================

class TrainingConfigInvalidException(TrainingException):
    """训练配置无效异常"""
    
    def __init__(self, message: str):
        super().__init__(f"训练配置无效: {message}", "CONFIG_INVALID")


class TrainingResourceInsufficientException(TrainingException):
    """训练资源不足异常"""
    
    def __init__(self, resource_type: str):
        super().__init__(f"训练资源不足: {resource_type}", "RESOURCE_INSUFFICIENT")


class TrainingScenarioNotSupportedException(TrainingException):
    """训练场景不支持异常"""
    
    def __init__(self, scenario_type: str):
        super().__init__(f"训练场景不支持: {scenario_type}", "SCENARIO_NOT_SUPPORTED")


# ==================== 模型相关异常 ====================

class ModelLoadError(TrainingException):
    """模型加载错误"""
    
    def __init__(self, model_path: str, reason: str = None):
        message = f"模型加载失败: {model_path}"
        if reason:
            message += f", 原因: {reason}"
        super().__init__(message, "MODEL_LOAD_ERROR")


class ModelSaveError(TrainingException):
    """模型保存错误"""
    
    def __init__(self, model_path: str, reason: str = None):
        message = f"模型保存失败: {model_path}"
        if reason:
            message += f", 原因: {reason}"
        super().__init__(message, "MODEL_SAVE_ERROR")


# ==================== 数据相关异常 ====================

class DataLoadError(TrainingException):
    """数据加载错误"""
    
    def __init__(self, data_path: str, reason: str = None):
        message = f"数据加载失败: {data_path}"
        if reason:
            message += f", 原因: {reason}"
        super().__init__(message, "DATA_LOAD_ERROR")


class DataValidationError(TrainingException):
    """数据验证错误"""
    
    def __init__(self, message: str):
        super().__init__(f"数据验证失败: {message}", "DATA_VALIDATION_ERROR")


# ==================== 分布式训练异常 ====================

class DistributedTrainingError(TrainingException):
    """分布式训练错误"""
    
    def __init__(self, message: str):
        super().__init__(f"分布式训练错误: {message}", "DISTRIBUTED_ERROR")


class CommunicationError(TrainingException):
    """通信错误"""
    
    def __init__(self, message: str):
        super().__init__(f"通信错误: {message}", "COMMUNICATION_ERROR")