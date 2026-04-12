"""训练异常模块

提供训练过程中可能发生的异常类。

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
│   ├── TrainingScenarioNotSupportedException
│   ├── ModelLoadError
│   ├── ModelSaveError
│   ├── DataLoadError
│   ├── DataValidationError
│   ├── DistributedTrainingError
│   └── CommunicationError
"""

from .training_exceptions import (
    # 基础异常
    TrainingException,
    
    # 通用异常
    TrainingError,
    ValidationError,
    BusinessLogicError,
    
    # 任务异常
    TrainingJobNotFoundException,
    TrainingJobAlreadyExistsException,
    TrainingJobNotRunningException,
    
    # 配置和资源异常
    TrainingConfigInvalidException,
    TrainingResourceInsufficientException,
    TrainingScenarioNotSupportedException,
    
    # 模型异常
    ModelLoadError,
    ModelSaveError,
    
    # 数据异常
    DataLoadError,
    DataValidationError,
    
    # 分布式异常
    DistributedTrainingError,
    CommunicationError
)

__all__ = [
    # 基础异常
    'TrainingException',
    
    # 通用异常
    'TrainingError',
    'ValidationError',
    'BusinessLogicError',
    
    # 任务异常
    'TrainingJobNotFoundException',
    'TrainingJobAlreadyExistsException',
    'TrainingJobNotRunningException',
    
    # 配置和资源异常
    'TrainingConfigInvalidException',
    'TrainingResourceInsufficientException',
    'TrainingScenarioNotSupportedException',
    
    # 模型异常
    'ModelLoadError',
    'ModelSaveError',
    
    # 数据异常
    'DataLoadError',
    'DataValidationError',
    
    # 分布式异常
    'DistributedTrainingError',
    'CommunicationError'
]