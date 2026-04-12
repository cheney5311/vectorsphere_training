"""训练模块API

提供训练相关的API接口。
"""

from .training_api import training_bp
from .training_jobs_api import training_jobs_bp
from .training_progress_api import training_progress_bp
from .three_stage_training_api import three_stage_training_bp
from .training_history_api import training_history_bp
from .training_statistics_api import training_statistics_bp

# 新增的API
from .hyperparameter_optimization_api import hyperparameter_optimization_bp
from .model_selection_api import model_selection_bp
from .training_execution_api import training_execution_bp
from .model_evaluation_api import model_evaluation_bp
from .model_optimization_api import model_optimization_bp
from .model_deployment_api import model_deployment_bp
from .monitoring_operations_api import monitoring_operations_bp
from .intelligent_decision_api import intelligent_decision_bp

__all__ = [
    'training_bp',
    'training_jobs_bp',
    'training_progress_bp',
    'three_stage_training_bp',
    'training_history_bp',
    'training_statistics_bp',
    'hyperparameter_optimization_bp',
    'model_selection_bp',
    'training_execution_bp',
    'model_evaluation_bp',
    'model_optimization_bp',
    'model_deployment_bp',
    'monitoring_operations_bp',
    'intelligent_decision_bp'
]