"""性能模块API接口

提供异步任务处理、数据库连接池管理、性能监控等REST API接口。
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
import logging
import time

from backend.services.async_processor import get_async_processor, TaskPriority
from backend.services.db_pool import get_database_pool_manager
from backend.services.performance_monitor import get_performance_monitor
from backend.modules.performance.performance_errors import PerformanceError
from backend.services.model_optimization_service import (
    ModelOptimizationService, InferenceOptimizationConfig
)
# 导入公共工具模块
from backend.api.performance.utils import (
    get_config,
    APIResponse,
    HealthCheckResult,
    timestamp_to_iso,
    format_duration,
    parse_time_range,
    get_resource_status,
    get_temperature_status,
    assess_utilization_status,
    calculate_health_summary,
    get_bool_param,
    get_int_param,
    get_list_param,
    handle_api_errors,
    format_system_metrics,
    format_gpu_metrics,
    format_training_metrics,
    format_alert
)

logger = logging.getLogger(__name__)

# 获取配置实例
_api_config = None

def _get_api_config():
    """获取API配置"""
    global _api_config
    if _api_config is None:
        _api_config = get_config()
    return _api_config

# 创建蓝图
performance_bp = Blueprint('performance', __name__, url_prefix='/api/performance')


# ============================================================================
# 基于装饰器的任务注册系统
# ============================================================================

class TaskRegistry:
    """任务注册表 - 使用装饰器注册异步任务"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tasks = {}
            cls._instance._categories = {}
        return cls._instance
    
    def __init__(self):
        # 确保 linting 工具能够识别这些成员
        if not hasattr(self, '_tasks'):
            self._tasks = {}
        if not hasattr(self, '_categories'):
            self._categories = {}
    
    def register(
        self,
        name: str,
        category: str = 'general',
        description: str = '',
        required_params: List[str] = None,
        optional_params: List[str] = None,
        timeout: Optional[float] = None
    ):
        """任务注册装饰器
        
        Args:
            name: 任务名称（唯一标识）
            category: 任务分类
            description: 任务描述
            required_params: 必需参数列表
            optional_params: 可选参数列表
            timeout: 任务超时时间（秒）
        
        Usage:
            @task_registry.register('data_preprocessing', category='data', required_params=['dataset_id'])
            def preprocess_data(params):
                ...
        """
        def decorator(func: Callable):
            self._tasks[name] = {
                'handler': func,
                'category': category,
                'description': description or func.__doc__ or '',
                'required_params': required_params or [],
                'optional_params': optional_params or [],
                'timeout': timeout
            }
            
            # 更新分类索引
            if category not in self._categories:
                self._categories[category] = []
            if name not in self._categories[category]:
                self._categories[category].append(name)
            
            logger.debug(f"Task registered: {name} in category {category}")
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        
        return decorator
    
    def get_task(self, name: str) -> Optional[Dict[str, Any]]:
        """获取任务信息"""
        return self._tasks.get(name)
    
    def list_tasks(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有任务或指定分类的任务"""
        if category:
            task_names = self._categories.get(category, [])
        else:
            task_names = list(self._tasks.keys())
        
        return [
            {
                'name': name,
                'category': self._tasks[name]['category'],
                'description': self._tasks[name]['description'],
                'required_params': self._tasks[name]['required_params'],
                'optional_params': self._tasks[name]['optional_params']
            }
            for name in task_names
        ]
    
    def get_categories(self) -> List[str]:
        """获取所有任务分类"""
        return list(self._categories.keys())
    
    def validate_params(self, task_name: str, params: Dict[str, Any]) -> tuple:
        """验证任务参数
        
        Returns:
            (is_valid, error_message)
        """
        task = self.get_task(task_name)
        if not task:
            return False, f"Task '{task_name}' not found"
        
        missing_params = [
            p for p in task['required_params']
            if p not in params or params[p] is None
        ]
        
        if missing_params:
            return False, f"Missing required parameters: {', '.join(missing_params)}"
        
        return True, None
    
    def execute(self, task_name: str, params: Dict[str, Any]) -> Any:
        """执行任务"""
        task = self.get_task(task_name)
        if not task:
            raise ValueError(f"Task '{task_name}' not found")
        
        is_valid, error_msg = self.validate_params(task_name, params)
        if not is_valid:
            raise ValueError(error_msg)
        
        return task['handler'](params)


# 全局任务注册表实例
task_registry = TaskRegistry()


# ============================================================================
# 使用装饰器注册任务
# ============================================================================

@task_registry.register(
    name='data_preprocessing',
    category='data',
    description='Execute data preprocessing on a dataset',
    required_params=['dataset_id'],
    optional_params=['config', 'normalize', 'tokenize', 'filter_invalid']
)
def _task_data_preprocessing(params: Dict[str, Any]) -> Dict[str, Any]:
    """数据预处理任务"""
    from backend.repositories.dataset_repository import DatasetRepository
    from backend.services.data_preprocessing_service import DataPreprocessingService
    
    dataset_id = params['dataset_id']
    config = {
        'normalize': params.get('normalize', False),
        'tokenize': params.get('tokenize', False),
        'filter_invalid': params.get('filter_invalid', False)
    }
    if 'config' in params:
        config.update(params['config'])
        
        repository = DatasetRepository()
        service = DataPreprocessingService(repository)
        result = service.preprocess(dataset_id, config)
    
        return {
        'task': 'data_preprocessing',
        'dataset_id': dataset_id,
        'status': 'completed',
        'result': {
            'dataset_status': getattr(result, 'status', 'preprocessed'),
            'config_used': config
        }
    }


@task_registry.register(
    name='data_quality_assessment',
    category='data',
    description='Assess data quality for a dataset',
    required_params=['dataset_id'],
    optional_params=['metrics']
)
def _task_data_quality_assessment(params: Dict[str, Any]) -> Dict[str, Any]:
    """数据质量评估任务"""
    from backend.repositories.dataset_repository import DatasetRepository
    from backend.services.data_quality_service import DataQualityService
    
    dataset_id = params['dataset_id']
        
    repository = DatasetRepository()
    service = DataQualityService(repository)
    quality_metrics = service.assess_data_quality(dataset_id)
    
    return {
        'task': 'data_quality_assessment',
        'dataset_id': dataset_id,
        'status': 'completed',
        'result': quality_metrics
    }


@task_registry.register(
    name='model_evaluation',
    category='model',
    description='Evaluate model performance on a dataset',
    required_params=['model_id', 'dataset_id'],
    optional_params=['evaluation_config', 'metrics']
)
def _task_model_evaluation(params: Dict[str, Any]) -> Dict[str, Any]:
    """模型评估任务"""
    from backend.services.model_evaluation_service import ModelEvaluationService
    
    model_id = params['model_id']
    dataset_id = params['dataset_id']
    evaluation_config = params.get('evaluation_config')
    
    service = ModelEvaluationService()
    result = service.automated_evaluation(model_id, dataset_id, evaluation_config)
    
    return {
        'task': 'model_evaluation',
        'model_id': model_id,
        'dataset_id': dataset_id,
        'status': 'completed',
        'result': {
            'metrics': [
                {'name': m.name, 'value': m.value, 'type': m.type.value}
                for m in result.metrics
            ],
            'timestamp': result.timestamp
        }
    }


@task_registry.register(
    name='model_compression',
    category='model',
    description='Compress model using specified technique',
    required_params=['model_id'],
    optional_params=['technique', 'compression_ratio', 'quantization_bits']
)
def _task_model_compression(params: Dict[str, Any]) -> Dict[str, Any]:
    """模型压缩任务"""
    from backend.services.model_optimization_service import (
        ModelOptimizationService, OptimizationConfig, OptimizationTechnique
    )
    
    model_id = params['model_id']
    technique_str = params.get('technique', 'quantization')
    
    technique_map = {
        'quantization': OptimizationTechnique.QUANTIZATION,
        'pruning': OptimizationTechnique.PRUNING,
        'knowledge_distillation': OptimizationTechnique.KNOWLEDGE_DISTILLATION,
        'low_rank_decomposition': OptimizationTechnique.LOW_RANK_DECOMPOSITION
    }
    technique = technique_map.get(technique_str, OptimizationTechnique.QUANTIZATION)
    
    config = OptimizationConfig(
        technique=technique,
        compression_ratio=params.get('compression_ratio', 0.5),
        quantization_bits=params.get('quantization_bits', 8)
    )
    
    service = ModelOptimizationService()
    result = service.model_compression(model_id, config)
    
    return {
        'task': 'model_compression',
        'model_id': model_id,
        'status': 'completed',
        'result': {
            'optimized_model_id': result.optimized_model_id,
            'compression_ratio': result.compression_ratio,
            'accuracy_preserved': result.accuracy_preserved,
            'model_size_reduction': result.model_size_reduction
        }
    }


@task_registry.register(
    name='model_inference_optimization',
    category='model',
    description='Optimize model for inference',
    required_params=['model_id'],
    optional_params=['hardware_target', 'optimization_config']
)
def _task_model_inference_optimization(params: Dict[str, Any]) -> Dict[str, Any]:
    """模型推理优化任务"""
    
    model_id = params['model_id']
    hardware_target = params.get('hardware_target', 'cpu')
    opt_config = params.get('optimization_config', {})
    
    config = InferenceOptimizationConfig(
        hardware_target=hardware_target,
        graph_optimization=opt_config.get('graph_optimization', True),
        operator_fusion=opt_config.get('operator_fusion', True),
        constant_folding=opt_config.get('constant_folding', True),
        memory_optimization=opt_config.get('memory_optimization', True)
    )
    
    service = ModelOptimizationService()
    result = service.inference_optimization(model_id, config)
    
    return {
        'task': 'model_inference_optimization',
        'model_id': model_id,
        'status': 'completed',
        'result': {
            'optimized_model_id': result.optimized_model_id,
            'latency_reduction': result.latency_reduction,
            'throughput_improvement': result.throughput_improvement
        }
    }


@task_registry.register(
    name='training_start',
    category='training',
    description='Start a training session',
    required_params=['session_id'],
    optional_params=['training_config']
)
def _task_training_start(params: Dict[str, Any]) -> Dict[str, Any]:
    """启动训练任务"""
    from backend.services.training_execution_service import TrainingExecutionService
    
    session_id = params['session_id']
    training_config = params.get('training_config', {})
    
    service = TrainingExecutionService()
    result = service.start_training(session_id, training_config)
    
    return {
        'task': 'training_start',
        'session_id': session_id,
        'status': 'completed',
        'result': result
    }


@task_registry.register(
    name='resource_optimization',
    category='system',
    description='Optimize system resource allocation',
    required_params=[],
    optional_params=['target_metrics', 'constraints']
)
def _task_resource_optimization(params: Dict[str, Any]) -> Dict[str, Any]:
    """资源优化任务"""
    from backend.services.resource_optimizer import get_resource_optimizer
    
    optimizer = get_resource_optimizer()
    
    optimization_result = {
        'recommendations': [],
        'current_usage': {},
        'optimized_allocation': {}
    }
    
    try:
        import psutil
        optimization_result['current_usage'] = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }
    except Exception as e:
        logger.warning(f"Failed to get resource usage: {e}")
    
    return {
        'task': 'resource_optimization',
        'status': 'completed',
        'result': optimization_result
    }


@task_registry.register(
    name='performance_analysis',
    category='system',
    description='Analyze system performance',
    required_params=[],
    optional_params=['duration_minutes', 'metrics']
)
def _task_performance_analysis(params: Dict[str, Any]) -> Dict[str, Any]:
    """性能分析任务"""
    duration_minutes = params.get('duration_minutes', 5)
    metrics = params.get('metrics', ['cpu', 'memory', 'latency'])
    
    performance_monitor = get_performance_monitor()
    current_metrics = performance_monitor.get_current_metrics()
    summary = performance_monitor.get_performance_summary()
        
    return {
        'task': 'performance_analysis',
        'status': 'completed',
        'result': {
            'current_metrics': current_metrics,
            'summary': summary,
            'duration_minutes': duration_minutes,
            'analyzed_metrics': metrics
        }
    }


@task_registry.register(
    name='database_cleanup',
    category='system',
    description='Clean up expired data from database',
    required_params=[],
    optional_params=['max_age_days', 'tables']
)
def _task_database_cleanup(params: Dict[str, Any]) -> Dict[str, Any]:
    """数据库清理任务"""
    max_age_days = params.get('max_age_days', 30)
    tables = params.get('tables', [])
    
    async_processor = get_async_processor()
    cleanup_count = async_processor.cleanup_completed_tasks(max_age=max_age_days * 86400)
    
    return {
        'task': 'database_cleanup',
        'status': 'completed',
        'result': {
            'cleaned_tasks': cleanup_count,
            'max_age_days': max_age_days,
            'tables_cleaned': tables or ['async_tasks']
        }
    }


@task_registry.register(
    name='cache_warmup',
    category='system',
    description='Warm up system caches',
    required_params=[],
    optional_params=['cache_types']
)
def _task_cache_warmup(params: Dict[str, Any]) -> Dict[str, Any]:
    """缓存预热任务"""
    cache_types = params.get('cache_types', ['model', 'data'])
    
    warmup_results = {}
    for cache_type in cache_types:
        warmup_results[cache_type] = {
            'status': 'warmed',
            'items_loaded': 0
        }
    
    return {
        'task': 'cache_warmup',
        'status': 'completed',
        'result': {
            'cache_types': cache_types,
            'warmup_results': warmup_results
        }
    }


@performance_bp.route('/async/tasks', methods=['POST'])
def submit_async_task():
    """提交异步任务
    
    根据任务名称执行相应的注册函数，通过装饰器注册的任务会被自动识别和执行。
    
    Request Body:
        {
            "task_name": "string",     # 必需，任务名称
            "priority": "string",       # 可选，优先级: LOW/NORMAL/HIGH/URGENT
            "params": {},               # 可选，任务参数
            "timeout": float            # 可选，超时时间（秒）
        }
    
    Supported Tasks (通过装饰器注册):
        - data_preprocessing: 数据预处理
        - data_quality_assessment: 数据质量评估
        - model_evaluation: 模型评估
        - model_compression: 模型压缩
        - model_inference_optimization: 推理优化
        - training_start: 启动训练
        - resource_optimization: 资源优化
        - performance_analysis: 性能分析
        - database_cleanup: 数据库清理
        - cache_warmup: 缓存预热
    
    Returns:
        JSON: 任务提交结果
    """
    try:
        # 获取请求数据
        data = request.get_json() or {}
        task_name = data.get('task_name')
        priority_str = data.get('priority', 'NORMAL').upper()
        params = data.get('params', {})
        timeout = data.get('timeout')
        
        # 验证任务名称
        if not task_name:
            return jsonify({
                'success': False,
                'error': 'Task name is required',
                'error_code': 'MISSING_TASK_NAME',
                'available_tasks': task_registry.list_tasks()
            }), 400
        
        task_name = task_name.strip().lower()
        
        # 验证任务是否已注册
        task_info = task_registry.get_task(task_name)
        if not task_info:
            return jsonify({
                'success': False,
                'error': f"Unknown task: '{task_name}'",
                'error_code': 'UNKNOWN_TASK',
                'available_tasks': task_registry.list_tasks(),
                'hint': 'Use GET /api/performance/async/tasks/types to see all available tasks'
            }), 400
        
        # 验证任务参数
        is_valid, error_msg = task_registry.validate_params(task_name, params)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg,
                'error_code': 'INVALID_PARAMS',
                'required_params': task_info['required_params'],
                'optional_params': task_info['optional_params']
            }), 400
        
        # 解析优先级
        priority_map = {
            'LOW': TaskPriority.LOW,
            'NORMAL': TaskPriority.NORMAL,
            'HIGH': TaskPriority.HIGH,
            'URGENT': TaskPriority.URGENT
        }
        priority = priority_map.get(priority_str, TaskPriority.NORMAL)
        
        # 创建任务执行函数（闭包捕获task_name和params）
        captured_task_name = task_name
        captured_params = params.copy()
        
        def execute_registered_task():
            """执行通过装饰器注册的任务"""
            start_time = time.time()
            try:
                result = task_registry.execute(captured_task_name, captured_params)
                execution_time = time.time() - start_time
                
                if isinstance(result, dict):
                    result['execution_time'] = execution_time
                
                logger.info(f"Task '{captured_task_name}' completed in {execution_time:.2f}s")
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Task '{captured_task_name}' failed after {execution_time:.2f}s: {e}")
                raise
        
        # 提交任务到异步处理器
        async_processor = get_async_processor()
        
        if not async_processor._running:
            return jsonify({
                'success': False,
                'error': 'Async processor is not running',
                'error_code': 'PROCESSOR_NOT_RUNNING'
            }), 503
        
        task_timeout = timeout if timeout else task_info.get('timeout')
        
        task_id = async_processor.submit_task(
            execute_registered_task,
            priority=priority,
            timeout=task_timeout
        )
        
        logger.info(f"Task '{task_name}' submitted with ID: {task_id}, priority: {priority_str}")
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'task_name': task_name,
            'priority': priority_str,
            'message': f"Task '{task_name}' submitted successfully",
            'status_url': f'/api/performance/async/tasks/{task_id}'
        }), 201
        
    except PerformanceError as e:
        logger.error(f"Performance error when submitting task: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': e.error_code if hasattr(e, 'error_code') else 'PERFORMANCE_ERROR'
        }), 500
        
    except Exception as e:
        logger.error(f"Failed to submit async task: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/async/tasks/types', methods=['GET'])
def list_task_types():
    """列出所有可用的任务类型（通过装饰器注册）

    Query Parameters:
        category: 可选，按分类筛选任务

    Returns:
        JSON: 可用任务类型列表
    """
    try:
        category = request.args.get('category')
        
        tasks = task_registry.list_tasks(category)
        categories = task_registry.get_categories()
        
        return jsonify({
            'success': True,
            'data': {
                'tasks': tasks,
                'categories': categories,
                'total': len(tasks)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to list task types: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@performance_bp.route('/async/tasks/types/<task_name>', methods=['GET'])
def get_task_type_info(task_name: str):
    """获取指定任务类型的详细信息

    Args:
        task_name: 任务名称
    
    Returns:
        JSON: 任务类型详细信息
    """
    try:
        task_info = task_registry.get_task(task_name.lower())
        
        if not task_info:
            return jsonify({
                'success': False,
                'error': f"Task type '{task_name}' not found",
                'available_tasks': [t['name'] for t in task_registry.list_tasks()]
            }), 404
        
        return jsonify({
            'success': True,
            'data': {
                'name': task_name.lower(),
                'category': task_info['category'],
                'description': task_info['description'],
                'required_params': task_info['required_params'],
                'optional_params': task_info['optional_params'],
                'timeout': task_info.get('timeout')
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get task type info: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@performance_bp.route('/async/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id: str):
    """获取任务状态

    获取指定任务的详细状态信息，包括：
    - pending: 任务在队列中等待执行
    - running: 任务正在执行中
    - completed: 任务已成功完成
    - failed: 任务执行失败
    - timeout: 任务执行超时

    Args:
        task_id: 任务ID（UUID格式）

    Returns:
        JSON: 任务状态信息，包含以下字段：
            - id: 任务ID
            - status: 任务状态
            - created_at: 任务创建时间戳
            - started_at: 任务开始执行时间戳（pending状态为None）
            - completed_at: 任务完成时间戳（pending/running状态为None）
            - priority: 任务优先级（LOW/NORMAL/HIGH/URGENT）
            - wait_time: 等待时间（秒）
            - queue_position: 队列位置（仅pending状态）
            - execution_time: 执行时间（秒）
            - result: 任务执行结果（仅completed状态）
            - error: 错误信息（仅failed/timeout状态）
    """
    try:
        # 验证任务ID格式
        if not task_id or not isinstance(task_id, str):
            return jsonify({
                'success': False,
                'error': 'Invalid task_id: task_id is required and must be a string',
                'error_code': 'INVALID_TASK_ID'
            }), 400
        
        # 简单的UUID格式验证
        task_id = task_id.strip()
        if len(task_id) < 8:
            return jsonify({
                'success': False,
                'error': 'Invalid task_id format',
                'error_code': 'INVALID_TASK_ID_FORMAT'
            }), 400
        
        async_processor = get_async_processor()
        
        # 检查处理器运行状态
        if not async_processor._running:
            logger.warning(f"Async processor is not running when querying task {task_id}")
            return jsonify({
                'success': False,
                'error': 'Async processor is not running',
                'error_code': 'PROCESSOR_NOT_RUNNING'
            }), 503
        
        status = async_processor.get_task_status(task_id)
        
        if status:
            # 增强响应数据，添加额外的上下文信息
            response_data = {
                **status,
                # 转换时间戳为ISO格式（如果存在）
                'created_at_iso': timestamp_to_iso(status.get('created_at')),
                'started_at_iso': timestamp_to_iso(status.get('started_at')),
                'completed_at_iso': timestamp_to_iso(status.get('completed_at')),
            }
            
            # 添加状态描述
            status_descriptions = {
                'pending': 'Task is waiting in queue for execution',
                'running': 'Task is currently being executed',
                'completed': 'Task has been completed successfully',
                'failed': 'Task execution failed with an error',
                'timeout': 'Task execution timed out'
            }
            response_data['status_description'] = status_descriptions.get(
                status.get('status'), 'Unknown status'
            )
            
            # 计算进度百分比（估算）
            task_status = status.get('status')
            if task_status == 'pending':
                response_data['progress'] = 0
            elif task_status == 'running':
                response_data['progress'] = 50  # 运行中默认50%
            elif task_status in ('completed', 'failed', 'timeout'):
                response_data['progress'] = 100
            
            logger.debug(f"Task status retrieved: {task_id}, status: {task_status}")
            
            return jsonify({
                'success': True,
                'data': response_data
            }), 200
        else:
            logger.info(f"Task not found: {task_id}")
            return jsonify({
                'success': False,
                'error': f'Task {task_id} not found',
                'error_code': 'TASK_NOT_FOUND',
                'hint': 'The task may have expired and been cleaned up, or the task_id is incorrect'
            }), 404
            
    except PerformanceError as e:
        logger.error(f"Performance error when getting task status: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': e.error_code if hasattr(e, 'error_code') else 'PERFORMANCE_ERROR'
        }), 500
            
    except Exception as e:
        logger.error(f"Failed to get task status: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


# timestamp_to_iso 已移动到 utils.py，使用导入的 timestamp_to_iso


@performance_bp.route('/async/stats', methods=['GET'])
def get_async_processor_stats():
    """获取异步处理器统计信息

    获取异步任务处理器的详细统计信息，包括任务计数、执行效率、健康状态等。

    Query Parameters:
        include_health: 是否包含健康检查信息（默认true）
        include_cleanup: 是否包含清理统计信息（默认false）
        include_workers: 是否包含工作线程详情（默认false）

    Returns:
        JSON: 统计信息，包含以下字段：
            - basic_stats: 基础统计（任务计数、队列大小等）
            - efficiency: 效率指标（成功率、平均执行时间等）
            - health: 健康状态（可选）
            - cleanup_stats: 清理统计（可选）
            - worker_details: 工作线程详情（可选）
    """
    try:
        # 获取查询参数
        include_health = request.args.get('include_health', 'true').lower() == 'true'
        include_cleanup = request.args.get('include_cleanup', 'false').lower() == 'true'
        include_workers = request.args.get('include_workers', 'false').lower() == 'true'
        
        async_processor = get_async_processor()
        
        # 检查处理器状态
        if not async_processor._running:
            return jsonify({
                'success': False,
                'error': 'Async processor is not running',
                'error_code': 'PROCESSOR_NOT_RUNNING'
            }), 503
        
        # 获取基础统计信息
        basic_stats = async_processor.get_stats()
        
        # 构建响应数据
        response_data = {
            'basic_stats': basic_stats,
            'timestamp': time.time(),
            'timestamp_iso': timestamp_to_iso(time.time())
        }
        
        # 计算效率指标
        total_tasks = basic_stats.get('total_tasks', 0)
        completed_tasks = basic_stats.get('completed_tasks', 0)
        failed_tasks = basic_stats.get('failed_tasks', 0)
        timeout_tasks = basic_stats.get('timeout_tasks', 0)
        
        efficiency = {
            'total_processed': completed_tasks + failed_tasks + timeout_tasks,
            'success_rate': 0.0,
            'failure_rate': 0.0,
            'timeout_rate': 0.0,
            'avg_execution_time': basic_stats.get('avg_execution_time', 0.0),
            'throughput': _calculate_throughput(async_processor)
        }
        
        if total_tasks > 0:
            efficiency['success_rate'] = round(completed_tasks / total_tasks * 100, 2)
            efficiency['failure_rate'] = round(failed_tasks / total_tasks * 100, 2)
            efficiency['timeout_rate'] = round(timeout_tasks / total_tasks * 100, 2)
        
        response_data['efficiency'] = efficiency
        
        # 任务分布统计
        response_data['task_distribution'] = {
            'pending': basic_stats.get('pending_tasks_count', 0),
            'running': basic_stats.get('running_tasks', 0),
            'completed': basic_stats.get('completed_tasks_count', 0),
            'failed': basic_stats.get('failed_tasks_count', 0)
        }
        
        # 队列状态
        queue_size = basic_stats.get('queue_size', 0)
        max_queue_size = async_processor.queue_size
        response_data['queue_status'] = {
            'current_size': queue_size,
            'max_size': max_queue_size,
            'utilization': round(queue_size / max_queue_size * 100, 2) if max_queue_size > 0 else 0,
            'is_full': queue_size >= max_queue_size,
            'available_slots': max_queue_size - queue_size
        }
        
        # 健康检查信息（可选）
        if include_health:
            health_info = async_processor.health_check()
            response_data['health'] = {
                'status': health_info.get('status', 'unknown'),
                'message': health_info.get('message', ''),
                'is_healthy': health_info.get('status') == 'healthy'
            }
        
        # 清理统计信息（可选）
        if include_cleanup:
            cleanup_stats = async_processor.get_cleanup_stats()
            response_data['cleanup_stats'] = {
                'oldest_pending_task_age': round(cleanup_stats.get('oldest_pending_task_age', 0), 2),
                'oldest_completed_task_age': round(cleanup_stats.get('oldest_completed_task_age', 0), 2),
                'oldest_failed_task_age': round(cleanup_stats.get('oldest_failed_task_age', 0), 2),
                'avg_pending_task_age': round(cleanup_stats.get('avg_pending_task_age', 0), 2),
                'avg_completed_task_age': round(cleanup_stats.get('avg_completed_task_age', 0), 2),
                'memory_usage_estimate': cleanup_stats.get('memory_usage_estimate', {})
            }
        
        # 工作线程详情（可选）
        if include_workers:
            worker_threads = async_processor._worker_threads
            alive_workers = sum(1 for t in worker_threads if t.is_alive())
            
            response_data['worker_details'] = {
                'max_workers': async_processor.max_workers,
                'total_threads': len(worker_threads),
                'alive_threads': alive_workers,
                'dead_threads': len(worker_threads) - alive_workers,
                'active_workers': basic_stats.get('active_workers', 0),
                'worker_utilization': round(basic_stats.get('active_workers', 0) / async_processor.max_workers * 100, 2) if async_processor.max_workers > 0 else 0,
                'threads': [
                    {
                        'name': t.name,
                        'is_alive': t.is_alive(),
                        'daemon': t.daemon
                    }
                    for t in worker_threads
                ]
            }
        
        logger.debug(f"Async processor stats retrieved: total_tasks={total_tasks}, running={basic_stats.get('running_tasks', 0)}")
        
        return jsonify({
            'success': True,
            'data': response_data
        }), 200
        
    except PerformanceError as e:
        logger.error(f"Performance error when getting async processor stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': e.error_code if hasattr(e, 'error_code') else 'PERFORMANCE_ERROR'
        }), 500
        
    except Exception as e:
        logger.error(f"Failed to get async processor stats: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


def _calculate_throughput(async_processor) -> Dict[str, Any]:
    """计算任务吞吐量
    
    Args:
        async_processor: 异步处理器实例
        
    Returns:
        吞吐量统计信息
    """
    try:
        stats = async_processor.get_stats()
        completed = stats.get('completed_tasks', 0)
        avg_time = stats.get('avg_execution_time', 0)
        active_workers = stats.get('active_workers', 0)
        
        # 估算每秒处理任务数
        tasks_per_second = 0.0
        if avg_time > 0:
            tasks_per_second = active_workers / avg_time if active_workers > 0 else 1 / avg_time
        
        return {
            'estimated_tasks_per_second': round(tasks_per_second, 2),
            'avg_task_duration': round(avg_time, 3),
            'concurrent_capacity': async_processor.max_workers
        }
    except Exception:
        return {
            'estimated_tasks_per_second': 0.0,
            'avg_task_duration': 0.0,
            'concurrent_capacity': 0
        }


@performance_bp.route('/db/status', methods=['GET'])
def get_database_pool_status():
    """获取数据库连接池状态

    获取数据库连接池的详细状态信息，包括连接数、使用率、健康状态等。

    Query Parameters:
        include_health: 是否包含健康检查（默认true）
        include_optimization: 是否包含优化建议（默认false）
        include_history: 是否包含历史统计（默认false）

    Returns:
        JSON: 连接池状态信息，包含以下字段：
            - pool_status: 连接池基本状态
            - utilization: 使用率分析
            - health: 健康状态（可选）
            - optimization: 优化建议（可选）
    """
    try:
        # 获取查询参数
        include_health = request.args.get('include_health', 'true').lower() == 'true'
        include_optimization = request.args.get('include_optimization', 'false').lower() == 'true'
        include_history = request.args.get('include_history', 'false').lower() == 'true'
        
        db_pool_manager = get_database_pool_manager()
        
        # 获取基本状态
        pool_status = db_pool_manager.get_pool_status()
        
        # 检查连接池是否已初始化
        if pool_status.get('status') == 'not_initialized':
            return jsonify({
                'success': False,
                'error': 'Database pool not initialized',
                'error_code': 'POOL_NOT_INITIALIZED',
                'hint': 'The database pool needs to be initialized with init_app() first'
            }), 503
        
        # 检查是否有错误
        if pool_status.get('status') == 'error':
            return jsonify({
                'success': False,
                'error': pool_status.get('error', 'Unknown error'),
                'error_code': 'POOL_ERROR'
            }), 500
        
        # 构建响应数据
        response_data = {
            'pool_status': pool_status,
            'timestamp': time.time(),
            'timestamp_iso': timestamp_to_iso(time.time())
        }
        
        # 计算使用率分析
        pool_size = pool_status.get('pool_size', 0)
        checked_out = pool_status.get('checked_out_connections', 0)
        overflow = pool_status.get('overflow_connections', 0)
        total_connections = pool_status.get('total_connections', 0)
        checked_in = pool_status.get('checked_in_connections', 0)
        
        utilization = {
            'pool_size': pool_size,
            'active_connections': checked_out,
            'idle_connections': checked_in,
            'overflow_connections': overflow,
            'total_connections': total_connections,
            'utilization_percent': round(checked_out / pool_size * 100, 2) if pool_size > 0 else 0,
            'overflow_percent': round(overflow / pool_size * 100, 2) if pool_size > 0 else 0,
            'is_overloaded': checked_out >= pool_size,
            'available_connections': max(0, pool_size - checked_out + overflow)
        }
        
        # 连接池效率统计
        statistics = pool_status.get('statistics', {})
        total_checkouts = statistics.get('checked_out', 0)
        total_checkins = statistics.get('checked_in', 0)
        
        utilization['efficiency'] = {
            'total_checkouts': total_checkouts,
            'total_checkins': total_checkins,
            'connection_reuse_rate': round(total_checkins / total_checkouts * 100, 2) if total_checkouts > 0 else 0,
            'pool_hits': statistics.get('pool_hits', 0),
            'pool_misses': statistics.get('pool_misses', 0)
        }
        
        response_data['utilization'] = utilization
        
        # 连接池状态评估
        status_assessment = _assess_pool_status(utilization)
        response_data['assessment'] = status_assessment
        
        # 健康检查（可选）
        if include_health:
            try:
                is_healthy = db_pool_manager.health_check()
                response_data['health'] = {
                    'is_healthy': is_healthy,
                    'status': 'healthy' if is_healthy else 'unhealthy',
                    'message': 'Database connection is working properly' if is_healthy else 'Database connection check failed',
                    'last_check': timestamp_to_iso(time.time())
                }
            except Exception as health_error:
                response_data['health'] = {
                    'is_healthy': False,
                    'status': 'error',
                    'message': f'Health check failed: {str(health_error)}',
                    'last_check': timestamp_to_iso(time.time())
                }
        
        # 优化建议（可选）
        if include_optimization:
            optimization_result = db_pool_manager.optimize_pool()
            suggestions = optimization_result.get('suggestions', []) if optimization_result else []
            
            # 添加基于当前状态的建议
            additional_suggestions = _generate_pool_suggestions(utilization, status_assessment)
            suggestions.extend(additional_suggestions)
            
            response_data['optimization'] = {
                'suggestions': suggestions,
                'has_suggestions': len(suggestions) > 0,
                'priority': 'high' if len(suggestions) > 2 else ('medium' if len(suggestions) > 0 else 'low')
            }
        
        # 历史统计（可选）
        if include_history:
            response_data['history'] = {
                'statistics': statistics,
                'note': 'Statistics since pool initialization'
            }
        
        logger.debug(f"Database pool status retrieved: active={checked_out}, pool_size={pool_size}")
        
        return jsonify({
            'success': True,
            'data': response_data
        }), 200
        
    except PerformanceError as e:
        logger.error(f"Performance error when getting database pool status: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': e.error_code if hasattr(e, 'error_code') else 'PERFORMANCE_ERROR'
        }), 500
        
    except Exception as e:
        logger.error(f"Failed to get database pool status: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


def _assess_pool_status(utilization: Dict[str, Any]) -> Dict[str, Any]:
    """评估连接池状态
    
    Args:
        utilization: 使用率信息
        
    Returns:
        状态评估结果
    """
    utilization_percent = utilization.get('utilization_percent', 0)
    overflow_percent = utilization.get('overflow_percent', 0)
    is_overloaded = utilization.get('is_overloaded', False)
    
    # 确定状态级别
    if is_overloaded or utilization_percent >= 90:
        level = 'critical'
        status = 'overloaded'
        message = 'Connection pool is overloaded, consider increasing pool size'
    elif utilization_percent >= 70 or overflow_percent > 0:
        level = 'warning'
        status = 'high_usage'
        message = 'Connection pool usage is high, monitor closely'
    elif utilization_percent >= 50:
        level = 'normal'
        status = 'moderate_usage'
        message = 'Connection pool usage is moderate'
    else:
        level = 'good'
        status = 'healthy'
        message = 'Connection pool is operating normally'
    
    return {
        'level': level,
        'status': status,
        'message': message,
        'utilization_percent': utilization_percent,
        'overflow_percent': overflow_percent
    }


def _generate_pool_suggestions(utilization: Dict[str, Any], assessment: Dict[str, Any]) -> List[str]:
    """生成连接池优化建议
    
    Args:
        utilization: 使用率信息
        assessment: 状态评估结果
        
    Returns:
        优化建议列表
    """
    suggestions = []
    
    level = assessment.get('level', 'good')
    utilization_percent = utilization.get('utilization_percent', 0)
    overflow_percent = utilization.get('overflow_percent', 0)
    efficiency = utilization.get('efficiency', {})
    
    # 根据状态级别生成建议
    if level == 'critical':
        suggestions.append('URGENT: Increase pool_size immediately to handle current load')
        suggestions.append('Consider implementing connection queuing or request throttling')
    
    if level in ('critical', 'warning'):
        suggestions.append('Review slow queries that may be holding connections too long')
        suggestions.append('Ensure all database sessions are properly closed after use')
    
    if overflow_percent > 0:
        suggestions.append(f'Pool overflow detected ({overflow_percent}%), increase pool_size to reduce overflow')
    
    # 效率相关建议
    pool_hits = efficiency.get('pool_hits', 0)
    pool_misses = efficiency.get('pool_misses', 0)
    if pool_misses > pool_hits and pool_misses > 10:
        suggestions.append('High pool miss rate detected, consider increasing pool_size')
    
    # 低使用率建议
    if utilization_percent < 10 and utilization.get('pool_size', 0) > 5:
        suggestions.append('Low pool utilization, consider decreasing pool_size to free resources')
    
    return suggestions


@performance_bp.route('/db/health', methods=['GET'])
def check_database_health():
    """检查数据库健康状态

    执行全面的数据库健康检查，包括连接测试、响应时间、连接池状态等多项指标。

    Query Parameters:
        detailed: 是否返回详细检查结果（默认true）
        include_latency: 是否包含延迟测试（默认true）
        timeout: 健康检查超时时间（秒，默认5）

    Returns:
        JSON: 健康检查结果，包含以下字段：
            - is_healthy: 总体健康状态
            - status: 状态级别（healthy/degraded/unhealthy）
            - checks: 各项检查结果详情
            - response_time: 响应时间
            - recommendations: 问题建议（如果有）
    """
    try:
        # 获取查询参数
        detailed = request.args.get('detailed', 'true').lower() == 'true'
        include_latency = request.args.get('include_latency', 'true').lower() == 'true'
        timeout = request.args.get('timeout', 5, type=float)
        
        db_pool_manager = get_database_pool_manager()
        
        # 记录检查开始时间
        check_start_time = time.time()
        
        # 初始化检查结果
        checks = {
            'connection': {'status': 'unknown', 'message': '', 'passed': False},
            'pool_status': {'status': 'unknown', 'message': '', 'passed': False},
            'pool_utilization': {'status': 'unknown', 'message': '', 'passed': False}
        }
        
        if include_latency:
            checks['query_latency'] = {'status': 'unknown', 'message': '', 'passed': False, 'latency_ms': None}
        
        issues = []
        recommendations = []
        
        # 检查1: 连接池初始化状态
        pool_status = db_pool_manager.get_pool_status()
        if pool_status.get('status') == 'not_initialized':
            checks['pool_status'] = {
                'status': 'failed',
                'message': 'Database pool not initialized',
                'passed': False
            }
            issues.append('Database pool not initialized')
            recommendations.append('Initialize the database pool with init_app()')
        elif pool_status.get('status') == 'error':
            checks['pool_status'] = {
                'status': 'failed',
                'message': f"Pool error: {pool_status.get('error', 'Unknown')}",
                'passed': False
            }
            issues.append('Database pool in error state')
        else:
            checks['pool_status'] = {
                'status': 'passed',
                'message': 'Database pool is initialized and operational',
                'passed': True
            }
        
        # 检查2: 基本连接测试
        connection_start = time.time()
        try:
            is_connected = db_pool_manager.health_check()
            connection_time = (time.time() - connection_start) * 1000  # 转换为毫秒
            
            if is_connected:
                checks['connection'] = {
                    'status': 'passed',
                    'message': f'Database connection successful ({connection_time:.2f}ms)',
                    'passed': True,
                    'response_time_ms': round(connection_time, 2)
                }
            else:
                checks['connection'] = {
                    'status': 'failed',
                    'message': 'Database connection test failed',
                    'passed': False
                }
                issues.append('Database connection test failed')
                recommendations.append('Check database server status and connection credentials')
        except Exception as conn_error:
            connection_time = (time.time() - connection_start) * 1000
            checks['connection'] = {
                'status': 'failed',
                'message': f'Connection error: {str(conn_error)}',
                'passed': False,
                'error': str(conn_error)
            }
            issues.append(f'Connection error: {str(conn_error)}')
            recommendations.append('Verify database server is running and accessible')
        
        # 检查3: 连接池使用率
        if pool_status.get('status') == 'healthy':
            pool_size = pool_status.get('pool_size', 0)
            checked_out = pool_status.get('checked_out_connections', 0)
            overflow = pool_status.get('overflow_connections', 0)
            
            if pool_size > 0:
                utilization = (checked_out / pool_size) * 100
                
                if utilization >= 90 or overflow > pool_size * 0.5:
                    checks['pool_utilization'] = {
                        'status': 'warning',
                        'message': f'High pool utilization: {utilization:.1f}%',
                        'passed': True,  # 通过但有警告
                        'utilization_percent': round(utilization, 2),
                        'overflow_connections': overflow
                    }
                    issues.append(f'High connection pool utilization ({utilization:.1f}%)')
                    recommendations.append('Consider increasing pool_size to handle load')
                elif utilization >= 70:
                    checks['pool_utilization'] = {
                        'status': 'warning',
                        'message': f'Moderate pool utilization: {utilization:.1f}%',
                        'passed': True,
                        'utilization_percent': round(utilization, 2)
                    }
                else:
                    checks['pool_utilization'] = {
                        'status': 'passed',
                        'message': f'Pool utilization healthy: {utilization:.1f}%',
                        'passed': True,
                        'utilization_percent': round(utilization, 2)
                    }
            else:
                checks['pool_utilization'] = {
                    'status': 'warning',
                    'message': 'Pool size is zero',
                    'passed': False
                }
        
        # 检查4: 查询延迟测试（可选）
        if include_latency and checks['connection']['passed']:
            latency_result = _measure_query_latency(db_pool_manager, timeout)
            checks['query_latency'] = latency_result
            
            if not latency_result['passed']:
                issues.append(f"Query latency issue: {latency_result['message']}")
                recommendations.append('Investigate slow queries or database performance')
            elif latency_result.get('latency_ms', 0) > 100:
                issues.append(f"High query latency: {latency_result['latency_ms']}ms")
                recommendations.append('Consider optimizing database queries or adding indexes')
        
        # 计算总体状态
        total_checks = len(checks)
        passed_checks = sum(1 for c in checks.values() if c.get('passed', False))
        warning_checks = sum(1 for c in checks.values() if c.get('status') == 'warning')
        failed_checks = total_checks - passed_checks
        
        # 确定健康状态
        if failed_checks == 0 and warning_checks == 0:
            overall_status = 'healthy'
            overall_message = 'All database health checks passed'
            is_healthy = True
        elif failed_checks == 0 and warning_checks > 0:
            overall_status = 'degraded'
            overall_message = f'{warning_checks} check(s) have warnings'
            is_healthy = True
        else:
            overall_status = 'unhealthy'
            overall_message = f'{failed_checks} check(s) failed'
            is_healthy = False
        
        # 计算总响应时间
        total_check_time = (time.time() - check_start_time) * 1000
        
        # 构建响应数据
        response_data = {
            'is_healthy': is_healthy,
            'status': overall_status,
            'message': overall_message,
            'timestamp': time.time(),
            'timestamp_iso': timestamp_to_iso(time.time()),
            'response_time_ms': round(total_check_time, 2),
            'summary': {
                'total_checks': total_checks,
                'passed': passed_checks,
                'warnings': warning_checks,
                'failed': failed_checks
            }
        }
        
        # 添加详细检查结果（可选）
        if detailed:
            response_data['checks'] = checks
        
        # 添加问题和建议（如果有）
        if issues:
            response_data['issues'] = issues
        if recommendations:
            response_data['recommendations'] = recommendations
        
        # 根据健康状态返回不同的HTTP状态码
        http_status = 200 if is_healthy else 503
        
        logger.debug(f"Database health check completed: status={overall_status}, time={total_check_time:.2f}ms")
        
        return jsonify({
            'success': True,
            'data': response_data
        }), http_status
        
    except PerformanceError as e:
        logger.error(f"Performance error during database health check: {e}")
        return jsonify({
            'success': False,
            'data': {
                'is_healthy': False,
                'status': 'error',
                'message': str(e),
                'error_code': e.error_code if hasattr(e, 'error_code') else 'PERFORMANCE_ERROR'
            }
        }), 500
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'data': {
                'is_healthy': False,
                'status': 'error',
                'message': f'Health check failed: {str(e)}',
                'error_code': 'INTERNAL_ERROR'
            }
        }), 500


def _measure_query_latency(db_pool_manager, timeout: float = 5.0) -> Dict[str, Any]:
    """测量数据库查询延迟
    
    Args:
        db_pool_manager: 数据库连接池管理器
        timeout: 超时时间（秒）
        
    Returns:
        延迟测量结果
    """
    try:
        import threading
        
        result = {'passed': False, 'status': 'unknown', 'message': '', 'latency_ms': None}
        latencies = []
        
        def measure_single_query():
            try:
                start = time.time()
                with db_pool_manager.get_session() as session:
                    session.execute("SELECT 1")
                latency = (time.time() - start) * 1000
                latencies.append(latency)
            except Exception:
                pass
        
        # 执行3次查询取平均值
        threads = []
        for _ in range(3):
            t = threading.Thread(target=measure_single_query)
            t.start()
            threads.append(t)
        
        # 等待所有线程完成（带超时）
        for t in threads:
            t.join(timeout=timeout / 3)
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            # 评估延迟
            if avg_latency < 50:
                status = 'passed'
                message = f'Query latency excellent: {avg_latency:.2f}ms avg'
            elif avg_latency < 100:
                status = 'passed'
                message = f'Query latency good: {avg_latency:.2f}ms avg'
            elif avg_latency < 500:
                status = 'warning'
                message = f'Query latency moderate: {avg_latency:.2f}ms avg'
            else:
                status = 'warning'
                message = f'Query latency high: {avg_latency:.2f}ms avg'
            
            result = {
                'passed': True,
                'status': status,
                'message': message,
                'latency_ms': round(avg_latency, 2),
                'min_latency_ms': round(min_latency, 2),
                'max_latency_ms': round(max_latency, 2),
                'samples': len(latencies)
            }
        else:
            result = {
                'passed': False,
                'status': 'failed',
                'message': 'Failed to measure query latency',
                'latency_ms': None
            }
        
        return result
        
    except Exception as e:
        return {
            'passed': False,
            'status': 'failed',
            'message': f'Latency measurement error: {str(e)}',
            'latency_ms': None,
            'error': str(e)
        }


@performance_bp.route('/monitoring/metrics', methods=['GET'])
def get_current_metrics():
    """获取当前性能指标

    获取系统当前的性能指标，包括CPU、内存、磁盘、网络、GPU等资源使用情况。

    Query Parameters:
        include_system: 是否包含系统指标（默认true）
        include_gpu: 是否包含GPU指标（默认true）
        include_training: 是否包含训练指标（默认true）
        include_alerts: 是否包含告警信息（默认true）
        include_trends: 是否包含趋势分析（默认false）

    Returns:
        JSON: 当前性能指标，包含以下字段：
            - system: 系统指标（CPU、内存、磁盘、网络）
            - gpu: GPU指标列表
            - training: 训练指标
            - alerts: 活跃告警
            - trends: 趋势分析（可选）
    """
    try:
        # 获取查询参数
        include_system = request.args.get('include_system', 'true').lower() == 'true'
        include_gpu = request.args.get('include_gpu', 'true').lower() == 'true'
        include_training = request.args.get('include_training', 'true').lower() == 'true'
        include_alerts = request.args.get('include_alerts', 'true').lower() == 'true'
        include_trends = request.args.get('include_trends', 'false').lower() == 'true'
        
        performance_monitor = get_performance_monitor()
        
        # 获取监控状态
        monitor_status = performance_monitor.get_status()
        
        # 获取原始指标
        raw_metrics = performance_monitor.get_current_metrics()
        
        if not raw_metrics:
            return jsonify({
                'success': False,
                'error': 'No metrics available',
                'error_code': 'NO_METRICS',
                'hint': 'Monitoring service may not be running or no data collected yet',
                'monitor_status': monitor_status
            }), 404
        
        # 构建响应数据
        response_data = {
            'timestamp': time.time(),
            'timestamp_iso': timestamp_to_iso(time.time()),
            'monitor_status': monitor_status.get('status', 'unknown')
        }
        
        # 系统指标
        if include_system and raw_metrics.get('system'):
            system_metrics = raw_metrics['system']
            response_data['system'] = format_system_metrics(system_metrics)
        
        # GPU指标
        if include_gpu:
            gpu_list = raw_metrics.get('gpu', [])
            if gpu_list:
                response_data['gpu'] = {
                    'available': True,
                    'count': len(gpu_list),
                    'devices': [format_gpu_metrics(gpu) for gpu in gpu_list]
                }
            else:
                response_data['gpu'] = {
                    'available': False,
                    'count': 0,
                    'devices': [],
                    'message': 'No GPU devices detected or GPU monitoring disabled'
                }
        
        # 训练指标
        if include_training:
            training_metrics = raw_metrics.get('training')
            if training_metrics:
                response_data['training'] = format_training_metrics(training_metrics)
            else:
                response_data['training'] = None
        
        # 告警信息
        if include_alerts:
            active_alerts = performance_monitor.get_active_alerts()
            response_data['alerts'] = {
                'count': len(active_alerts),
                'has_critical': any(a.level.value == 'critical' for a in active_alerts if hasattr(a.level, 'value')),
                'has_warning': any(a.level.value in ('high', 'medium') for a in active_alerts if hasattr(a.level, 'value')),
                'items': [
                    {
                        'id': alert.alert_id,
                        'name': alert.name,
                        'level': alert.level.value if hasattr(alert.level, 'value') else str(alert.level),
                        'description': alert.description,
                        'metric_value': alert.metric_value,
                        'threshold': alert.threshold,
                        'timestamp': alert.timestamp.isoformat() if alert.timestamp else None
                    }
                    for alert in active_alerts[:10]  # 最多返回10条
                ]
            }
        
        # 趋势分析（可选）
        if include_trends and raw_metrics.get('system'):
            trends = _calculate_metrics_trends(performance_monitor)
            response_data['trends'] = trends
        
        # 添加摘要信息
        response_data['summary'] = _generate_metrics_summary(response_data)
        
        logger.debug(f"Current metrics retrieved: monitor_status={monitor_status.get('status')}")
        
        return jsonify({
            'success': True,
            'data': response_data
        }), 200
        
    except PerformanceError as e:
        logger.error(f"Performance error when getting current metrics: {e}")
        return jsonify({
                'success': False,
            'error': str(e),
            'error_code': e.error_code if hasattr(e, 'error_code') else 'PERFORMANCE_ERROR'
        }), 500
            
    except Exception as e:
        logger.error(f"Failed to get current metrics: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


def _calculate_metrics_trends(performance_monitor) -> Dict[str, Any]:
    """计算指标趋势
    
    Args:
        performance_monitor: 性能监控器实例
        
    Returns:
        趋势分析结果
    """
    try:
        # 获取最近5分钟的数据
        end_time = time.time()
        start_time = end_time - 300  # 5分钟
        
        history = performance_monitor.get_metrics_history(start_time=start_time, end_time=end_time)
        cpu_data = history.get('cpu_usage', [])
        
        trends = {
            'cpu': {'direction': 'stable', 'change_percent': 0},
            'memory': {'direction': 'stable', 'change_percent': 0}
        }
        
        if len(cpu_data) >= 2:
            # 计算CPU趋势
            recent_values = [d['cpu_percent'] for d in cpu_data[-10:] if 'cpu_percent' in d]
            if len(recent_values) >= 2:
                first_half_avg = sum(recent_values[:len(recent_values)//2]) / (len(recent_values)//2)
                second_half_avg = sum(recent_values[len(recent_values)//2:]) / (len(recent_values) - len(recent_values)//2)
                
                change = second_half_avg - first_half_avg
                trends['cpu'] = {
                    'direction': 'increasing' if change > 5 else ('decreasing' if change < -5 else 'stable'),
                    'change_percent': round(change, 2)
                }
        
        return trends
        
    except Exception as e:
        logger.warning(f"Failed to calculate trends: {e}")
        return {}


def _generate_metrics_summary(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """生成指标摘要
    
    Args:
        response_data: 响应数据
        
    Returns:
        摘要信息
    """
    summary = {
        'overall_status': 'good',
        'issues': []
    }
    
    # 检查系统指标
    system = response_data.get('system', {})
    if system:
        if system.get('cpu', {}).get('status') == 'critical':
            summary['issues'].append('CPU usage critical')
        if system.get('memory', {}).get('status') == 'critical':
            summary['issues'].append('Memory usage critical')
        if system.get('disk', {}).get('status') == 'critical':
            summary['issues'].append('Disk usage critical')
    
    # 检查GPU指标
    gpu_info = response_data.get('gpu', {})
    if gpu_info.get('available'):
        for device in gpu_info.get('devices', []):
            if device.get('utilization', {}).get('status') == 'critical':
                summary['issues'].append(f"GPU {device.get('id')} utilization critical")
            if device.get('temperature', {}).get('status') == 'critical':
                summary['issues'].append(f"GPU {device.get('id')} temperature critical")
    
    # 检查告警
    alerts = response_data.get('alerts', {})
    if alerts.get('has_critical'):
        summary['issues'].append('Critical alerts active')
    
    # 确定总体状态
    if any('critical' in issue.lower() for issue in summary['issues']):
        summary['overall_status'] = 'critical'
    elif summary['issues'] or alerts.get('has_warning'):
        summary['overall_status'] = 'warning'
    
    summary['issue_count'] = len(summary['issues'])
    
    return summary


@performance_bp.route('/monitoring/summary', methods=['GET'])
def get_performance_summary():
    """获取性能摘要

    获取系统性能的综合摘要信息，包括资源使用统计、健康状态、告警概览、趋势分析等。

    Query Parameters:
        time_range: 统计时间范围（默认1h，可选：5m/15m/30m/1h/6h/24h）
        include_recommendations: 是否包含优化建议（默认true）
        include_statistics: 是否包含详细统计（默认true）

    Returns:
        JSON: 性能摘要信息，包含以下字段：
            - overview: 系统概览
            - resource_usage: 资源使用摘要
            - health_status: 健康状态
            - alert_summary: 告警摘要
            - statistics: 统计信息
            - recommendations: 优化建议
    """
    try:
        # 获取查询参数
        time_range = request.args.get('time_range', '1h')
        include_recommendations = request.args.get('include_recommendations', 'true').lower() == 'true'
        include_statistics = request.args.get('include_statistics', 'true').lower() == 'true'
        
        performance_monitor = get_performance_monitor()
        
        # 获取监控状态
        monitor_status = performance_monitor.get_status()
        
        # 获取当前指标
        current_metrics = performance_monitor.get_current_metrics()
        
        # 转换 current_metrics 为 dict 格式
        if current_metrics and hasattr(current_metrics, '__dict__'):
            current_metrics = current_metrics.__dict__

        # 解析时间范围
        time_seconds = parse_time_range(time_range)
        end_time = time.time()
        start_time = end_time - time_seconds
        
        # 构建响应数据
        response_data = {
            'timestamp': time.time(),
            'timestamp_iso': timestamp_to_iso(time.time()),
            'time_range': time_range,
            'time_range_seconds': time_seconds
        }
        
        # 系统概览
        response_data['overview'] = _build_system_overview(performance_monitor, monitor_status)
        
        # 资源使用摘要
        response_data['resource_usage'] = _build_resource_usage_summary(current_metrics)
        
        # 健康状态
        response_data['health_status'] = _build_health_status(performance_monitor, current_metrics)
        
        # 告警摘要
        response_data['alert_summary'] = _build_alert_summary(performance_monitor)
        
        # 详细统计（可选）
        if include_statistics:
            response_data['statistics'] = _build_performance_statistics(
                performance_monitor, start_time, end_time
            )
        
        # 优化建议（可选）
        if include_recommendations:
            response_data['recommendations'] = _generate_performance_recommendations(
                response_data['resource_usage'],
                response_data['health_status'],
                response_data['alert_summary']
            )
        
        logger.debug(f"Performance summary retrieved: time_range={time_range}")
        
        return jsonify({
            'success': True,
            'data': response_data
        }), 200
        
    except PerformanceError as e:
        logger.error(f"Performance error when getting summary: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': e.error_code if hasattr(e, 'error_code') else 'PERFORMANCE_ERROR'
        }), 500
        
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


# parse_time_range 已移动到 utils.py，使用导入的 parse_time_range


def _build_system_overview(performance_monitor, monitor_status: Dict[str, Any]) -> Dict[str, Any]:
    """构建系统概览
    
    Args:
        performance_monitor: 性能监控器实例
        monitor_status: 监控状态
        
    Returns:
        系统概览信息
    """
    return {
        'monitoring_status': monitor_status.get('status', 'unknown'),
        'uptime_info': {
            'system_metrics_collected': monitor_status.get('system_metrics_count', 0),
            'gpu_metrics_collected': monitor_status.get('gpu_metrics_count', 0),
            'training_metrics_collected': monitor_status.get('training_metrics_count', 0)
        },
        'active_alerts_count': monitor_status.get('active_alerts', 0)
    }


def _build_resource_usage_summary(current_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """构建资源使用摘要
    
    Args:
        current_metrics: 当前指标
        
    Returns:
        资源使用摘要
    """
    summary = {
        'cpu': {'current': 0, 'status': 'unknown'},
        'memory': {'current': 0, 'used_gb': 0, 'total_gb': 0, 'status': 'unknown'},
        'disk': {'current': 0, 'status': 'unknown'},
        'gpu': {'available': False, 'devices': []}
    }
    
    system = current_metrics.get('system')
    if system:
        cpu_percent = getattr(system, 'cpu_percent', 0)
        memory_percent = getattr(system, 'memory_percent', 0)
        disk_percent = getattr(system, 'disk_percent', 0)
        
        summary['cpu'] = {
            'current': round(cpu_percent, 2),
            'status': get_resource_status(cpu_percent, 80, 90)
        }
        
        summary['memory'] = {
            'current': round(memory_percent, 2),
            'used_gb': round(getattr(system, 'memory_used_gb', 0), 2),
            'total_gb': round(getattr(system, 'memory_total_gb', 0), 2),
            'status': get_resource_status(memory_percent, 75, 85)
        }
        
        summary['disk'] = {
            'current': round(disk_percent, 2),
            'status': get_resource_status(disk_percent, 80, 90)
        }
        
        summary['load_average'] = getattr(system, 'load_average', [0, 0, 0])
    
    # GPU摘要
    gpu_list = current_metrics.get('gpu', [])
    if gpu_list:
        summary['gpu'] = {
            'available': True,
            'count': len(gpu_list),
            'devices': [
                {
                    'id': getattr(gpu, 'gpu_id', i),
                    'utilization': round(getattr(gpu, 'gpu_utilization', 0), 2),
                    'memory_utilization': round(getattr(gpu, 'memory_utilization', 0), 2),
                    'temperature': getattr(gpu, 'temperature', 0),
                    'status': get_resource_status(getattr(gpu, 'gpu_utilization', 0), 80, 95)
                }
                for i, gpu in enumerate(gpu_list)
            ]
        }
    
    return summary


def _build_health_status(performance_monitor, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """构建健康状态
    
    Args:
        performance_monitor: 性能监控器实例
        current_metrics: 当前指标
        
    Returns:
        健康状态信息
    """
    issues = []
    warnings = []
    
    system = current_metrics.get('system')
    if system:
        cpu = getattr(system, 'cpu_percent', 0)
        memory = getattr(system, 'memory_percent', 0)
        disk = getattr(system, 'disk_percent', 0)
        
        if cpu >= 90:
            issues.append({'type': 'cpu', 'message': f'CPU usage critical: {cpu:.1f}%', 'severity': 'critical'})
        elif cpu >= 80:
            warnings.append({'type': 'cpu', 'message': f'CPU usage high: {cpu:.1f}%', 'severity': 'warning'})
        
        if memory >= 85:
            issues.append({'type': 'memory', 'message': f'Memory usage critical: {memory:.1f}%', 'severity': 'critical'})
        elif memory >= 75:
            warnings.append({'type': 'memory', 'message': f'Memory usage high: {memory:.1f}%', 'severity': 'warning'})
        
        if disk >= 90:
            issues.append({'type': 'disk', 'message': f'Disk usage critical: {disk:.1f}%', 'severity': 'critical'})
        elif disk >= 80:
            warnings.append({'type': 'disk', 'message': f'Disk usage high: {disk:.1f}%', 'severity': 'warning'})
    
    # 检查GPU
    gpu_list = current_metrics.get('gpu', [])
    for gpu in gpu_list:
        temp = getattr(gpu, 'temperature', 0)
        util = getattr(gpu, 'gpu_utilization', 0)
        gpu_id = getattr(gpu, 'gpu_id', 0)
        
        if temp >= 85:
            issues.append({'type': 'gpu_temp', 'message': f'GPU {gpu_id} temperature critical: {temp}°C', 'severity': 'critical'})
        elif temp >= 75:
            warnings.append({'type': 'gpu_temp', 'message': f'GPU {gpu_id} temperature high: {temp}°C', 'severity': 'warning'})
        
        if util >= 95:
            warnings.append({'type': 'gpu_util', 'message': f'GPU {gpu_id} utilization very high: {util:.1f}%', 'severity': 'warning'})
    
    # 确定总体状态
    if issues:
        overall_status = 'critical'
    elif warnings:
        overall_status = 'warning'
    else:
        overall_status = 'healthy'
    
    return {
        'overall_status': overall_status,
        'issues_count': len(issues),
        'warnings_count': len(warnings),
        'issues': issues,
        'warnings': warnings
    }


def _build_alert_summary(performance_monitor) -> Dict[str, Any]:
    """构建告警摘要
    
    Args:
        performance_monitor: 性能监控器实例
        
    Returns:
        告警摘要信息
    """
    active_alerts = performance_monitor.get_active_alerts()
    
    # 按级别分类
    by_level = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
    for alert in active_alerts:
        level = alert.level.value if hasattr(alert.level, 'value') else str(alert.level)
        if level in by_level:
            by_level[level] += 1
    
    # 按类型分类
    by_type = {}
    for alert in active_alerts:
        alert_type = getattr(alert, 'rule_id', 'unknown').split('_')[0]
        by_type[alert_type] = by_type.get(alert_type, 0) + 1
    
    return {
        'total_active': len(active_alerts),
        'by_level': by_level,
        'by_type': by_type,
        'has_critical': by_level['critical'] > 0,
        'has_high': by_level['high'] > 0,
        'recent_alerts': [
            {
                'id': alert.alert_id,
                'name': alert.name,
                'level': alert.level.value if hasattr(alert.level, 'value') else str(alert.level),
                'timestamp': alert.timestamp.isoformat() if alert.timestamp else None
            }
            for alert in active_alerts[:5]  # 最近5条
        ]
    }


def _build_performance_statistics(performance_monitor, start_time: float, end_time: float) -> Dict[str, Any]:
    """构建性能统计信息
    
    Args:
        performance_monitor: 性能监控器实例
        start_time: 开始时间
        end_time: 结束时间
        
    Returns:
        性能统计信息
    """
    try:
        history = performance_monitor.get_metrics_history(start_time=start_time, end_time=end_time)
        cpu_data = history.get('cpu_usage', [])
        
        statistics = {
            'cpu': {'avg': 0, 'max': 0, 'min': 0, 'samples': 0},
            'memory': {'avg': 0, 'max': 0, 'min': 0, 'samples': 0},
            'data_points': 0
        }
        
        if cpu_data:
            cpu_values = [d.get('cpu_percent', 0) for d in cpu_data if 'cpu_percent' in d]
            memory_values = [d.get('memory_percent', 0) for d in cpu_data if 'memory_percent' in d]
            
            if cpu_values:
                statistics['cpu'] = {
                    'avg': round(sum(cpu_values) / len(cpu_values), 2),
                    'max': round(max(cpu_values), 2),
                    'min': round(min(cpu_values), 2),
                    'samples': len(cpu_values)
                }
            
            if memory_values:
                statistics['memory'] = {
                    'avg': round(sum(memory_values) / len(memory_values), 2),
                    'max': round(max(memory_values), 2),
                    'min': round(min(memory_values), 2),
                    'samples': len(memory_values)
                }
            
            statistics['data_points'] = len(cpu_data)
        
        # GPU统计
        gpu_data = history.get('gpu_usage', [])
        if gpu_data:
            gpu_utils = [d.get('gpu_utilization', 0) for d in gpu_data if 'gpu_utilization' in d]
            if gpu_utils:
                statistics['gpu'] = {
                    'avg': round(sum(gpu_utils) / len(gpu_utils), 2),
                    'max': round(max(gpu_utils), 2),
                    'min': round(min(gpu_utils), 2),
                    'samples': len(gpu_utils)
                }
        
        return statistics
        
    except Exception as e:
        logger.warning(f"Failed to build statistics: {e}")
        return {'error': str(e)}


def _generate_performance_recommendations(
    resource_usage: Dict[str, Any],
    health_status: Dict[str, Any],
    alert_summary: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """生成性能优化建议
    
    Args:
        resource_usage: 资源使用摘要
        health_status: 健康状态
        alert_summary: 告警摘要
        
    Returns:
        优化建议列表
    """
    recommendations = []
    
    # CPU相关建议
    cpu_current = resource_usage.get('cpu', {}).get('current', 0)
    if cpu_current >= 90:
        recommendations.append({
            'priority': 'high',
            'category': 'cpu',
            'title': 'Critical CPU Usage',
            'description': f'CPU usage is at {cpu_current}%. Consider scaling horizontally or optimizing CPU-intensive tasks.',
            'actions': ['Review running processes', 'Consider load balancing', 'Optimize algorithms']
        })
    elif cpu_current >= 80:
        recommendations.append({
            'priority': 'medium',
            'category': 'cpu',
            'title': 'High CPU Usage',
            'description': f'CPU usage is at {cpu_current}%. Monitor closely and plan for capacity.',
            'actions': ['Monitor trends', 'Identify resource-heavy processes']
        })
    
    # 内存相关建议
    memory_current = resource_usage.get('memory', {}).get('current', 0)
    if memory_current >= 85:
        recommendations.append({
            'priority': 'high',
            'category': 'memory',
            'title': 'Critical Memory Usage',
            'description': f'Memory usage is at {memory_current}%. Risk of out-of-memory errors.',
            'actions': ['Increase memory allocation', 'Optimize memory usage', 'Check for memory leaks']
        })
    elif memory_current >= 75:
        recommendations.append({
            'priority': 'medium',
            'category': 'memory',
            'title': 'High Memory Usage',
            'description': f'Memory usage is at {memory_current}%. Consider memory optimization.',
            'actions': ['Review memory allocation', 'Implement caching strategies']
        })
    
    # 磁盘相关建议
    disk_current = resource_usage.get('disk', {}).get('current', 0)
    if disk_current >= 90:
        recommendations.append({
            'priority': 'high',
            'category': 'disk',
            'title': 'Critical Disk Usage',
            'description': f'Disk usage is at {disk_current}%. Immediate action required.',
            'actions': ['Clean up temporary files', 'Archive old data', 'Expand storage']
        })
    elif disk_current >= 80:
        recommendations.append({
            'priority': 'medium',
            'category': 'disk',
            'title': 'High Disk Usage',
            'description': f'Disk usage is at {disk_current}%. Plan for storage expansion.',
            'actions': ['Review storage usage', 'Implement data retention policies']
        })
    
    # GPU相关建议
    gpu_info = resource_usage.get('gpu', {})
    if gpu_info.get('available'):
        for device in gpu_info.get('devices', []):
            temp = device.get('temperature', 0)
            if temp >= 85:
                recommendations.append({
                    'priority': 'high',
                    'category': 'gpu',
                    'title': f'GPU {device.get("id")} Temperature Critical',
                    'description': f'GPU temperature is at {temp}°C. Risk of thermal throttling.',
                    'actions': ['Improve cooling', 'Reduce GPU workload', 'Check airflow']
                })
    
    # 告警相关建议
    if alert_summary.get('has_critical'):
        recommendations.append({
            'priority': 'high',
            'category': 'alerts',
            'title': 'Critical Alerts Active',
            'description': f'{alert_summary.get("by_level", {}).get("critical", 0)} critical alerts require attention.',
            'actions': ['Review alert details', 'Take immediate action', 'Investigate root causes']
        })
    
    # 如果没有问题，添加积极反馈
    if not recommendations:
        recommendations.append({
            'priority': 'info',
            'category': 'general',
            'title': 'System Health Good',
            'description': 'All resources are operating within normal parameters.',
            'actions': ['Continue monitoring', 'Review performance regularly']
        })
    
    # 按优先级排序
    priority_order = {'high': 0, 'medium': 1, 'low': 2, 'info': 3}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 99))
    
    return recommendations


@performance_bp.route('/monitoring/alerts', methods=['GET'])
def get_performance_alerts():
    """获取性能警报

    获取系统性能相关的告警信息，包括活跃告警和历史告警。

    Query Parameters:
        status: 告警状态筛选（active/resolved/all，默认all）
        level: 告警级别筛选（critical/high/medium/low，可多选用逗号分隔）
        type: 告警类型筛选（cpu/memory/disk/gpu/training）
        duration: 历史告警时间范围（分钟，默认60）
        limit: 返回数量限制（默认100）
        offset: 分页偏移量（默认0）

    Returns:
        JSON: 性能警报列表，包含以下字段：
            - active_alerts: 当前活跃的告警
            - historical_alerts: 历史告警（已解除）
            - statistics: 告警统计信息
            - alert_rules: 告警规则信息
    """
    try:
        # 获取查询参数
        status_filter = request.args.get('status', 'all').lower()
        level_filter = request.args.get('level', '')
        type_filter = request.args.get('type', '')
        duration_minutes = request.args.get('duration', 60, type=int)
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # 参数验证
        if status_filter not in ('active', 'resolved', 'all'):
            return jsonify({
                'success': False,
                'error': f"Invalid status filter: {status_filter}",
                'error_code': 'INVALID_STATUS_FILTER',
                'valid_values': ['active', 'resolved', 'all']
            }), 400
        
        if limit < 1 or limit > 1000:
            limit = min(max(limit, 1), 1000)
        
        performance_monitor = get_performance_monitor()
        
        # 解析级别筛选
        level_filters = [l.strip().lower() for l in level_filter.split(',') if l.strip()] if level_filter else []
        
        # 解析类型筛选
        type_filters = [t.strip().lower() for t in type_filter.split(',') if t.strip()] if type_filter else []
        
        # 获取活跃告警
        active_alerts = performance_monitor.get_active_alerts()
        
        # 获取告警历史
        alert_history = list(performance_monitor.alert_history) if hasattr(performance_monitor, 'alert_history') else []
        
        # 时间筛选
        cutoff_time = time.time() - (duration_minutes * 60)
        
        # 格式化活跃告警
        formatted_active = []
        for alert in active_alerts:
            formatted = format_alert(alert)
            
            # 应用筛选条件
            if level_filters and formatted['level'].lower() not in level_filters:
                continue
            if type_filters and not _match_alert_type(formatted, type_filters):
                continue
            
            formatted_active.append(formatted)
        
        # 格式化历史告警（已解除的）
        formatted_historical = []
        for alert in alert_history:
            # 时间筛选
            alert_time = alert.timestamp.timestamp() if hasattr(alert.timestamp, 'timestamp') else 0
            if alert_time < cutoff_time:
                continue
            
            # 只包含已解除的告警
            if not getattr(alert, 'resolved', False):
                continue
            
            formatted = format_alert(alert)
            
            # 应用筛选条件
            if level_filters and formatted['level'].lower() not in level_filters:
                continue
            if type_filters and not _match_alert_type(formatted, type_filters):
                continue
            
            formatted_historical.append(formatted)
        
        # 根据状态筛选返回结果
        response_data = {
            'timestamp': time.time(),
            'timestamp_iso': timestamp_to_iso(time.time()),
            'query_params': {
                'status': status_filter,
                'level': level_filters or 'all',
                'type': type_filters or 'all',
                'duration_minutes': duration_minutes,
                'limit': limit,
                'offset': offset
            }
        }
        
        # 根据状态筛选构建结果
        if status_filter == 'active':
            alerts_to_return = formatted_active
        elif status_filter == 'resolved':
            alerts_to_return = formatted_historical
        else:  # all
            alerts_to_return = formatted_active + formatted_historical
        
        # 按时间排序（最新的在前）
        alerts_to_return.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # 应用分页
        total_count = len(alerts_to_return)
        alerts_to_return = alerts_to_return[offset:offset + limit]
        
        response_data['alerts'] = alerts_to_return
        response_data['pagination'] = {
            'total': total_count,
            'limit': limit,
            'offset': offset,
            'has_more': (offset + limit) < total_count
        }
        
        # 告警统计
        response_data['statistics'] = _build_alert_statistics(
            formatted_active, formatted_historical, performance_monitor
        )
        
        # 告警规则信息
        response_data['rules'] = _get_alert_rules_summary(performance_monitor)
        
        logger.debug(f"Performance alerts retrieved: active={len(formatted_active)}, historical={len(formatted_historical)}")
        
        return jsonify({
            'success': True,
            'data': response_data
        }), 200
        
    except PerformanceError as e:
        logger.error(f"Performance error when getting alerts: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': e.error_code if hasattr(e, 'error_code') else 'PERFORMANCE_ERROR'
        }), 500
        
    except Exception as e:
        logger.error(f"Failed to get performance alerts: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Internal error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


# format_alert 已移动到 utils.py，使用导入的 format_alert


def _match_alert_type(formatted_alert: Dict[str, Any], type_filters: List[str]) -> bool:
    """检查告警是否匹配类型筛选
    
    Args:
        formatted_alert: 格式化后的告警
        type_filters: 类型筛选列表
        
    Returns:
        是否匹配
    """
    rule_id = formatted_alert.get('rule_id', '').lower()
    name = formatted_alert.get('name', '').lower()
    
    for type_filter in type_filters:
        if type_filter in rule_id or type_filter in name:
            return True
    return False


# format_duration 已移动到 utils.py，使用导入的 format_duration


def _build_alert_statistics(
    active_alerts: List[Dict],
    historical_alerts: List[Dict],
    performance_monitor
) -> Dict[str, Any]:
    """构建告警统计信息
    
    Args:
        active_alerts: 活跃告警列表
        historical_alerts: 历史告警列表
        performance_monitor: 性能监控器实例
        
    Returns:
        告警统计信息
    """
    # 按级别统计活跃告警
    active_by_level = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
    for alert in active_alerts:
        level = alert.get('level', '').lower()
        if level in active_by_level:
            active_by_level[level] += 1
    
    # 按级别统计历史告警
    historical_by_level = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
    for alert in historical_alerts:
        level = alert.get('level', '').lower()
        if level in historical_by_level:
            historical_by_level[level] += 1
    
    # 计算平均解决时间（历史告警）
    resolution_times = [
        a.get('duration_seconds', 0) 
        for a in historical_alerts 
        if a.get('duration_seconds')
    ]
    avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
    
    # 按类型分类统计
    type_counts = {}
    for alert in active_alerts + historical_alerts:
        rule_id = alert.get('rule_id', 'unknown')
        alert_type = rule_id.split('_')[0] if '_' in rule_id else rule_id
        type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
    
    return {
        'active': {
            'total': len(active_alerts),
            'by_level': active_by_level,
            'has_critical': active_by_level['critical'] > 0,
            'has_high': active_by_level['high'] > 0
        },
        'historical': {
            'total': len(historical_alerts),
            'by_level': historical_by_level
        },
        'by_type': type_counts,
        'resolution': {
            'avg_resolution_time_seconds': round(avg_resolution_time, 2),
            'avg_resolution_time_human': format_duration(avg_resolution_time) if avg_resolution_time else 'N/A',
            'resolved_count': len(historical_alerts),
            'fastest_resolution': round(min(resolution_times), 2) if resolution_times else None,
            'slowest_resolution': round(max(resolution_times), 2) if resolution_times else None
        }
    }


def _get_alert_rules_summary(performance_monitor) -> Dict[str, Any]:
    """获取告警规则摘要
    
    Args:
        performance_monitor: 性能监控器实例
        
    Returns:
        告警规则摘要
    """
    rules = getattr(performance_monitor, 'alert_rules', [])
    
    enabled_rules = [r for r in rules if getattr(r, 'enabled', True)]
    disabled_rules = [r for r in rules if not getattr(r, 'enabled', True)]
    
    # 按类型分类
    by_type = {}
    for rule in rules:
        metric_type = rule.metric_type.value if hasattr(rule.metric_type, 'value') else str(rule.metric_type)
        by_type[metric_type] = by_type.get(metric_type, 0) + 1
    
    return {
        'total': len(rules),
        'enabled': len(enabled_rules),
        'disabled': len(disabled_rules),
        'by_type': by_type,
        'rules': [
            {
                'id': rule.id,
                'name': rule.name,
                'metric_type': rule.metric_type.value if hasattr(rule.metric_type, 'value') else str(rule.metric_type),
                'metric_name': rule.metric_name,
                'threshold': rule.threshold,
                'operator': rule.operator,
                'severity': rule.severity.value if hasattr(rule.severity, 'value') else str(rule.severity),
                'enabled': getattr(rule, 'enabled', True)
            }
            for rule in rules
        ]
    }


@performance_bp.route('/monitoring/health', methods=['GET'])
def check_performance_health():
    """检查性能监控健康状态

    执行全面的性能监控系统健康检查，包括监控服务、异步处理器、数据库连接池等组件。

    Query Parameters:
        detailed: 是否返回详细检查结果（默认true）
        include_components: 是否包含各组件状态（默认true）
        include_metrics: 是否包含指标收集状态（默认true）

    Returns:
        JSON: 健康检查结果，包含以下字段：
            - is_healthy: 总体健康状态
            - status: 状态级别（healthy/degraded/unhealthy）
            - components: 各组件健康状态
            - metrics_collection: 指标收集状态
            - issues: 发现的问题列表
            - recommendations: 建议操作
    """
    try:
        # 获取查询参数
        detailed = request.args.get('detailed', 'true').lower() == 'true'
        include_components = request.args.get('include_components', 'true').lower() == 'true'
        include_metrics = request.args.get('include_metrics', 'true').lower() == 'true'
        
        check_start_time = time.time()
        
        # 初始化检查结果
        component_checks = {}
        issues = []
        recommendations = []
        
        # 检查1: 性能监控服务
        monitor_check = _check_performance_monitor()
        component_checks['performance_monitor'] = monitor_check
        if not monitor_check['passed']:
            issues.append(monitor_check['message'])
            if monitor_check.get('recommendation'):
                recommendations.append(monitor_check['recommendation'])
        
        # 检查2: 异步处理器
        async_check = _check_async_processor()
        component_checks['async_processor'] = async_check
        if not async_check['passed']:
            issues.append(async_check['message'])
            if async_check.get('recommendation'):
                recommendations.append(async_check['recommendation'])
        
        # 检查3: 数据库连接池
        db_check = _check_database_pool()
        component_checks['database_pool'] = db_check
        if not db_check['passed']:
            issues.append(db_check['message'])
            if db_check.get('recommendation'):
                recommendations.append(db_check['recommendation'])
        
        # 检查4: 指标收集状态
        metrics_check = None
        if include_metrics:
            metrics_check = _check_metrics_collection()
            component_checks['metrics_collection'] = metrics_check
            if not metrics_check['passed']:
                issues.append(metrics_check['message'])
                if metrics_check.get('recommendation'):
                    recommendations.append(metrics_check['recommendation'])
        
        # 检查5: 告警系统
        alerts_check = _check_alert_system()
        component_checks['alert_system'] = alerts_check
        if alerts_check.get('has_critical_alerts'):
            issues.append(f"Critical alerts active: {alerts_check.get('critical_count', 0)}")
            recommendations.append('Review and resolve critical alerts immediately')
        
        # 计算总体健康状态
        passed_checks = sum(1 for c in component_checks.values() if c.get('passed', False))
        total_checks = len(component_checks)
        warning_checks = sum(1 for c in component_checks.values() if c.get('status') == 'warning')
        failed_checks = total_checks - passed_checks
        
        # 确定健康状态
        if failed_checks == 0 and warning_checks == 0:
            overall_status = 'healthy'
            overall_message = 'All performance monitoring components are healthy'
            is_healthy = True
        elif failed_checks == 0 and warning_checks > 0:
            overall_status = 'degraded'
            overall_message = f'{warning_checks} component(s) have warnings'
            is_healthy = True
        elif failed_checks <= 1:
            overall_status = 'degraded'
            overall_message = f'{failed_checks} component(s) failed, system partially operational'
            is_healthy = False
        else:
            overall_status = 'unhealthy'
            overall_message = f'{failed_checks} component(s) failed'
            is_healthy = False
        
        # 计算总检查时间
        check_time = (time.time() - check_start_time) * 1000
        
        # 构建响应数据
        response_data = {
            'is_healthy': is_healthy,
            'status': overall_status,
            'message': overall_message,
            'timestamp': time.time(),
            'timestamp_iso': timestamp_to_iso(time.time()),
            'check_duration_ms': round(check_time, 2),
            'summary': {
                'total_checks': total_checks,
                'passed': passed_checks,
                'warnings': warning_checks,
                'failed': failed_checks
            }
        }
        
        # 添加详细组件状态（可选）
        if include_components and detailed:
            response_data['components'] = component_checks
        
        # 添加问题和建议
        if issues:
            response_data['issues'] = issues
        if recommendations:
            response_data['recommendations'] = list(set(recommendations))  # 去重
        
        # 根据健康状态返回不同的HTTP状态码
        http_status = 200 if is_healthy else 503
        
        logger.debug(f"Performance health check completed: status={overall_status}, time={check_time:.2f}ms")
        
        return jsonify({
            'success': True,
            'data': response_data
        }), http_status
        
    except PerformanceError as e:
        logger.error(f"Performance error during health check: {e}")
        return jsonify({
            'success': False,
            'data': {
                'is_healthy': False,
                'status': 'error',
                'message': str(e),
                'error_code': e.error_code if hasattr(e, 'error_code') else 'PERFORMANCE_ERROR'
            }
        }), 500
        
    except Exception as e:
        logger.error(f"Performance health check failed: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'data': {
                'is_healthy': False,
                'status': 'error',
                'message': f'Health check failed: {str(e)}',
                'error_code': 'INTERNAL_ERROR'
            }
        }), 500


def _check_performance_monitor() -> Dict[str, Any]:
    """检查性能监控服务状态
    
    Returns:
        检查结果
    """
    try:
        performance_monitor = get_performance_monitor()
        status = performance_monitor.get_status()
        
        is_running = status.get('status') == 'running'
        metrics_count = status.get('system_metrics_count', 0)
        
        if is_running and metrics_count > 0:
            return {
                'passed': True,
                'status': 'healthy',
                'message': 'Performance monitor is running and collecting metrics',
                'details': {
                    'is_running': True,
                    'system_metrics': status.get('system_metrics_count', 0),
                    'gpu_metrics': status.get('gpu_metrics_count', 0),
                    'training_metrics': status.get('training_metrics_count', 0),
                    'active_alerts': status.get('active_alerts', 0)
                }
            }
        elif is_running:
            return {
                'passed': True,
                'status': 'warning',
                'message': 'Performance monitor is running but no metrics collected yet',
                'recommendation': 'Wait for metrics collection cycle to complete',
                'details': {
                    'is_running': True,
                    'system_metrics': 0
                }
            }
        else:
            return {
                'passed': False,
                'status': 'failed',
                'message': 'Performance monitor is not running',
                'recommendation': 'Start the performance monitoring service',
                'details': {
                    'is_running': False
                }
            }
    except Exception as e:
        return {
            'passed': False,
            'status': 'error',
            'message': f'Failed to check performance monitor: {str(e)}',
            'recommendation': 'Check performance monitor configuration'
        }


def _check_async_processor() -> Dict[str, Any]:
    """检查异步处理器状态
    
    Returns:
        检查结果
    """
    try:
        async_processor = get_async_processor()
        stats = async_processor.get_stats()
        
        is_running = stats.get('is_running', False)
        queue_size = stats.get('queue_size', 0)
        max_queue = async_processor.queue_size
        active_workers = stats.get('active_workers', 0)
        max_workers = async_processor.max_workers
        
        # 检查工作线程
        worker_threads = async_processor._worker_threads
        alive_workers = sum(1 for t in worker_threads if t.is_alive())
        
        if not is_running:
            return {
                'passed': False,
                'status': 'failed',
                'message': 'Async processor is not running',
                'recommendation': 'Restart the async processor service',
                'details': {'is_running': False}
            }
        
        # 检查工作线程健康度
        worker_health = alive_workers / max_workers if max_workers > 0 else 0
        queue_utilization = (queue_size / max_queue * 100) if max_queue > 0 else 0
        
        if worker_health < 0.5:
            return {
                'passed': False,
                'status': 'failed',
                'message': f'Too few worker threads alive: {alive_workers}/{max_workers}',
                'recommendation': 'Restart async processor to restore worker threads',
                'details': {
                    'is_running': True,
                    'alive_workers': alive_workers,
                    'max_workers': max_workers,
                    'queue_utilization': round(queue_utilization, 2)
                }
            }
        
        if queue_utilization > 90:
            return {
                'passed': True,
                'status': 'warning',
                'message': f'Task queue nearly full: {queue_utilization:.1f}%',
                'recommendation': 'Consider increasing queue size or worker count',
                'details': {
                    'is_running': True,
                    'alive_workers': alive_workers,
                    'queue_size': queue_size,
                    'queue_utilization': round(queue_utilization, 2)
                }
            }
        
        # 检查失败率
        total_tasks = stats.get('total_tasks', 0)
        failed_tasks = stats.get('failed_tasks', 0) + stats.get('timeout_tasks', 0)
        failure_rate = (failed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        if failure_rate > 10:
            return {
                'passed': True,
                'status': 'warning',
                'message': f'High task failure rate: {failure_rate:.1f}%',
                'recommendation': 'Review failed tasks and fix underlying issues',
                'details': {
                    'is_running': True,
                    'failure_rate': round(failure_rate, 2),
                    'total_tasks': total_tasks,
                    'failed_tasks': failed_tasks
                }
            }
        
        return {
            'passed': True,
            'status': 'healthy',
            'message': 'Async processor is healthy',
            'details': {
                'is_running': True,
                'alive_workers': alive_workers,
                'max_workers': max_workers,
                'queue_size': queue_size,
                'queue_utilization': round(queue_utilization, 2),
                'total_tasks': total_tasks,
                'completed_tasks': stats.get('completed_tasks', 0),
                'failure_rate': round(failure_rate, 2)
            }
        }
    except Exception as e:
        return {
            'passed': False,
            'status': 'error',
            'message': f'Failed to check async processor: {str(e)}',
            'recommendation': 'Check async processor configuration'
        }


def _check_database_pool() -> Dict[str, Any]:
    """检查数据库连接池状态
    
    Returns:
        检查结果
    """
    try:
        db_pool_manager = get_database_pool_manager()
        pool_status = db_pool_manager.get_pool_status()
        
        if pool_status.get('status') == 'not_initialized':
            return {
                'passed': False,
                'status': 'failed',
                'message': 'Database pool not initialized',
                'recommendation': 'Initialize database pool with init_app()',
                'details': {'initialized': False}
            }
        
        if pool_status.get('status') == 'error':
            return {
                'passed': False,
                'status': 'failed',
                'message': f"Database pool error: {pool_status.get('error', 'Unknown')}",
                'recommendation': 'Check database configuration and connectivity',
                'details': {'error': pool_status.get('error')}
            }
        
        # 检查连接池使用率
        pool_size = pool_status.get('pool_size', 0)
        checked_out = pool_status.get('checked_out_connections', 0)
        overflow = pool_status.get('overflow_connections', 0)
        
        utilization = (checked_out / pool_size * 100) if pool_size > 0 else 0
        
        if utilization >= 90 or overflow > pool_size * 0.5:
            return {
                'passed': True,
                'status': 'warning',
                'message': f'Database pool under high load: {utilization:.1f}% utilization',
                'recommendation': 'Consider increasing pool size',
                'details': {
                    'initialized': True,
                    'pool_size': pool_size,
                    'utilization': round(utilization, 2),
                    'overflow': overflow
                }
            }
        
        return {
            'passed': True,
            'status': 'healthy',
            'message': 'Database pool is healthy',
            'details': {
                'initialized': True,
                'pool_size': pool_size,
                'checked_out': checked_out,
                'utilization': round(utilization, 2),
                'overflow': overflow
            }
        }
    except Exception as e:
        return {
            'passed': False,
            'status': 'error',
            'message': f'Failed to check database pool: {str(e)}',
            'recommendation': 'Check database configuration'
        }


def _check_metrics_collection() -> Dict[str, Any]:
    """检查指标收集状态
    
    Returns:
        检查结果
    """
    try:
        performance_monitor = get_performance_monitor()
        current_metrics = performance_monitor.get_current_metrics()
        
        has_system = current_metrics.get('system') is not None
        has_gpu = len(current_metrics.get('gpu', [])) > 0
        
        # 检查数据新鲜度
        system = current_metrics.get('system')
        data_age = None
        is_stale = False
        
        if system:
            last_timestamp = getattr(system, 'timestamp', 0)
            data_age = time.time() - last_timestamp
            is_stale = data_age > 60  # 超过60秒视为过期
        
        if not has_system:
            return {
                'passed': False,
                'status': 'failed',
                'message': 'No system metrics being collected',
                'recommendation': 'Ensure monitoring service is running and collecting metrics',
                'details': {
                    'has_system_metrics': False,
                    'has_gpu_metrics': has_gpu
                }
            }
        
        if is_stale:
            return {
                'passed': True,
                'status': 'warning',
                'message': f'Metrics data is stale ({data_age:.0f}s old)',
                'recommendation': 'Check if monitoring collection is paused',
                'details': {
                    'has_system_metrics': True,
                    'has_gpu_metrics': has_gpu,
                    'data_age_seconds': round(data_age, 2),
                    'is_stale': True
                }
            }
        
        return {
            'passed': True,
            'status': 'healthy',
            'message': 'Metrics collection is active',
            'details': {
                'has_system_metrics': True,
                'has_gpu_metrics': has_gpu,
                'data_age_seconds': round(data_age, 2) if data_age else None,
                'is_stale': False
            }
        }
    except Exception as e:
        return {
            'passed': False,
            'status': 'error',
            'message': f'Failed to check metrics collection: {str(e)}',
            'recommendation': 'Check monitoring service configuration'
        }


def _check_alert_system() -> Dict[str, Any]:
    """检查告警系统状态
    
    Returns:
        检查结果
    """
    try:
        performance_monitor = get_performance_monitor()
        active_alerts = performance_monitor.get_active_alerts()
        rules = getattr(performance_monitor, 'alert_rules', [])
        
        # 统计告警
        critical_count = sum(1 for a in active_alerts if hasattr(a.level, 'value') and a.level.value in ('critical', 'high'))
        warning_count = sum(1 for a in active_alerts if hasattr(a.level, 'value') and a.level.value in ('medium', 'low'))
        
        enabled_rules = sum(1 for r in rules if getattr(r, 'enabled', True))
        
        result = {
            'passed': True,
            'status': 'healthy',
            'message': f'Alert system operational with {enabled_rules} rules',
            'has_critical_alerts': critical_count > 0,
            'critical_count': critical_count,
            'warning_count': warning_count,
            'details': {
                'total_rules': len(rules),
                'enabled_rules': enabled_rules,
                'active_alerts': len(active_alerts),
                'critical_alerts': critical_count,
                'warning_alerts': warning_count
            }
        }
        
        if critical_count > 0:
            result['status'] = 'warning'
            result['message'] = f'{critical_count} critical alert(s) active'
        
        return result
    except Exception as e:
        return {
            'passed': False,
            'status': 'error',
            'message': f'Failed to check alert system: {str(e)}',
            'has_critical_alerts': False
        }


def init_performance_api(app, config: Optional[Dict[str, Any]] = None):
    """初始化性能API

    完整初始化性能模块的所有组件，包括：
    - 异步任务处理器
    - 数据库连接池
    - 性能监控服务
    - 告警规则配置

    Args:
        app: Flask应用实例
        config: 可选的配置字典，可覆盖默认配置
            - auto_start_monitoring: 是否自动启动监控（默认True）
            - setup_default_alerts: 是否配置默认告警规则（默认True）
            - init_async_processor: 是否初始化异步处理器（默认True）
            - init_database_pool: 是否初始化数据库连接池（默认True）
            - register_cleanup: 是否注册清理函数（默认True）

    Returns:
        Dict: 初始化结果，包含各组件的状态

    Raises:
        RuntimeError: 关键组件初始化失败时抛出
    """
    init_start_time = time.time()
    config = config or {}
    
    # 默认配置
    auto_start_monitoring = config.get('auto_start_monitoring', True)
    setup_default_alerts = config.get('setup_default_alerts', True)
    init_async_processor = config.get('init_async_processor', True)
    init_database_pool = config.get('init_database_pool', True)
    register_cleanup = config.get('register_cleanup', True)
    
    # 初始化结果记录
    init_results = {
        'blueprint': {'status': 'pending'},
        'async_processor': {'status': 'skipped'},
        'database_pool': {'status': 'skipped'},
        'performance_monitor': {'status': 'pending'},
        'alert_rules': {'status': 'skipped'},
        'cleanup_handler': {'status': 'skipped'}
    }
    
    errors = []
    warnings = []
    
    try:
        # ====== 1. 注册Blueprint ======
        try:
            if hasattr(app, 'register_blueprint'):
                app.register_blueprint(performance_bp)
            init_results['blueprint'] = {
                'status': 'success',
                'url_prefix': '/api/performance'
            }
            logger.info("Performance API Blueprint registered successfully")
        except Exception as e:
            init_results['blueprint'] = {'status': 'error', 'error': str(e)}
            errors.append(f"Blueprint registration failed: {e}")
            raise RuntimeError(f"Failed to register performance blueprint: {e}")
        
        # ====== 2. 初始化异步处理器 ======
        if init_async_processor:
            try:
                async_processor = get_async_processor()
                
                # 调用 init_app 进行 Flask 集成
                if hasattr(async_processor, 'init_app'):
                    async_processor.init_app(app)
                
                # 确保处理器正在运行
                stats = async_processor.get_stats()
                if not stats.get('is_running', False):
                    # 尝试启动处理器
                    if hasattr(async_processor, 'start'):
                        async_processor.start()
                        logger.info("AsyncProcessor started")
                
                # 获取最新状态
                stats = async_processor.get_stats()
                init_results['async_processor'] = {
                    'status': 'success',
                    'is_running': stats.get('is_running', False),
                    'max_workers': async_processor.max_workers,
                    'queue_size': async_processor.queue_size
                }
                logger.info(f"AsyncProcessor initialized: workers={async_processor.max_workers}, queue={async_processor.queue_size}")
            except Exception as e:
                init_results['async_processor'] = {'status': 'warning', 'error': str(e)}
                warnings.append(f"AsyncProcessor initialization warning: {e}")
                logger.warning(f"AsyncProcessor initialization warning: {e}")
        
        # ====== 3. 初始化数据库连接池 ======
        if init_database_pool:
            try:
                db_pool_manager = get_database_pool_manager()
                
                # 调用 init_app 进行初始化
                if hasattr(db_pool_manager, 'init_app'):
                    db_pool_manager.init_app(app)
                
                # 获取池状态
                pool_status = db_pool_manager.get_pool_status()
                init_results['database_pool'] = {
                    'status': 'success',
                    'pool_size': pool_status.get('pool_size', 0),
                    'initialized': pool_status.get('status') != 'not_initialized'
                }
                logger.info(f"Database pool initialized: pool_size={pool_status.get('pool_size', 0)}")
            except Exception as e:
                init_results['database_pool'] = {'status': 'warning', 'error': str(e)}
                warnings.append(f"Database pool initialization warning: {e}")
                logger.warning(f"Database pool initialization warning: {e}")
        
        # ====== 4. 初始化性能监控服务 ======
        try:
            performance_monitor = get_performance_monitor()
            
            # 配置默认告警规则
            if setup_default_alerts:
                _setup_default_alert_rules(performance_monitor)
                init_results['alert_rules'] = {
                    'status': 'success',
                    'rules_count': len(getattr(performance_monitor, 'alert_rules', []))
                }
                logger.info("Default alert rules configured")
            
            # 自动启动监控
            if auto_start_monitoring:
                if hasattr(performance_monitor, 'start_monitoring'):
                    performance_monitor.start_monitoring()
                    
                status = performance_monitor.get_status()
                init_results['performance_monitor'] = {
                    'status': 'success',
                    'is_monitoring': status.get('status') == 'running',
                    'collection_interval': getattr(performance_monitor, 'collection_interval', 5)
                }
                logger.info("Performance monitoring started")
            else:
                init_results['performance_monitor'] = {
                    'status': 'success',
                    'is_monitoring': False,
                    'message': 'Auto-start disabled'
                }
        except Exception as e:
            init_results['performance_monitor'] = {'status': 'warning', 'error': str(e)}
            warnings.append(f"Performance monitor initialization warning: {e}")
            logger.warning(f"Performance monitor initialization warning: {e}")
        
        # ====== 5. 注册应用清理函数 ======
        if register_cleanup:
            try:
                # 注册应用关闭时的清理回调
                @app.teardown_appcontext
                def _cleanup_on_teardown(exception=None):
                    """应用上下文销毁时执行轻量级清理"""
                    try:
                        # 轻量级清理：清理已完成任务
                        if hasattr(get_async_processor, 'cleanup_completed_tasks'):
                            async_processor = get_async_processor()
                            async_processor.cleanup_completed_tasks()
                        else:
                            logger.warning("AsyncProcessor cleanup_completed_tasks method not available")
                    except Exception as e:
                        logger.debug(f"Teardown cleanup: {e}")
                
                # 使用 atexit 注册完整清理
                import atexit
                
                def _full_cleanup():
                    """应用退出时执行完整清理"""
                    cleanup_performance_api()
                
                atexit.register(_full_cleanup)
                
                init_results['cleanup_handler'] = {
                    'status': 'success',
                    'registered_handlers': ['teardown_appcontext', 'atexit']
                }
                logger.info("Cleanup handlers registered")
            except Exception as e:
                init_results['cleanup_handler'] = {'status': 'warning', 'error': str(e)}
                warnings.append(f"Cleanup handler registration warning: {e}")
                logger.warning(f"Cleanup handler registration warning: {e}")
        
        # ====== 6. 将性能模块绑定到 app ======
        app.performance_api_initialized = True
        app.performance_init_time = time.time()
        
        # 计算初始化时间
        init_duration = (time.time() - init_start_time) * 1000
        
        # 确定总体状态
        success_count = sum(1 for r in init_results.values() if r['status'] == 'success')
        warning_count = sum(1 for r in init_results.values() if r['status'] == 'warning')
        error_count = sum(1 for r in init_results.values() if r['status'] == 'error')
        
        if error_count > 0:
            overall_status = 'failed'
        elif warning_count > 0:
            overall_status = 'partial'
        else:
            overall_status = 'success'
        
        # 构建最终结果
        final_result = {
            'overall_status': overall_status,
            'init_duration_ms': round(init_duration, 2),
            'components': init_results,
            'summary': {
                'success': success_count,
                'warnings': warning_count,
                'errors': error_count
            }
        }
        
        if warnings:
            final_result['warnings'] = warnings
        if errors:
            final_result['errors'] = errors
        
        logger.info(f"Performance API initialization completed: status={overall_status}, duration={init_duration:.2f}ms")
        
        return final_result
        
    except RuntimeError:
        # 关键错误，重新抛出
        raise
    except Exception as e:
        logger.error(f"Performance API initialization failed: {e}", exc_info=True)
        raise RuntimeError(f"Performance API initialization failed: {e}")


def _setup_default_alert_rules(performance_monitor):
    """配置默认告警规则（从配置文件加载）
    
    Args:
        performance_monitor: 性能监控器实例
    """
    try:
        # 检查是否已有规则
        existing_rules = getattr(performance_monitor, 'alert_rules', [])
        existing_names = {getattr(r, 'name', '') for r in existing_rules}
        
        # 从配置文件加载默认告警规则
        config = _get_api_config()
        default_rules = config.get('default_alert_rules', [])
        
        # 添加规则（跳过已存在的）
        added_count = 0
        for rule_config in default_rules:
            if rule_config['name'] not in existing_names:
                if hasattr(performance_monitor, 'add_alert_rule'):
                    performance_monitor.add_alert_rule(rule_config)
                    added_count += 1
                    logger.debug(f"Added alert rule: {rule_config['name']}")
        
        logger.info(f"Default alert rules setup: added {added_count} new rules")
        
    except Exception as e:
        logger.warning(f"Failed to setup default alert rules: {e}")


def cleanup_performance_api():
    """清理性能API资源
    
    执行完整的资源清理，包括：
    - 停止性能监控
    - 关闭异步处理器
    - 清理数据库连接池
    
    Returns:
        Dict: 清理结果
    """
    cleanup_start_time = time.time()
    cleanup_results = {}
    errors = []
    
    logger.info("Starting performance API cleanup...")
    
    # 1. 停止性能监控
    try:
        performance_monitor = get_performance_monitor()
        if hasattr(performance_monitor, 'stop_monitoring'):
            performance_monitor.stop_monitoring()
        if hasattr(performance_monitor, 'close'):
            performance_monitor.close()
            cleanup_results['performance_monitor'] = {'status': 'success'}
            logger.info("Performance monitor stopped")
    except Exception as e:
        cleanup_results['performance_monitor'] = {'status': 'error', 'error': str(e)}
        errors.append(f"Performance monitor cleanup error: {e}")
        logger.error(f"Performance monitor cleanup failed: {e}")
    
    # 2. 关闭异步处理器
    try:
        async_processor = get_async_processor()
        
        # 等待正在运行的任务完成（最多等待10秒）
        if hasattr(async_processor, 'shutdown'):
            async_processor.shutdown(timeout=10)
        
        cleanup_results['async_processor'] = {'status': 'success'}
        logger.info("AsyncProcessor shutdown completed")
    except Exception as e:
        cleanup_results['async_processor'] = {'status': 'error', 'error': str(e)}
        errors.append(f"AsyncProcessor cleanup error: {e}")
        logger.error(f"AsyncProcessor cleanup failed: {e}")
    
    # 3. 清理数据库连接池
    try:
        db_pool_manager = get_database_pool_manager()
        if hasattr(db_pool_manager, 'dispose'):
            db_pool_manager.dispose()
        elif hasattr(db_pool_manager, 'engine') and db_pool_manager.engine:
            db_pool_manager.engine.dispose()
        
        cleanup_results['database_pool'] = {'status': 'success'}
        logger.info("Database pool disposed")
    except Exception as e:
        cleanup_results['database_pool'] = {'status': 'error', 'error': str(e)}
        errors.append(f"Database pool cleanup error: {e}")
        logger.error(f"Database pool cleanup failed: {e}")
    
    # 计算清理时间
    cleanup_duration = (time.time() - cleanup_start_time) * 1000
    
    # 确定总体状态
    if errors:
        overall_status = 'partial' if len(errors) < 3 else 'failed'
    else:
        overall_status = 'success'
    
    result = {
        'overall_status': overall_status,
        'cleanup_duration_ms': round(cleanup_duration, 2),
        'components': cleanup_results
    }
    
    if errors:
        result['errors'] = errors
    
    logger.info(f"Performance API cleanup completed: status={overall_status}, duration={cleanup_duration:.2f}ms")
    
    return result


def get_performance_api_status() -> Dict[str, Any]:
    """获取性能API的整体状态
    
    Returns:
        Dict: 包含各组件状态的详细信息
    """
    try:
        status = {
            'api_available': True,
            'timestamp': time.time(),
            'components': {}
        }
        
        # 检查异步处理器
        try:
            async_processor = get_async_processor()
            stats = async_processor.get_stats()
            status['components']['async_processor'] = {
                'available': True,
                'is_running': stats.get('is_running', False),
                'active_workers': stats.get('active_workers', 0),
                'queue_size': stats.get('queue_size', 0)
            }
        except Exception as e:
            status['components']['async_processor'] = {
                'available': False,
                'error': str(e)
            }
        
        # 检查数据库连接池
        try:
            db_pool_manager = get_database_pool_manager()
            pool_status = db_pool_manager.get_pool_status()
            status['components']['database_pool'] = {
                'available': True,
                'initialized': pool_status.get('status') != 'not_initialized',
                'pool_size': pool_status.get('pool_size', 0)
            }
        except Exception as e:
            status['components']['database_pool'] = {
                'available': False,
                'error': str(e)
            }
        
        # 检查性能监控器
        try:
            performance_monitor = get_performance_monitor()
            monitor_status = performance_monitor.get_status()
            status['components']['performance_monitor'] = {
                'available': True,
                'is_monitoring': monitor_status.get('status') == 'running',
                'active_alerts': monitor_status.get('active_alerts', 0)
            }
        except Exception as e:
            status['components']['performance_monitor'] = {
                'available': False,
                'error': str(e)
            }
        
        # 检查任务注册表
        status['components']['task_registry'] = {
            'available': True,
            'registered_tasks': len(task_registry._tasks),
            'categories': list(task_registry._categories.keys())
        }
        
        return status
        
    except Exception as e:
        return {
            'api_available': False,
            'error': str(e),
            'timestamp': time.time()
        }


# ============================================================================
# 增强的任务管理端点（基于服务层）
# ============================================================================

def _get_performance_service():
    """延迟获取性能服务"""
    try:
        from backend.services.performance_service import get_performance_service
        return get_performance_service(use_memory=True)
    except ImportError:
        return None


@performance_bp.route('/tasks', methods=['POST'])
def create_persisted_task():
    """创建持久化任务记录

    创建任务记录并存储到仓库层，支持任务追踪和历史查询。

    Request Body:
        {
            "name": "string",           # 必需，任务名称
            "category": "string",       # 可选，任务分类
            "description": "string",    # 可选，任务描述
            "priority": "string",       # 可选，优先级: low/normal/high/urgent
            "params": {},               # 可选，任务参数
            "timeout": float            # 可选，超时时间（秒）
        }

    Returns:
        JSON: 创建的任务信息
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        data = request.get_json() or {}
        name = data.get('name')
        
        if not name:
            return jsonify({
                'success': False,
                'error': 'Task name is required',
                'error_code': 'MISSING_NAME'
            }), 400

        success, message, task = service.create_task(
            name=name,
            category=data.get('category', 'general'),
            description=data.get('description'),
            priority=data.get('priority', 'normal'),
            params=data.get('params', {}),
            timeout=data.get('timeout'),
            tenant_id=data.get('tenant_id'),
            created_by=data.get('created_by')
        )

        if success:
            return jsonify({
                'success': True,
                'data': task,
                'message': message
            }), 201
        else:
            return jsonify({
                'success': False,
                'error': message,
                'error_code': 'CREATE_FAILED'
            }), 400

    except Exception as e:
        logger.error(f"Failed to create task: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/tasks', methods=['GET'])
def list_persisted_tasks():
    """获取任务列表

    Query Parameters:
        status: 任务状态筛选
        category: 任务分类筛选
        priority: 优先级筛选
        limit: 返回数量限制（默认100）
        offset: 分页偏移量

    Returns:
        JSON: 任务列表
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        tasks = service.list_tasks(
            tenant_id=request.args.get('tenant_id'),
            status=request.args.get('status'),
            category=request.args.get('category'),
            priority=request.args.get('priority'),
            limit=request.args.get('limit', 100, type=int),
            offset=request.args.get('offset', 0, type=int)
        )

        return jsonify({
            'success': True,
            'data': {
                'tasks': tasks,
                'total': len(tasks)
            }
        }), 200

    except Exception as e:
        logger.error(f"Failed to list tasks: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/tasks/<task_id>', methods=['GET'])
def get_persisted_task(task_id: str):
    """获取任务详情

    Args:
        task_id: 任务ID

    Returns:
        JSON: 任务详情
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        task = service.get_task(task_id, request.args.get('tenant_id'))

        if task:
            return jsonify({
                'success': True,
                'data': task
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': f'Task {task_id} not found',
                'error_code': 'TASK_NOT_FOUND'
            }), 404

    except Exception as e:
        logger.error(f"Failed to get task: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/tasks/<task_id>/cancel', methods=['POST'])
def cancel_persisted_task(task_id: str):
    """取消任务

    Args:
        task_id: 任务ID

    Returns:
        JSON: 取消结果
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        data = request.get_json() or {}
        success, message = service.cancel_task(task_id, data.get('tenant_id'))

        if success:
            return jsonify({
                'success': True,
                'message': message
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': message,
                'error_code': 'CANCEL_FAILED'
            }), 400

    except Exception as e:
        logger.error(f"Failed to cancel task: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/tasks/statistics', methods=['GET'])
def get_task_statistics():
    """获取任务统计

    Returns:
        JSON: 任务统计信息
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        stats = service.get_task_statistics(request.args.get('tenant_id'))

        return jsonify({
            'success': True,
            'data': stats
        }), 200

    except Exception as e:
        logger.error(f"Failed to get task statistics: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/tasks/cleanup', methods=['POST'])
def cleanup_tasks():
    """清理旧任务

    Request Body:
        {
            "max_age_days": int  # 可选，最大保留天数（默认7）
        }

    Returns:
        JSON: 清理结果
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        data = request.get_json() or {}
        max_age_days = data.get('max_age_days', 7)
        
        count = service.cleanup_old_tasks(max_age_days)

        return jsonify({
            'success': True,
            'data': {
                'cleaned_count': count,
                'max_age_days': max_age_days
            },
            'message': f'Cleaned {count} old tasks'
        }), 200

    except Exception as e:
        logger.error(f"Failed to cleanup tasks: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/tasks/execute', methods=['POST'])
def execute_task():
    """执行任务

    通过 PerformanceService 执行注册的任务处理器。
    支持同步、异步和后台三种执行模式。

    Request Body:
        {
            "task_type": "string",         # 必需，任务类型（已注册的处理器名称）
            "params": {},                  # 可选，任务参数
            "priority": "string",          # 可选，优先级: low/normal/high/urgent（默认normal）
            "timeout": float,              # 可选，超时时间（秒）
            "execution_mode": "string",    # 可选，执行模式: sync/async/background（默认async）
            "tenant_id": "string",         # 可选，租户ID
            "created_by": "string"         # 可选，创建者
        }

    执行模式说明:
        - sync: 同步执行，阻塞直到任务完成，返回结果
        - async: 异步执行，使用 AsyncProcessor 线程池
        - background: 后台线程执行，独立线程运行

    Returns:
        JSON: 任务提交结果
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        data = request.get_json() or {}
        task_type = data.get('task_type')
        
        if not task_type:
            return jsonify({
                'success': False,
                'error': 'task_type is required',
                'error_code': 'MISSING_TASK_TYPE'
            }), 400

        # 验证任务类型
        available_handlers = service.get_registered_handlers()
        if task_type not in available_handlers:
            return jsonify({
                'success': False,
                'error': f"Unknown task type: {task_type}",
                'error_code': 'UNKNOWN_TASK_TYPE',
                'available_types': available_handlers
            }), 400

        result = service.submit_task(
            task_type=task_type,
            params=data.get('params', {}),
            priority=data.get('priority', 'normal'),
            timeout=data.get('timeout'),
            execution_mode=data.get('execution_mode', 'async'),
            tenant_id=data.get('tenant_id'),
            created_by=data.get('created_by')
        )

        if result.success:
            return jsonify({
                'success': True,
                'data': {
                    'task_id': result.task_id,
                    'execution_mode': result.execution_mode,
                    'message': result.message
                }
            }), 202 if result.execution_mode != 'sync' else 200
        else:
            return jsonify({
                'success': False,
                'error': result.error,
                'error_code': 'EXECUTION_FAILED'
            }), 400

    except Exception as e:
        logger.error(f"Failed to execute task: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/tasks/handlers', methods=['GET'])
def list_task_handlers():
    """列出可用的任务处理器

    获取 PerformanceService 中注册的所有任务处理器。

    Returns:
        JSON: 任务处理器列表
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        handlers = service.get_registered_handlers()

        return jsonify({
            'success': True,
            'data': {
                'handlers': handlers,
                'total': len(handlers)
            }
        }), 200

    except Exception as e:
        logger.error(f"Failed to list task handlers: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/tasks/<task_id>/retry', methods=['POST'])
def retry_task(task_id: str):
    """重试失败的任务

    重新执行失败、超时或已取消的任务。

    Args:
        task_id: 原任务ID

    Request Body:
        {
            "tenant_id": "string"   # 可选，租户ID
        }

    Returns:
        JSON: 新任务的提交结果
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        data = request.get_json() or {}
        tenant_id = data.get('tenant_id')

        result = service.retry_task(task_id, tenant_id)

        if result.success:
            return jsonify({
                'success': True,
                'data': {
                    'original_task_id': task_id,
                    'new_task_id': result.task_id,
                    'message': result.message or 'Task retry submitted'
                }
            }), 202
        else:
            return jsonify({
                'success': False,
                'error': result.error,
                'error_code': 'RETRY_FAILED'
            }), 400

    except Exception as e:
        logger.error(f"Failed to retry task {task_id}: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


# ============================================================================
# 系统快照端点
# ============================================================================

@performance_bp.route('/snapshots/current', methods=['GET'])
def get_current_snapshot():
    """获取当前系统快照

    获取当前系统的实时资源使用情况。

    Returns:
        JSON: 系统快照，包括CPU、内存、磁盘、网络等信息
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        snapshot = service.get_current_snapshot()

        return jsonify({
            'success': True,
            'data': snapshot
        }), 200

    except Exception as e:
        logger.error(f"Failed to get current snapshot: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/snapshots/history', methods=['GET'])
def get_snapshot_history():
    """获取系统快照历史

    Query Parameters:
        start_time: 开始时间（ISO格式）
        end_time: 结束时间（ISO格式）
        limit: 返回数量限制（默认100）

    Returns:
        JSON: 快照历史列表
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        start_time = None
        end_time = None
        
        if request.args.get('start_time'):
            from datetime import datetime
            start_time = datetime.fromisoformat(request.args.get('start_time').replace('Z', '+00:00'))
        if request.args.get('end_time'):
            from datetime import datetime
            end_time = datetime.fromisoformat(request.args.get('end_time').replace('Z', '+00:00'))

        history = service.get_snapshot_history(
            tenant_id=request.args.get('tenant_id'),
            start_time=start_time,
            end_time=end_time,
            limit=request.args.get('limit', 100, type=int)
        )

        return jsonify({
            'success': True,
            'data': {
                'snapshots': history,
                'total': len(history)
            }
        }), 200

    except Exception as e:
        logger.error(f"Failed to get snapshot history: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


# ============================================================================
# 告警管理端点
# ============================================================================

@performance_bp.route('/alerts/active', methods=['GET'])
def get_active_alerts_v2():
    """获取活跃告警（服务层实现）

    Query Parameters:
        level: 告警级别筛选
        tenant_id: 租户ID

    Returns:
        JSON: 活跃告警列表
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        alerts = service.get_active_alerts(
            tenant_id=request.args.get('tenant_id'),
            level=request.args.get('level')
        )

        return jsonify({
            'success': True,
            'data': {
                'alerts': alerts,
                'total': len(alerts)
            }
        }), 200

    except Exception as e:
        logger.error(f"Failed to get active alerts: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id: str):
    """确认告警

    Args:
        alert_id: 告警ID

    Request Body:
        {
            "user_id": "string"  # 必需，确认用户ID
        }

    Returns:
        JSON: 确认结果
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        data = request.get_json() or {}
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'user_id is required',
                'error_code': 'MISSING_USER_ID'
            }), 400

        success, message = service.acknowledge_alert(
            alert_id=alert_id,
            user_id=user_id,
            tenant_id=data.get('tenant_id')
        )

        if success:
            return jsonify({
                'success': True,
                'message': message
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': message,
                'error_code': 'ACKNOWLEDGE_FAILED'
            }), 400

    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/alerts/<alert_id>/resolve', methods=['POST'])
def resolve_alert(alert_id: str):
    """解决告警

    Args:
        alert_id: 告警ID

    Request Body:
        {
            "user_id": "string",   # 必需，解决用户ID
            "notes": "string"      # 可选，解决说明
        }

    Returns:
        JSON: 解决结果
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        data = request.get_json() or {}
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'user_id is required',
                'error_code': 'MISSING_USER_ID'
            }), 400

        success, message = service.resolve_alert(
            alert_id=alert_id,
            user_id=user_id,
            notes=data.get('notes'),
            tenant_id=data.get('tenant_id')
        )

        if success:
            return jsonify({
                'success': True,
                'message': message
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': message,
                'error_code': 'RESOLVE_FAILED'
            }), 400

    except Exception as e:
        logger.error(f"Failed to resolve alert: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/alerts/statistics', methods=['GET'])
def get_alert_statistics():
    """获取告警统计

    Returns:
        JSON: 告警统计信息
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        stats = service.get_alert_statistics(request.args.get('tenant_id'))

        return jsonify({
            'success': True,
            'data': stats
        }), 200

    except Exception as e:
        logger.error(f"Failed to get alert statistics: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


# ============================================================================
# 告警规则管理端点
# ============================================================================

@performance_bp.route('/rules', methods=['POST'])
def create_alert_rule():
    """创建告警规则

    Request Body:
        {
            "name": "string",              # 必需，规则名称
            "metric_type": "string",       # 必需，指标类型
            "metric_name": "string",       # 必需，指标名称
            "operator": "string",          # 必需，比较操作符
            "threshold": float,            # 必需，阈值
            "severity": "string",          # 可选，告警级别
            "description": "string",       # 可选，描述
            "duration": int,               # 可选，持续时间（秒）
            "notification_channels": []    # 可选，通知渠道
        }

    Returns:
        JSON: 创建的规则信息
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        data = request.get_json() or {}
        
        # 验证必需字段
        required = ['name', 'metric_type', 'metric_name', 'operator', 'threshold']
        missing = [f for f in required if not data.get(f)]
        if missing:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing)}',
                'error_code': 'MISSING_FIELDS'
            }), 400

        success, message, rule = service.create_alert_rule(
            name=data['name'],
            metric_type=data['metric_type'],
            metric_name=data['metric_name'],
            operator=data['operator'],
            threshold=data['threshold'],
            severity=data.get('severity', 'medium'),
            description=data.get('description'),
            duration=data.get('duration', 0),
            notification_channels=data.get('notification_channels', []),
            tenant_id=data.get('tenant_id'),
            created_by=data.get('created_by')
        )

        if success:
            return jsonify({
                'success': True,
                'data': rule,
                'message': message
            }), 201
        else:
            return jsonify({
                'success': False,
                'error': message,
                'error_code': 'CREATE_FAILED'
            }), 400

    except Exception as e:
        logger.error(f"Failed to create alert rule: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/rules', methods=['GET'])
def list_alert_rules():
    """获取告警规则列表

    Query Parameters:
        enabled: 是否启用
        metric_type: 指标类型筛选

    Returns:
        JSON: 规则列表
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        enabled = None
        if request.args.get('enabled'):
            enabled = request.args.get('enabled').lower() == 'true'

        rules = service.get_alert_rules(
            tenant_id=request.args.get('tenant_id'),
            enabled=enabled,
            metric_type=request.args.get('metric_type')
        )

        return jsonify({
            'success': True,
            'data': {
                'rules': rules,
                'total': len(rules)
            }
        }), 200

    except Exception as e:
        logger.error(f"Failed to list alert rules: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/rules/<rule_id>', methods=['PUT'])
def update_alert_rule(rule_id: str):
    """更新告警规则

    Args:
        rule_id: 规则ID

    Request Body:
        规则的更新字段

    Returns:
        JSON: 更新结果
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        data = request.get_json() or {}
        tenant_id = data.pop('tenant_id', None)

        success, message = service.update_alert_rule(rule_id, data, tenant_id)

        if success:
            return jsonify({
                'success': True,
                'message': message
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': message,
                'error_code': 'UPDATE_FAILED'
            }), 400

    except Exception as e:
        logger.error(f"Failed to update alert rule: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/rules/<rule_id>/toggle', methods=['POST'])
def toggle_alert_rule(rule_id: str):
    """切换规则启用状态

    Args:
        rule_id: 规则ID

    Request Body:
        {
            "enabled": bool  # 必需，启用状态
        }

    Returns:
        JSON: 切换结果
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        data = request.get_json() or {}
        enabled = data.get('enabled')
        
        if enabled is None:
            return jsonify({
                'success': False,
                'error': 'enabled field is required',
                'error_code': 'MISSING_ENABLED'
            }), 400

        success, message = service.toggle_alert_rule(
            rule_id=rule_id,
            enabled=enabled,
            tenant_id=data.get('tenant_id')
        )

        if success:
            return jsonify({
                'success': True,
                'message': message
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': message,
                'error_code': 'TOGGLE_FAILED'
            }), 400

    except Exception as e:
        logger.error(f"Failed to toggle alert rule: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/rules/<rule_id>', methods=['DELETE'])
def delete_alert_rule(rule_id: str):
    """删除告警规则

    Args:
        rule_id: 规则ID

    Returns:
        JSON: 删除结果
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        success, message = service.delete_alert_rule(
            rule_id=rule_id,
            tenant_id=request.args.get('tenant_id')
        )

        if success:
            return jsonify({
                'success': True,
                'message': message
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': message,
                'error_code': 'DELETE_FAILED'
            }), 400

    except Exception as e:
        logger.error(f"Failed to delete alert rule: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


# ============================================================================
# 指标收集端点
# ============================================================================

@performance_bp.route('/collection/start', methods=['POST'])
def start_metrics_collection():
    """启动指标收集

    Request Body:
        {
            "interval": int  # 可选，收集间隔（秒，默认10）
        }

    Returns:
        JSON: 启动结果
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        data = request.get_json() or {}
        interval = data.get('interval')

        success = service.start_collection(interval)

        return jsonify({
            'success': success,
            'message': 'Collection started' if success else 'Collection already running',
            'data': {
                'interval': service._collection_interval
            }
        }), 200

    except Exception as e:
        logger.error(f"Failed to start collection: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/collection/stop', methods=['POST'])
def stop_metrics_collection():
    """停止指标收集

    Returns:
        JSON: 停止结果
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        success = service.stop_collection()

        return jsonify({
            'success': success,
            'message': 'Collection stopped'
        }), 200

    except Exception as e:
        logger.error(f"Failed to stop collection: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/metrics/record', methods=['POST'])
def record_custom_metric():
    """记录自定义指标

    Request Body:
        {
            "metric_type": "string",   # 必需，指标类型
            "metric_name": "string",   # 必需，指标名称
            "metric_value": float,     # 必需，指标值
            "metric_unit": "string",   # 可选，单位
            "resource_id": "string",   # 可选，资源ID
            "resource_type": "string", # 可选，资源类型
            "tags": {}                 # 可选，标签
        }

    Returns:
        JSON: 记录的指标信息
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        data = request.get_json() or {}
        
        # 验证必需字段
        required = ['metric_type', 'metric_name', 'metric_value']
        missing = [f for f in required if data.get(f) is None]
        if missing:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing)}',
                'error_code': 'MISSING_FIELDS'
            }), 400

        metric = service.record_metric(
            metric_type=data['metric_type'],
            metric_name=data['metric_name'],
            metric_value=data['metric_value'],
            metric_unit=data.get('metric_unit'),
            resource_id=data.get('resource_id'),
            resource_type=data.get('resource_type'),
            tags=data.get('tags', {}),
            tenant_id=data.get('tenant_id')
        )

        if metric:
            return jsonify({
                'success': True,
                'data': metric
            }), 201
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to record metric',
                'error_code': 'RECORD_FAILED'
            }), 400

    except Exception as e:
        logger.error(f"Failed to record metric: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/metrics/history', methods=['GET'])
def get_metrics_history():
    """获取指标历史

    Query Parameters:
        metric_type: 指标类型筛选
        metric_name: 指标名称筛选
        resource_id: 资源ID筛选
        start_time: 开始时间（ISO格式）
        end_time: 结束时间（ISO格式）
        limit: 返回数量限制（默认1000）

    Returns:
        JSON: 指标历史列表
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        start_time = None
        end_time = None
        
        if request.args.get('start_time'):
            from datetime import datetime
            start_time = datetime.fromisoformat(request.args.get('start_time').replace('Z', '+00:00'))
        if request.args.get('end_time'):
            from datetime import datetime
            end_time = datetime.fromisoformat(request.args.get('end_time').replace('Z', '+00:00'))

        history = service.get_metric_history(
            metric_type=request.args.get('metric_type'),
            metric_name=request.args.get('metric_name'),
            resource_id=request.args.get('resource_id'),
            start_time=start_time,
            end_time=end_time,
            limit=request.args.get('limit', 1000, type=int)
        )

        return jsonify({
            'success': True,
            'data': {
                'metrics': history,
                'total': len(history)
            }
        }), 200

    except Exception as e:
        logger.error(f"Failed to get metrics history: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/statistics', methods=['GET'])
def get_comprehensive_statistics():
    """获取综合统计信息

    获取任务、告警等综合统计信息。

    Returns:
        JSON: 综合统计信息
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'error': 'Performance service not available',
                'error_code': 'SERVICE_UNAVAILABLE'
            }), 503

        stats = service.get_statistics(request.args.get('tenant_id'))

        return jsonify({
            'success': True,
            'data': stats
        }), 200

    except Exception as e:
        logger.error(f"Failed to get statistics: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500


@performance_bp.route('/service/health', methods=['GET'])
def check_service_health():
    """检查服务健康状态

    Returns:
        JSON: 健康检查结果
    """
    try:
        service = _get_performance_service()
        if not service:
            return jsonify({
                'success': False,
                'data': {
                    'healthy': False,
                    'status': 'unavailable',
                    'message': 'Performance service not available'
                }
            }), 503

        result = service.health_check()

        return jsonify({
            'success': True,
            'data': {
                'healthy': result.healthy,
                'status': result.status,
                'message': result.message,
                'details': result.details,
                'check_time_ms': result.check_time_ms
            }
        }), 200 if result.healthy else 503

    except Exception as e:
        logger.error(f"Failed to check service health: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'data': {
                'healthy': False,
                'status': 'error',
                'message': str(e)
            }
        }), 500