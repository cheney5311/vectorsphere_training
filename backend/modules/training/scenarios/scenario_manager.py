"""训练场景管理器

生产级训练任务调度和场景管理功能，支持：
- 任务优先级队列调度
- 并发任务控制
- 进度追踪和状态管理
- 策略层集成
- 硬件资源管理
- 分布式训练协调

架构调用层次：
├── scenario_manager.py (本模块)
│   ├── 调用 backend/modules/training/strategies (策略层)
│   ├── 调用 backend/lib/hardware (硬件层)
│   ├── 调用 backend/lib/distributed (分布式层)
│   ├── 调用 backend/modules/training/progress (进度管理)
│   └── 管理各场景实例
└── 被服务层调用
"""

import logging
import os
import sys
import time
import threading
import uuid
import json
from typing import Optional, List, Dict, Any, Type, Callable, Union
from dataclasses import dataclass, field
from queue import PriorityQueue
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime
from enum import Enum

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 导入枚举类型
from backend.schemas.enums import TrainingScenario, TrainingPriority, ScheduleType
from backend.core.exceptions import TrainingError

logger = logging.getLogger(__name__)


# ==================== 策略层导入 ====================

from backend.modules.training.strategies.base_strategy import (
    StrategyContext, StrategyResult, StrategyMetrics,
)


# ==================== 分布式策略层导入 ====================

from backend.modules.training.strategies.distributed_strategy import (
    DistributedMode, DistributedStrategyConfig,
    diagnose_distributed_strategy,
)


# ==================== 硬件层导入 ====================

from backend.lib.hardware import (
    DeviceManager, get_device_manager,
    get_available_memory, clear_memory,
    recommend_batch_size,
)


# ==================== 分布式层导入 ====================

from backend.lib.distributed import (
    DistributedManager, get_distributed_manager,
)


# ==================== 进度管理导入 ====================

from backend.modules.training.progress.progress_manager import (
    TrainingProgressManager, TrainingProgress, get_progress_manager,
)


# ==================== 编排器模块导入 ====================

from backend.modules.training.orchestrator import (
    UnifiedTrainingOrchestrator, LayerManager, LayerConfig,
    OrchestratorPlan,
    create_orchestrator, create_quick_plan,
)


# ==================== 流水线模块导入 ====================

from backend.modules.training.pipeline import (
    PipelineDefinition, PipelineStep,
    PipelineExecutor, PipelineRunner,
    create_pipeline, execute_pipeline,
)


# ==================== 插件模块导入 ====================

from backend.modules.training.plugins import (
    TrainingPlugin, PluginRegistry,
    PluginContext, HookPoint,
    get_plugin_registry, execute_hook,
)


# ==================== 任务状态枚举 ====================

class JobStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    QUEUED = "queued"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


# ==================== 数据类 ====================

@dataclass
class ScenarioConfig:
    """场景配置"""
    scenario: TrainingScenario
    name: str
    description: str = ""
    
    # 调度配置
    schedule_config: Any = None
    training_method: Any = None
    
    # 输出和模型配置
    output_dir: str = "./outputs"
    base_model_path: Optional[str] = None
    
    # 分布式配置
    use_distributed: bool = False
    distributed_mode: str = "ddp"
    world_size: int = 1
    
    # 其他配置
    enable_wandb: bool = False
    device: str = "auto"
    precision: str = "fp16"
    
    # 自定义配置
    custom_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_config is None:
            self.custom_config = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'scenario': self.scenario.value if isinstance(self.scenario, TrainingScenario) else self.scenario,
            'name': self.name,
            'description': self.description,
            'output_dir': self.output_dir,
            'base_model_path': self.base_model_path,
            'use_distributed': self.use_distributed,
            'distributed_mode': self.distributed_mode,
            'world_size': self.world_size,
            'enable_wandb': self.enable_wandb,
            'device': self.device,
            'precision': self.precision,
            'custom_config': self.custom_config,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScenarioConfig':
        """从字典创建"""
        if 'scenario' in data:
            if isinstance(data['scenario'], str):
                try:
                    data['scenario'] = TrainingScenario(data['scenario'])
                except ValueError:
                    data['scenario'] = TrainingScenario.BASIC
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class TrainingJob:
    """训练任务"""
    job_id: str
    user_id: str
    scenario_type: str
    name: str
    description: str
    config: Dict[str, Any]
    priority: int
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # 策略层扩展
    strategy_context_id: Optional[str] = None
    
    # 资源使用
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'job_id': self.job_id,
            'user_id': self.user_id,
            'scenario_type': self.scenario_type,
            'name': self.name,
            'description': self.description,
            'config': self.config,
            'priority': self.priority,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'scheduled_at': self.scheduled_at.isoformat() if self.scheduled_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status,
            'result': self.result,
            'error': self.error,
            'metrics': self.metrics,
            'resource_usage': self.resource_usage,
        }
    
    def update_status(self, new_status: Union[str, JobStatus]) -> None:
        """更新任务状态"""
        if isinstance(new_status, JobStatus):
            new_status = new_status.value
        self.status = new_status
        
        if new_status == JobStatus.RUNNING.value and not self.started_at:
            self.started_at = datetime.now()
        elif new_status in [JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
            self.completed_at = datetime.now()


@dataclass
class TrainingJobWrapper:
    """训练任务包装器（用于优先队列）"""
    job: TrainingJob
    priority: int  # 数值越小优先级越高
    
    def __lt__(self, other):
        """用于优先队列排序"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.job.created_at < other.job.created_at


# ==================== 场景管理器 ====================

class ScenarioManager:
    """训练场景管理器
    
    生产级场景管理器，支持：
    - 任务优先级队列调度
    - 并发任务控制
    - 进度追踪和状态管理
    - 策略层集成
    - 硬件资源管理
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        max_concurrent_jobs: Optional[int] = None,
        enable_progress_tracking: bool = True,
        enable_resource_monitoring: bool = True,
    ):
        # 避免重复初始化
        if getattr(self, '_initialized', False):
            return
        
        self.max_concurrent_jobs = max_concurrent_jobs or self._get_recommended_concurrent_jobs()
        self.enable_progress_tracking = enable_progress_tracking
        self.enable_resource_monitoring = enable_resource_monitoring
        
        # 场景注册表
        self.scenario_registry: Dict[str, Type] = {}
        
        # 任务队列和管理
        self.job_queue = PriorityQueue()
        self.running_jobs: Dict[str, Future] = {}
        self.running_job_info: Dict[str, TrainingJob] = {}
        self.completed_jobs: Dict[str, TrainingJob] = {}
        self.job_history: List[TrainingJob] = []
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_jobs)
        
        # 调度器
        self.scheduler_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        
        # 锁
        self._job_lock = threading.Lock()
        
        # 策略层组件
        self._strategy_metrics: Optional['StrategyMetrics'] = None
        self._init_strategy_components()
        
        # 硬件层组件
        self._device_manager: Optional['DeviceManager'] = None
        self._init_hardware_components()
        
        # 分布式层组件
        self._distributed_manager: Optional['DistributedManager'] = None
        
        # 进度管理器
        self._progress_manager: Optional['TrainingProgressManager'] = None
        self._init_progress_manager()
        
        # 编排器组件
        self._orchestrator: Optional['UnifiedTrainingOrchestrator'] = None
        self._layer_manager: Optional['LayerManager'] = None
        self._init_orchestrator()
        
        # 流水线组件
        self._pipeline_runners: Dict[str, 'PipelineRunner'] = {}
        self._pipeline_executors: Dict[str, 'PipelineExecutor'] = {}
        self._init_pipeline()
        
        # 插件组件
        self._plugin_registry: Optional['PluginRegistry'] = None
        self._init_plugins()
        
        # 回调
        self.callbacks: Dict[str, List[Callable]] = {
            'job_started': [],
            'job_completed': [],
            'job_failed': [],
            'job_cancelled': [],
            'progress_updated': [],
        }
        
        self._initialized = True
        
        logger.info(f"ScenarioManager initialized with max_concurrent_jobs={self.max_concurrent_jobs}")
        logger.info(f"Layer availability: {self._get_layer_availability()}")
    
    def _get_recommended_concurrent_jobs(self) -> int:
        """获取推荐的并发任务数"""
        base = 3
        
        # 根据硬件层推荐
        try:
            device_manager = get_device_manager()
            if device_manager is not None:
                if hasattr(device_manager, 'get_device_count'):
                    base = max(1, device_manager.get_device_count())
        except Exception:
            pass
        
        return base
    
    def _init_strategy_components(self) -> None:
        """初始化策略层组件"""
        try:
            if StrategyMetrics is not None:
                self._strategy_metrics = StrategyMetrics()
        except Exception as e:
            logger.warning(f"Failed to init strategy components: {e}")
    
    def _init_hardware_components(self) -> None:
        """初始化硬件层组件"""
        try:
            if get_device_manager is not None:
                self._device_manager = get_device_manager()
        except Exception as e:
            logger.warning(f"Failed to init hardware components: {e}")
    
    def _init_progress_manager(self) -> None:
        """初始化进度管理器"""
        if not self.enable_progress_tracking:
            return
        
        try:
            if get_progress_manager is not None:
                self._progress_manager = get_progress_manager()
            elif TrainingProgressManager is not None:
                self._progress_manager = TrainingProgressManager()
        except Exception as e:
            logger.warning(f"Failed to init progress manager: {e}")
    
    def _init_orchestrator(self) -> None:
        """初始化编排器组件"""
        try:
            if create_orchestrator is not None:
                self._orchestrator = create_orchestrator(output_dir='./outputs/manager')
                logger.debug("Orchestrator created for manager")
            
            if LayerManager is not None and LayerConfig is not None:
                self._layer_manager = LayerManager(LayerConfig())
                logger.debug("Layer manager created for manager")
        except Exception as e:
            logger.warning(f"Failed to init orchestrator: {e}")
    
    def _init_pipeline(self) -> None:
        """初始化流水线组件"""
        logger.debug("Pipeline module available for manager")
    
    def _init_plugins(self) -> None:
        """初始化插件组件"""
        try:
            if get_plugin_registry is not None:
                self._plugin_registry = get_plugin_registry()
                logger.debug("Plugin registry obtained for manager")
        except Exception as e:
            logger.warning(f"Failed to init plugins: {e}")
    
    def _get_layer_availability(self) -> Dict[str, bool]:
        """获取各层可用性"""
        return {
        }
    
    def register_scenario(self, name: str, scenario_class: Type) -> None:
        """注册场景类型
        
        Args:
            name: 场景名称
            scenario_class: 场景类
        """
        self.scenario_registry[name] = scenario_class
        logger.info(f"Registered scenario: {name}")
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """注册回调函数
        
        Args:
            event: 事件名称
            callback: 回调函数
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, data: Dict[str, Any]) -> None:
        """触发回调"""
        if event not in self.callbacks:
            return
        
        for callback in self.callbacks[event]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
    
    def submit_job(
        self,
        user_id: str,
        scenario_type: str,
        name: str,
        config: Dict[str, Any],
        priority: int = 5,
        description: str = "",
        scheduled_at: Optional[datetime] = None,
    ) -> str:
        """提交训练任务
        
        Args:
            user_id: 用户 ID
            scenario_type: 场景类型
            name: 任务名称
            config: 任务配置
            priority: 优先级 (1-10, 1 为最高)
            description: 任务描述
            scheduled_at: 计划执行时间
        
        Returns:
            任务 ID
        """
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        
        job = TrainingJob(
            job_id=job_id,
            user_id=user_id,
            scenario_type=scenario_type,
            name=name,
            description=description,
            config=config,
            priority=min(max(priority, 1), 10),
            created_at=datetime.now(),
            scheduled_at=scheduled_at,
            status=JobStatus.PENDING.value,
        )
        
        with self._job_lock:
            # 添加到队列
            wrapper = TrainingJobWrapper(job=job, priority=priority)
            self.job_queue.put(wrapper)
            job.update_status(JobStatus.QUEUED)
            
            # 记录历史
            self.job_history.append(job)
        
        logger.info(f"Job submitted: {job_id}, scenario={scenario_type}, priority={priority}")
        
        # 如果调度器未运行，启动它
        if not self.scheduler_running:
            self.start_scheduler()
        
        return job_id
    
    def start_scheduler(self) -> None:
        """启动任务调度器"""
        if self.scheduler_running:
            return
        
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Job scheduler started")
    
    def stop_scheduler(self) -> None:
        """停止任务调度器"""
        self.scheduler_running = False
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        logger.info("Job scheduler stopped")
    
    def _scheduler_loop(self) -> None:
        """调度器主循环"""
        while self.scheduler_running:
            try:
                # 检查是否可以启动新任务
                with self._job_lock:
                    if len(self.running_jobs) >= self.max_concurrent_jobs:
                        time.sleep(1)
                        continue
                    
                    if self.job_queue.empty():
                        time.sleep(1)
                        continue
                    
                    # 获取下一个任务
                    wrapper = self.job_queue.get_nowait()
                    job = wrapper.job
                    
                    # 检查计划时间
                    if job.scheduled_at and job.scheduled_at > datetime.now():
                        # 重新放回队列
                        self.job_queue.put(wrapper)
                        time.sleep(1)
                        continue
                    
                    # 启动任务
                    future = self.executor.submit(self._run_job, job)
                    self.running_jobs[job.job_id] = future
                    self.running_job_info[job.job_id] = job
                    job.update_status(JobStatus.RUNNING)
                
                logger.info(f"Job started: {job.job_id}")
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(1)
    
    def _run_job(self, job: TrainingJob) -> Dict[str, Any]:
        """执行任务
        
        Args:
            job: 训练任务
        
        Returns:
            执行结果
        """
        try:
            # 触发开始回调
            self._trigger_callbacks('job_started', job.to_dict())
            
            # 清理内存
            try:
                clear_memory()
            except Exception:
                pass
            
            # 获取或创建场景实例
            scenario = self._create_scenario(job)
            
            if scenario is None:
                raise TrainingError(f"Failed to create scenario: {job.scenario_type}")
            
            # 运行场景
            result = scenario.run()
            
            # 更新任务
            job.result = result if isinstance(result, dict) else {'status': 'success'}
            job.update_status(JobStatus.COMPLETED)
            
            # 记录资源使用
            if self.enable_resource_monitoring:
                job.resource_usage = self._get_resource_usage()
            
            # 触发完成回调
            self._trigger_callbacks('job_completed', job.to_dict())
            
            logger.info(f"Job completed: {job.job_id}")
            return job.result
            
        except Exception as e:
            job.error = str(e)
            job.update_status(JobStatus.FAILED)
            
            # 触发失败回调
            self._trigger_callbacks('job_failed', job.to_dict())
            
            logger.error(f"Job failed: {job.job_id}, error={e}")
            import traceback
            traceback.print_exc()
            
            return {'success': False, 'error': str(e)}
        
        finally:
            # 清理
            with self._job_lock:
                if job.job_id in self.running_jobs:
                    del self.running_jobs[job.job_id]
                if job.job_id in self.running_job_info:
                    del self.running_job_info[job.job_id]
                self.completed_jobs[job.job_id] = job
            
            # 清理内存
            try:
                clear_memory()
            except Exception:
                pass
    
    def _create_scenario(self, job: TrainingJob):
        """创建场景实例
        
        Args:
            job: 训练任务
        
        Returns:
            场景实例
        """
        scenario_type = job.scenario_type
        
        # 检查注册的场景
        if scenario_type in self.scenario_registry:
            scenario_class = self.scenario_registry[scenario_type]
            return scenario_class(job.config, session_id=job.job_id)
        
        # 使用内置场景
        try:
            if scenario_type == 'basic_model':
                from backend.modules.training.scenarios.basic_model_scenario import BasicModelScenario
                return BasicModelScenario(job.config, session_id=job.job_id)
            
            elif scenario_type == 'advanced_model':
                from backend.modules.training.scenarios.advanced_model_scenario import AdvancedModelScenario
                return AdvancedModelScenario(job.config, session_id=job.job_id)
            
            elif scenario_type == 'industry' or scenario_type.startswith('industry_'):
                from backend.modules.training.scenarios.industry_scenario import IndustryScenario, IndustryScenarioConfig
                config = IndustryScenarioConfig(**job.config) if isinstance(job.config, dict) else job.config
                return IndustryScenario(config, session_id=job.job_id)
            
            elif scenario_type == 'scheduled':
                from backend.modules.training.scenarios.scheduled_training_scenario import ScheduledTrainingScenario
                return ScheduledTrainingScenario(job.config, session_id=job.job_id)
            
        except ImportError as e:
            logger.error(f"Failed to import scenario {scenario_type}: {e}")
        
        return None
    
    def _get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        usage = {}
        
        try:
            usage['available_memory_mb'] = get_available_memory()
        except Exception:
            pass
        
        try:
            import torch
            if torch.cuda.is_available():
                usage['cuda_allocated_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
                usage['cuda_max_allocated_mb'] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        except Exception:
            pass
        
        return usage
    
    def get_active_jobs(self) -> List[Dict[str, Any]]:
        """获取活跃的训练任务列表
        
        Returns:
            活跃任务列表
        """
        try:
            with self._job_lock:
                active_jobs = []
                
                # 获取正在运行的任务
                for job_id, job in self.running_job_info.items():
                    active_jobs.append({
                        **job.to_dict(),
                        'status': 'running',
                    })
                
                # 获取队列中等待的任务
                temp_queue = []
                while not self.job_queue.empty():
                    try:
                        job_wrapper = self.job_queue.get_nowait()
                        temp_queue.append(job_wrapper)
                        
                        job = job_wrapper.job
                        active_jobs.append({
                            **job.to_dict(),
                            'status': 'pending',
                        })
                    except:
                        break
                
                # 将任务重新放回队列
                for job_wrapper in temp_queue:
                    self.job_queue.put(job_wrapper)
                
                return active_jobs
                
        except Exception as e:
            logger.error(f"Failed to get active jobs: {e}")
            return []
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态
        
        Args:
            job_id: 任务 ID
        
        Returns:
            任务状态字典
        """
        with self._job_lock:
            # 检查运行中的任务
            if job_id in self.running_job_info:
                return self.running_job_info[job_id].to_dict()
            
            # 检查已完成的任务
            if job_id in self.completed_jobs:
                return self.completed_jobs[job_id].to_dict()
            
            # 检查历史
            for job in self.job_history:
                if job.job_id == job_id:
                    return job.to_dict()
        
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """取消训练任务
        
        Args:
            job_id: 任务 ID
        
        Returns:
            是否成功取消
        """
        try:
            with self._job_lock:
                # 检查是否在运行中的任务
                if job_id in self.running_jobs:
                    future = self.running_jobs[job_id]
                    if not future.done():
                        future.cancel()
                    
                    if job_id in self.running_job_info:
                        job = self.running_job_info[job_id]
                        job.update_status(JobStatus.CANCELLED)
                        self._trigger_callbacks('job_cancelled', job.to_dict())
                        del self.running_job_info[job_id]
                    
                    del self.running_jobs[job_id]
                    logger.info(f"Cancelled running job: {job_id}")
                    return True
                
                # 检查是否在队列中
                temp_queue = []
                found = False
                while not self.job_queue.empty():
                    try:
                        job_wrapper = self.job_queue.get_nowait()
                        if job_wrapper.job.job_id != job_id:
                            temp_queue.append(job_wrapper)
                        else:
                            found = True
                            job_wrapper.job.update_status(JobStatus.CANCELLED)
                            self._trigger_callbacks('job_cancelled', job_wrapper.job.to_dict())
                            logger.info(f"Removed job from queue: {job_id}")
                    except:
                        break
                
                # 将其他任务重新放回队列
                for job_wrapper in temp_queue:
                    self.job_queue.put(job_wrapper)
                
                return found
                
        except Exception as e:
            logger.error(f"Failed to cancel job: {e}")
            return False
    
    def pause_job(self, job_id: str) -> bool:
        """暂停任务
        
        Args:
            job_id: 任务 ID
        
        Returns:
            是否成功暂停
        """
        with self._job_lock:
            if job_id in self.running_job_info:
                job = self.running_job_info[job_id]
                job.update_status(JobStatus.PAUSED)
                logger.info(f"Paused job: {job_id}")
                return True
        return False
    
    def resume_job(self, job_id: str) -> bool:
        """恢复任务
        
        Args:
            job_id: 任务 ID
        
        Returns:
            是否成功恢复
        """
        with self._job_lock:
            if job_id in self.running_job_info:
                job = self.running_job_info[job_id]
                if job.status == JobStatus.PAUSED.value:
                    job.update_status(JobStatus.RUNNING)
                    logger.info(f"Resumed job: {job_id}")
                    return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息
        
        Returns:
            统计信息字典
        """
        try:
            with self._job_lock:
                total_jobs = len(self.job_history)
                running_jobs = len(self.running_jobs)
                pending_jobs = self.job_queue.qsize()
                completed_jobs = len(self.completed_jobs)
                
                # 计算各状态任务数
                failed_jobs = sum(1 for job in self.job_history if job.status == JobStatus.FAILED.value)
                cancelled_jobs = sum(1 for job in self.job_history if job.status == JobStatus.CANCELLED.value)
                successful_jobs = sum(1 for job in self.job_history if job.status == JobStatus.COMPLETED.value)
                
                # 计算平均执行时间
                durations = []
                for job in self.completed_jobs.values():
                    if job.started_at and job.completed_at:
                        duration = (job.completed_at - job.started_at).total_seconds()
                        durations.append(duration)
                
                avg_duration = sum(durations) / len(durations) if durations else 0
                
                stats = {
                    'total_jobs': total_jobs,
                    'running_jobs': running_jobs,
                    'pending_jobs': pending_jobs,
                    'completed_jobs': completed_jobs,
                    'successful_jobs': successful_jobs,
                    'failed_jobs': failed_jobs,
                    'cancelled_jobs': cancelled_jobs,
                    'success_rate': (successful_jobs / total_jobs * 100) if total_jobs > 0 else 0,
                    'avg_duration_seconds': avg_duration,
                    'max_concurrent_jobs': self.max_concurrent_jobs,
                    'layer_availability': self._get_layer_availability(),
                    'timestamp': datetime.now().isoformat(),
                }
                
                # 资源使用
                if self.enable_resource_monitoring:
                    stats['resource_usage'] = self._get_resource_usage()
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                'total_jobs': 0,
                'running_jobs': 0,
                'pending_jobs': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
            }
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断管理器状态
        
        Returns:
            诊断结果
        """
        diagnosis = {
            'manager_status': 'running' if self.scheduler_running else 'stopped',
            'layer_availability': self._get_layer_availability(),
            'statistics': self.get_statistics(),
            'registered_scenarios': list(self.scenario_registry.keys()),
            'components': {
                'strategy_metrics': self._strategy_metrics is not None,
                'device_manager': self._device_manager is not None,
                'distributed_manager': self._distributed_manager is not None,
                'progress_manager': self._progress_manager is not None,
            },
            'warnings': [],
            'errors': [],
        }
        
        # 诊断分布式策略
        try:
            diagnosis['distributed_diagnosis'] = diagnose_distributed_strategy()
        except Exception as e:
            diagnosis['warnings'].append(f"Distributed diagnosis failed: {e}")
        
        return diagnosis
    
    # ==================== 编排器方法 ====================
    
    def create_job_orchestrator_plan(
        self,
        job_id: str,
        phases: List[str] = None,
    ) -> Optional['OrchestratorPlan']:
        """为任务创建编排器计划
        
        Args:
            job_id: 任务 ID
            phases: 阶段列表
            
        Returns:
            编排器计划或 None
        """
        if  self._orchestrator is None:
            return None
        
        job = self.get_job_status(job_id)
        if job is None:
            return None
        
        try:
            name = f"job_{job_id}_plan"
            phases = phases or ['finetune']
            
            plan = self._orchestrator.create_plan(
                name=name,
                phases=phases,
            )
            logger.info(f"Created orchestrator plan for job {job_id}")
            return plan
            
        except Exception as e:
            logger.warning(f"Failed to create orchestrator plan: {e}")
            return None
    
    def get_orchestrator(self) -> Optional['UnifiedTrainingOrchestrator']:
        """获取编排器实例"""
        return self._orchestrator
    
    # ==================== 流水线方法 ====================
    
    def create_job_pipeline(
        self,
        job_id: str,
        steps: List[Dict[str, Any]] = None,
    ) -> Optional['PipelineDefinition']:
        """为任务创建流水线
        
        Args:
            job_id: 任务 ID
            steps: 步骤列表
            
        Returns:
            流水线定义或 None
        """
        job = self.get_job_status(job_id)
        if job is None:
            return None
        
        try:
            name = f"job_{job_id}_pipeline"
            steps = steps or [
                {'name': 'train', 'type': 'finetune', 'params': job.get('config', {})}
            ]
            
            pipeline = create_pipeline(name=name, steps=steps)
            logger.info(f"Created pipeline for job {job_id}")
            return pipeline
            
        except Exception as e:
            logger.warning(f"Failed to create pipeline: {e}")
            return None
    
    def execute_job_pipeline(
        self,
        job_id: str,
        pipeline: 'PipelineDefinition',
    ) -> Optional[Dict[str, Any]]:
        """执行任务流水线
        
        Args:
            job_id: 任务 ID
            pipeline: 流水线定义
            
        Returns:
            执行结果或 None
        """
        try:
            if job_id not in self._pipeline_runners and PipelineRunner is not None:
                self._pipeline_runners[job_id] = PipelineRunner(session_id=job_id)
            
            runner = self._pipeline_runners.get(job_id)
            result = execute_pipeline(pipeline, runner, session_id=job_id)
            
            logger.info(f"Pipeline executed for job {job_id}")
            return result.to_dict() if hasattr(result, 'to_dict') else {'success': True}
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # ==================== 插件方法 ====================
    
    def register_manager_plugin(self, plugin: 'TrainingPlugin') -> bool:
        """注册管理器级插件
        
        Args:
            plugin: 插件实例
            
        Returns:
            是否成功注册
        """
        if self._plugin_registry is None:
            return False
        
        try:
            self._plugin_registry.register(plugin)
            logger.info(f"Registered manager plugin: {plugin.name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to register plugin: {e}")
            return False
    
    def execute_manager_hook(
        self,
        hook: 'HookPoint',
        context_data: Dict[str, Any] = None,
    ) -> List:
        """执行管理器级插件钩子
        
        Args:
            hook: 钩子点
            context_data: 上下文数据
            
        Returns:
            执行结果列表
        """
        try:
            context_data = context_data or {}
            if PluginContext is not None:
                context = PluginContext(hook=hook, **context_data)
                return execute_hook(hook, context)
        except Exception as e:
            logger.warning(f"Hook execution failed: {e}")
        
        return []
    
    def shutdown(self) -> None:
        """关闭管理器"""
        logger.info("Shutting down ScenarioManager...")
        
        # 停止调度器
        self.stop_scheduler()
        
        # 取消所有运行中的任务
        for job_id in list(self.running_jobs.keys()):
            try:
                self.cancel_job(job_id)
            except Exception:
                pass
        
        # 关闭线程池
        self.executor.shutdown(wait=False)
        
        # 清理内存
        try:
            clear_memory()
        except Exception:
            pass
        
        logger.info("ScenarioManager shutdown complete")


# ==================== 全局实例管理 ====================

_global_manager: Optional[ScenarioManager] = None
_global_lock = threading.Lock()


def get_scenario_manager(**kwargs) -> ScenarioManager:
    """获取全局场景管理器实例
    
    Args:
        **kwargs: 传递给 ScenarioManager 的参数
    
    Returns:
        ScenarioManager 实例
    """
    global _global_manager
    
    with _global_lock:
        if _global_manager is None:
            _global_manager = ScenarioManager(**kwargs)
        return _global_manager


def shutdown_scenario_manager() -> None:
    """关闭全局场景管理器"""
    global _global_manager
    
    with _global_lock:
        if _global_manager is not None:
            _global_manager.shutdown()
            _global_manager = None


def get_layer_availability() -> Dict[str, bool]:
    """获取各层可用性"""
    return {
    }


# ==================== 导出 ====================

__all__ = [
    # 管理器
    'ScenarioManager',
    'get_scenario_manager',
    'shutdown_scenario_manager',
    
    # 数据类
    'ScenarioConfig',
    'TrainingJob',
    'TrainingJobWrapper',
    
    # 枚举
    'JobStatus',
    'TrainingScenario',
    'TrainingPriority',
    'ScheduleType',
    
    # 便捷函数
    'get_layer_availability',
    
]
