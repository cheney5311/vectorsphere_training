# -*- coding: utf-8 -*-
"""训练任务服务层

提供训练任务管理的完整业务逻辑。

架构调用关系：
API层 (training_jobs_api.py)
    -> Service层 (本模块)
        -> Launcher层 (training_launcher.py)
        -> Repository层 (training_job_repository.py)
        -> Training模块 (backend/modules/training)
"""

import logging
import uuid
import threading
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

# =============================================================================
# 训练启动器集成 (统一通过 launcher 执行训练)
# =============================================================================

from backend.modules.training.launcher import (
    TrainingSystemLauncher,
    ProductionTrainingLauncher,
    get_module_availability,
    diagnose_launcher_module,
    create_production_training_config,
)

from backend.modules.training.progress import (
    get_progress_manager,
)

# 异常类
from backend.core.exceptions import (
    ValidationError, BusinessLogicError,
    ResourceNotFoundError as TrainingJobNotFoundException
)


class TrainingJobsService:
    """
    训练任务服务层
    
    提供训练任务的完整生命周期管理：
    - 创建、查询、删除任务
    - 开始、暂停、恢复、取消任务
    - 获取日志、指标、检查点
    - 统计信息
    
    调用关系：
    - 使用 TrainingJobRepository 进行数据持久化
    - 使用 TrainingLauncher 执行训练
    - 使用 ScenarioManager 管理训练场景
    """
    
    def __init__(self, use_memory_storage: bool = False):
        """
        初始化服务
        
        Args:
            use_memory_storage: 是否使用内存存储（用于测试）
        """
        self._use_memory_storage = use_memory_storage
        self._repo = None
        self._launcher_class = None
        self._scenario_manager = None
        self._running_jobs: Dict[str, Any] = {}  # 运行中的任务
        self._memory_store: Dict[str, Dict] = {}  # 内存存储
        
        self._init_repository()
        self._init_training_modules()
    
    def _init_repository(self):
        """初始化仓库层"""
        if self._use_memory_storage:
            self._repo = None
            return
        
        try:
            from backend.repositories.training_job_repository import TrainingJobRepository
            self._repo = TrainingJobRepository(use_memory_storage=False)
            logger.info("TrainingJobRepository initialized")
        except ImportError as e:
            logger.warning("Failed to import TrainingJobRepository: %s", e)
            self._repo = None
    
    def _init_training_modules(self):
        """初始化训练模块"""
        self._launcher_class = TrainingSystemLauncher
        logger.info("TrainingSystemLauncher available")
        try:
            availability = get_module_availability()
            logger.info("Launcher module availability: %s", availability)
        except Exception as e:
            logger.warning("Failed to check launcher availability: %s", e)
        
        try:
            from backend.modules.training.scenarios.scenario_manager import get_scenario_manager
            self._scenario_manager = get_scenario_manager()
            logger.info("ScenarioManager available")
        except ImportError as e:
            logger.warning("ScenarioManager not available: %s", e)
            self._scenario_manager = None
        
        # 初始化进度管理器
        try:
            self._progress_manager = get_progress_manager()
            logger.info("Progress manager available")
        except Exception as e:
            logger.warning("Failed to initialize progress manager: %s", e)
            self._progress_manager = None
    
    # ==================== 任务 CRUD ====================
    
    def create_job(
        self,
        user_id: str,
        tenant_id: str,
        job_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        创建训练任务
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            job_config: 任务配置
            
        Returns:
            创建的任务信息
        """
        # 验证配置
        self._validate_job_config(job_config)
        
        # 生成任务ID
        job_id = str(uuid.uuid4())
        
        # 构建任务数据
        job_data = {
            'job_id': job_id,
            'user_id': user_id,
            'tenant_id': tenant_id,
            'name': job_config.get('name', f'Training Job {job_id[:8]}'),
            'description': job_config.get('description', ''),
            'model_name': job_config.get('model_name'),
            'model_id': job_config.get('model_id'),
            'dataset_id': job_config.get('dataset_id'),
            'scenario_type': job_config.get('scenario_type', 'standard'),
            'training_mode': job_config.get('training_mode', 'standard'),
            'config': job_config,
            'status': 'pending',
            'progress': 0.0,
            'current_epoch': 0,
            'total_epochs': job_config.get('config', {}).get('epochs', 
                            job_config.get('epochs', 10)),
            'created_at': datetime.utcnow()
        }
        
        # 保存到数据库
        if self._repo is not None:
            try:
                # 使用字典创建
                job = self._repo.create(job_data)
                job_id = getattr(job, 'job_id', job_id)
            except Exception as e:
                logger.error("Failed to save job to database: %s", e)
                # 回退到内存存储
                self._memory_store[job_id] = job_data
        else:
            self._memory_store[job_id] = job_data
        
        logger.info("Training job created: %s", job_id)
        
        # 如果配置了立即执行，则自动启动
        schedule = job_config.get('schedule', {})
        if schedule.get('type') == 'immediate':
            self._start_job_async(job_id, user_id, tenant_id, job_config)
        
        return {
            'job_id': job_id,
            'name': job_data['name'],
            'status': 'pending',
            'scenario_type': job_data['scenario_type'],
            'created_at': job_data['created_at'].isoformat() if isinstance(job_data['created_at'], datetime) else job_data['created_at']
        }
    
    def get_job_detail(
        self,
        job_id: str,
        user_id: str,
        tenant_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        获取任务详情
        
        Args:
            job_id: 任务ID
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            任务详情字典
        """
        job = self._get_job(job_id, user_id, tenant_id)
        if not job:
            return None
        
        return self._format_job_detail(job)
    
    def list_jobs(
        self,
        user_id: str,
        tenant_id: str,
        status: Optional[str] = None,
        scenario_type: Optional[str] = None,
        page: int = 1,
        per_page: int = 20,
        sort_by: str = 'created_at',
        sort_order: str = 'desc'
    ) -> Dict[str, Any]:
        """
        获取任务列表
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            status: 状态过滤
            scenario_type: 场景类型过滤
            page: 页码
            per_page: 每页数量
            sort_by: 排序字段
            sort_order: 排序方向
            
        Returns:
            任务列表和分页信息
        """
        if self._repo is not None:
            try:
                # 转换参数
                limit = per_page
                offset = (page - 1) * per_page
                order_desc = (sort_order == 'desc')
                
                jobs = self._repo.list_by_tenant(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    status=status,
                    scenario_type=scenario_type,
                    limit=limit,
                    offset=offset,
                    order_by=sort_by,
                    order_desc=order_desc
                )
                
                # 获取总数 (近似值，Repository 没有提供带过滤的 count)
                total = self._repo.count_by_tenant(tenant_id, status)
                
                job_list = [self._format_job_for_list(j) for j in jobs]
                
            except Exception as e:
                logger.error("Failed to list jobs from database: %s", e)
                job_list, total = self._list_from_memory(
                    user_id, status, scenario_type, page, per_page
                )
        else:
            job_list, total = self._list_from_memory(
                user_id, status, scenario_type, page, per_page
            )
        
        return {
            'jobs': job_list,
            'total': total,
            'page': page,
            'per_page': per_page
        }
    
    def delete_job(
        self,
        job_id: str,
        user_id: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """
        删除任务
        
        只能删除非运行状态的任务
        """
        job = self._get_job(job_id, user_id, tenant_id)
        if not job:
            raise TrainingJobNotFoundException(job_id)
        
        status = self._get_job_status(job)
        if status == 'running':
            raise BusinessLogicError("无法删除运行中的任务，请先取消")
        
        # 从数据库删除
        if self._repo is not None:
            try:
                self._repo.delete(job_id, tenant_id)
            except Exception as e:
                logger.error("Failed to delete job from database: %s", e)
        
        # 从内存删除
        if job_id in self._memory_store:
            del self._memory_store[job_id]
        
        # 清理相关文件
        self._cleanup_job_files(job_id)
        
        logger.info(f"Training job deleted: {job_id}")
        
        return {'success': True, 'job_id': job_id}
    
    # ==================== 任务控制 ====================
    
    def start_job(
        self,
        job_id: str,
        user_id: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """开始训练任务"""
        job = self._get_job(job_id, user_id, tenant_id)
        if not job:
            raise TrainingJobNotFoundException(job_id)
        
        status = self._get_job_status(job)
        if status not in ['pending', 'created']:
            raise BusinessLogicError(f"无法启动状态为 {status} 的任务")
        
        # 获取配置
        config = self._get_job_config(job)
        
        # 更新状态
        self._update_job_status(job_id, user_id, tenant_id, 'running')
        
        # 异步启动训练
        self._start_job_async(job_id, user_id, tenant_id, config)
        
        return {
            'success': True,
            'job_id': job_id,
            'status': 'running',
            'started_at': datetime.utcnow().isoformat()
        }
    
    def pause_job(
        self,
        job_id: str,
        user_id: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """暂停训练任务"""
        job = self._get_job(job_id, user_id, tenant_id)
        if not job:
            raise TrainingJobNotFoundException(job_id)
        
        status = self._get_job_status(job)
        if status != 'running':
            raise BusinessLogicError(f"只能暂停运行中的任务，当前状态: {status}")
        
        # 停止训练线程
        self._pause_running_job(job_id)
        
        # 更新状态
        self._update_job_status(job_id, user_id, tenant_id, 'paused')
        
        return {
            'success': True,
            'job_id': job_id,
            'status': 'paused',
            'checkpoint_path': None # 暂不支持返回checkpoint path
        }
    
    def resume_job(
        self,
        job_id: str,
        user_id: str,
        tenant_id: str,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """恢复训练任务"""
        job = self._get_job(job_id, user_id, tenant_id)
        if not job:
            raise TrainingJobNotFoundException(job_id)
        
        status = self._get_job_status(job)
        if status != 'paused':
            raise BusinessLogicError(f"只能恢复暂停的任务，当前状态: {status}")
        
        # 获取配置
        config = self._get_job_config(job)
        
        # 如果指定了检查点，更新配置
        if checkpoint_path:
            config['resume_from_checkpoint'] = checkpoint_path
        
        # 更新状态
        self._update_job_status(job_id, user_id, tenant_id, 'running')
        
        # 异步恢复训练
        self._start_job_async(job_id, user_id, tenant_id, config)
        
        return {
            'success': True,
            'job_id': job_id,
            'status': 'running',
            'resumed_from': checkpoint_path
        }
    
    def cancel_job(
        self,
        job_id: str,
        user_id: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """取消训练任务"""
        job = self._get_job(job_id, user_id, tenant_id)
        if not job:
            raise TrainingJobNotFoundException(job_id)
        
        status = self._get_job_status(job)
        if status in ['completed', 'failed', 'cancelled']:
            raise BusinessLogicError(f"任务已结束，状态: {status}")
        
        # 停止训练线程（如果正在运行）
        if status == 'running':
            self._stop_running_job(job_id)
        
        # 更新状态
        self._update_job_status(job_id, user_id, tenant_id, 'cancelled')
        
        return {
            'success': True,
            'job_id': job_id,
            'status': 'cancelled'
        }
    
    def restart_job(
        self,
        job_id: str,
        user_id: str,
        tenant_id: str,
        from_checkpoint: bool = False,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """重启训练任务"""
        job = self._get_job(job_id, user_id, tenant_id)
        if not job:
            raise TrainingJobNotFoundException(job_id)
        
        # 获取原配置
        original_config = self._get_job_config(job)
        
        # 合并配置覆盖
        new_config = {**original_config}
        if config_overrides:
            new_config.update(config_overrides)
        
        # 如果从检查点恢复，查找最新检查点
        if from_checkpoint:
            checkpoints = self.get_job_checkpoints(job_id, user_id, tenant_id)
            if checkpoints.get('checkpoints'):
                latest_checkpoint = checkpoints['checkpoints'][0]
                new_config['resume_from_checkpoint'] = latest_checkpoint['path']
        
        # 创建新任务
        new_config['name'] = f"{new_config.get('name', 'Job')} (Restart)"
        result = self.create_job(user_id, tenant_id, new_config)
        
        return {
            'success': True,
            'new_job_id': result['job_id'],
            'original_job_id': job_id,
            'status': 'pending'
        }
    
    # ==================== 日志和指标 ====================
    
    def get_job_logs(
        self,
        job_id: str,
        user_id: str,
        tenant_id: str,
        log_type: Optional[str] = None,
        log_level: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取任务日志"""
        job = self._get_job(job_id, user_id, tenant_id)
        if not job:
            raise TrainingJobNotFoundException(job_id)
        
        if self._repo is not None:
            try:
                # 使用 list_logs 而不是 get_logs
                logs = self._repo.list_logs(
                    job_id=job_id,
                    log_type=log_type,
                    limit=limit,
                    offset=offset
                )
                # 内存中过滤 log_level
                if log_level:
                    logs = [l for l in logs if getattr(l, 'log_level', None) == log_level]
                    
                return {
                    'logs': [self._format_log(log) for log in logs],
                    'total': len(logs)
                }
            except Exception as e:
                logger.error("Failed to get logs from database: %s", e)
        
        # 返回空日志
        return {'logs': [], 'total': 0}
    
    def get_job_metrics(
        self,
        job_id: str,
        user_id: str,
        tenant_id: str,
        metric_names: Optional[List[str]] = None,
        start_epoch: Optional[int] = None,
        end_epoch: Optional[int] = None
    ) -> Dict[str, Any]:
        """获取任务指标"""
        job = self._get_job(job_id, user_id, tenant_id)
        if not job:
            raise TrainingJobNotFoundException(job_id)
        
        # 获取当前指标
        current_metrics = {}
        if hasattr(job, 'metrics') and job.metrics:
            current_metrics = job.metrics
        elif isinstance(job, dict) and 'metrics' in job:
            current_metrics = job['metrics']
        
        # 获取历史指标
        history = []
        if self._repo is not None:
            try:
                # 使用 list_logs
                logs = self._repo.list_logs(job_id=job_id, log_type='metric')
                for log in logs:
                    epoch = getattr(log, 'epoch', None)
                    if epoch is not None:
                        if start_epoch and epoch < start_epoch:
                            continue
                        if end_epoch and epoch > end_epoch:
                            continue
                        metrics = getattr(log, 'metrics', {}) or {}
                        if metric_names:
                            metrics = {k: v for k, v in metrics.items() if k in metric_names}
                        history.append({'epoch': epoch, **metrics})
            except Exception as e:
                logger.error("Failed to get metric history: %s", e)
        
        return {
            'current_metrics': current_metrics,
            'history': history
        }
    
    def get_job_checkpoints(
        self,
        job_id: str,
        user_id: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """获取任务检查点列表"""
        job = self._get_job(job_id, user_id, tenant_id)
        if not job:
            raise TrainingJobNotFoundException(job_id)
        
        checkpoints = []
        
        # 从结果中获取检查点信息
        result = None
        if hasattr(job, 'result') and job.result:
            result = job.result
        elif isinstance(job, dict) and 'result' in job:
            result = job['result']
        
        if result and 'checkpoints' in result:
            checkpoints = result['checkpoints']
        
        return {'checkpoints': checkpoints}
    
    # ==================== 统计信息 ====================
    
    def get_statistics(
        self,
        user_id: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """获取训练统计信息"""
        if self._repo is not None:
            try:
                # 修正参数顺序
                stats = self._repo.get_statistics(tenant_id, user_id)
                return stats
            except Exception as e:
                logger.error("Failed to get statistics from database: %s", e)
        
        # 从内存计算统计
        return self._calculate_memory_statistics(user_id)
    
    def get_statistics_by_scenario(
        self,
        _user_id: str,
        _tenant_id: str
    ) -> Dict[str, Any]:
        """按场景类型获取统计"""
        # TrainingJobRepository 没有 get_statistics_by_scenario 方法
        # 这里返回空字典，或者可以实现一个基于 list_by_tenant 的统计逻辑
        return {}
    
    # ==================== 批量操作 ====================
    
    def batch_cancel_jobs(
        self,
        job_ids: List[str],
        user_id: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """批量取消任务"""
        cancelled = []
        failed = []
        
        for job_id in job_ids:
            try:
                self.cancel_job(job_id, user_id, tenant_id)
                cancelled.append(job_id)
            except Exception as e:
                failed.append({'id': job_id, 'reason': str(e)})
        
        return {
            'cancelled': cancelled,
            'failed': failed
        }
    
    # ==================== 私有方法 ====================
    
    def _validate_job_config(self, config: Dict[str, Any]):
        """验证任务配置"""
        if not config.get('model_name'):
            raise ValidationError("model_name is required", field="model_name")
        if not config.get('scenario_type'):
            raise ValidationError("scenario_type is required", field="scenario_type")
    
    def _get_job(
        self,
        job_id: str,
        user_id: str,
        tenant_id: str
    ) -> Optional[Any]:
        """获取任务对象"""
        # 先从内存获取
        if job_id in self._memory_store:
            job = self._memory_store[job_id]
            if job.get('user_id') == user_id:
                return job
        
        # 从数据库获取
        if self._repo is not None:
            try:
                return self._repo.get_by_job_id(job_id, tenant_id)
            except Exception as e:
                logger.error("Failed to get job from database: %s", e)
        
        return None
    
    def _get_job_status(self, job) -> str:
        """获取任务状态"""
        if isinstance(job, dict):
            return job.get('status', 'unknown')
        return getattr(job, 'status', 'unknown')
    
    def _get_job_config(self, job) -> Dict[str, Any]:
        """获取任务配置"""
        if isinstance(job, dict):
            return job.get('config', {})
        return getattr(job, 'config', {}) or {}
    
    def _update_job_status(
        self,
        job_id: str,
        _user_id: str,
        tenant_id: str,
        status: str,
        error_message: str = None
    ):
        """更新任务状态"""
        update_data = {
            'status': status,
            'updated_at': datetime.utcnow()
        }
        
        if status == 'running':
            update_data['started_at'] = datetime.utcnow()
        elif status in ['completed', 'failed', 'cancelled']:
            update_data['completed_at'] = datetime.utcnow()
        
        if error_message:
            update_data['error_message'] = error_message
        
        # 更新数据库
        if self._repo is not None:
            try:
                # 使用 update_status 方法
                self._repo.update_status(
                    job_id=job_id,
                    tenant_id=tenant_id,
                    **update_data
                )
            except Exception as e:
                logger.error("Failed to update job status in database: %s", e)
        
        # 更新内存
        if job_id in self._memory_store:
            self._memory_store[job_id].update(update_data)
        
        # 记录日志
        self._add_job_log(job_id, 'status_change', "Status changed to %s" % status)
    
    def _add_job_log(
        self,
        job_id: str,
        log_type: str,
        message: str,
        **kwargs
    ):
        """添加任务日志"""
        if self._repo is not None:
            try:
                # 使用 create_log
                log_data = {
                    'job_id': job_id,
                    'log_type': log_type,
                    'message': message,
                    'created_at': datetime.utcnow(),
                    **kwargs
                }
                self._repo.create_log(log_data)
            except Exception as e:
                logger.error("Failed to add job log: %s", e)
    
    def _start_job_async(
        self,
        job_id: str,
        user_id: str,
        tenant_id: str,
        config: Dict[str, Any]
    ):
        """异步启动训练任务 (通过 Launcher 统一调度)"""
        def run_training():
            try:
                logger.info(f"Starting training for job {job_id}")
                
                # 初始化进度跟踪
                try:
                    progress_manager = get_progress_manager()
                    total_steps = config.get('total_steps', config.get('config', {}).get('total_steps', 1000))
                    progress_manager.create_progress_tracker(job_id, total_steps=total_steps)
                    progress_manager.set_status(job_id, 'running')
                except Exception as e:
                    logger.warning("Failed to initialize progress tracking: %s", e)
                
                # 使用训练启动器
                # 构建启动器配置
                launcher_config = self._build_launcher_config(config)
                launcher = TrainingSystemLauncher(launcher_config)

                # 分析配置
                analysis = launcher.analyze_config()
                logger.info(f"Launcher analysis for job {job_id}: {analysis}")

                # 选择训练器
                trainer = launcher.select_trainer(analysis)

                if trainer is None:
                    raise BusinessLogicError("Launcher returned no trainer")

                # 设置进度回调
                def progress_callback(progress: float, metrics: Dict = None):
                    self._update_job_progress(job_id, user_id, tenant_id, progress, metrics)
                    # 同时更新进度管理器
                    if metrics:
                        try:
                            progress_manager = get_progress_manager()
                            progress_manager.update_progress(job_id, progress=progress, metrics=metrics)
                        except Exception:
                            pass

                if hasattr(trainer, 'set_progress_callback'):
                    trainer.set_progress_callback(progress_callback)

                # 执行训练
                result = trainer.train()

                # 更新结果
                if result.get('success'):
                    self._complete_job(job_id, user_id, tenant_id, result)
                else:
                    self._fail_job(job_id, user_id, tenant_id, result.get('error', 'Training failed'))
                    
            except Exception as e:
                logger.error("Training failed for job %s: %s", job_id, e)
                self._fail_job(job_id, user_id, tenant_id, str(e))
        
        # 启动训练线程
        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()
        self._running_jobs[job_id] = {'thread': thread, 'stop_flag': False}
        
        logger.info(f"Training thread started for job {job_id}")
    
    def _build_launcher_config(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """构建启动器配置
        
        Args:
            job_config: 任务配置
            
        Returns:
            启动器配置字典
        """
        scenario_type = job_config.get('scenario_type', 'standard')
        training_mode = job_config.get('training_mode', 'standard')
        inner_config = job_config.get('config', {})
        
        launcher_config = {
            'model': {
                'name': job_config.get('model_name', 'default_model'),
                'id': job_config.get('model_id'),
                'type': job_config.get('model_type', 'transformer')
            },
            'training': {
                'mode': training_mode,
                'scenario_type': scenario_type,
                'num_epochs': inner_config.get('epochs', job_config.get('epochs', 10)),
                'batch_size': inner_config.get('batch_size', job_config.get('batch_size', 16)),
                'learning_rate': inner_config.get('learning_rate', job_config.get('learning_rate', 2e-5)),
                'output_dir': inner_config.get('output_dir', './outputs')
            },
            'data': {
                'dataset_id': job_config.get('dataset_id'),
                'train_path': inner_config.get('train_data_path')
            }
        }
        
        # 添加特定模式配置
        if training_mode == 'three_stage':
            launcher_config['three_stage'] = {
                'enabled': True,
                **inner_config.get('three_stage', {})
            }
        elif training_mode == 'distributed':
            launcher_config['distributed'] = {
                'enabled': True,
                **inner_config.get('distributed', {})
            }
        elif training_mode == 'multimodal':
            launcher_config['multimodal'] = {
                'enabled': True,
                **inner_config.get('multimodal', {})
            }
        elif training_mode == 'distillation':
            launcher_config['distillation'] = {
                'enabled': True,
                **inner_config.get('distillation', {})
            }
        elif training_mode == 'industry':
            launcher_config['industry'] = {
                'enabled': True,
                **inner_config.get('industry', {})
            }
        
        return launcher_config
    
    def launch_production_training(
        self,
        job_id: str,
        user_id: str,
        tenant_id: str,
        training_type: str = 'standard',
        **kwargs
    ) -> Dict[str, Any]:
        """使用生产级启动器启动训练
        
        Args:
            job_id: 任务ID
            user_id: 用户ID
            tenant_id: 租户ID
            training_type: 训练类型
            **kwargs: 额外配置
            
        Returns:
            启动结果
        """
        try:
            # 获取任务
            job = self._get_job(job_id, user_id, tenant_id)
            if not job:
                raise TrainingJobNotFoundException(job_id)
            
            config = self._get_job_config(job)
            
            # 构建生产级配置
            production_config = create_production_training_config(
                training_type=training_type,
                output_dir=config.get('output_dir', f'./outputs/{job_id}'),
                model_name=job.get('model_name') if isinstance(job, dict) else getattr(job, 'model_name', 'production_model'),
                num_epochs=config.get('epochs', kwargs.get('num_epochs', 10)),
                batch_size=config.get('batch_size', kwargs.get('batch_size', 16)),
                learning_rate=config.get('learning_rate', kwargs.get('learning_rate', 2e-5)),
                enable_checkpoint=kwargs.get('enable_checkpoint', True),
                enable_monitoring=kwargs.get('enable_monitoring', True),
            )
            
            # 更新状态
            self._update_job_status(job_id, user_id, tenant_id, 'running')
            
            # 在后台线程启动生产级训练
            def run_production():
                try:
                    launcher = ProductionTrainingLauncher(production_config)
                    result = launcher.launch_training()
                    
                    if result.get('success'):
                        self._complete_job(job_id, user_id, tenant_id, result)
                    else:
                        self._fail_job(job_id, user_id, tenant_id, result.get('error', 'Training failed'))
                except Exception as e:
                    logger.error("Production training failed: %s", e)
                    self._fail_job(job_id, user_id, tenant_id, str(e))
            
            thread = threading.Thread(target=run_production, daemon=True)
            thread.start()
            self._running_jobs[job_id] = {
                'thread': thread,
                'stop_flag': False,
                'launcher_type': 'production'
            }
            
            return {
                'success': True,
                'job_id': job_id,
                'status': 'running',
                'training_type': training_type,
                'message': '生产级训练已启动'
            }
            
        except TrainingJobNotFoundException:
            raise
        except Exception as e:
            logger.error(f"Failed to launch production training: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_launcher_diagnostics(self) -> Dict[str, Any]:
        """获取启动器诊断信息"""
        try:
            return {
                'available': True,
                'diagnostics': diagnose_launcher_module(),
                'module_availability': get_module_availability()
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def _simulate_training(
        self,
        job_id: str,
        user_id: str,
        tenant_id: str,
        config: Dict[str, Any]
    ):
        """模拟训练过程（用于测试）"""
        import time
        
        epochs = config.get('epochs', config.get('config', {}).get('epochs', 5))
        
        for epoch in range(1, epochs + 1):
            # 检查停止标志
            if job_id in self._running_jobs and self._running_jobs[job_id].get('stop_flag'):
                logger.info("Training stopped for job %s", job_id)
                return
            
            # 更新进度
            progress = (epoch / epochs) * 100
            metrics = {
                'epoch': epoch,
                'loss': 1.0 / epoch,
                'accuracy': 0.5 + 0.5 * (epoch / epochs)
            }
            self._update_job_progress(job_id, user_id, tenant_id, progress, metrics)
            
            time.sleep(0.5)  # 模拟训练时间
        
        # 完成训练
        self._complete_job(job_id, user_id, tenant_id, {
            'success': True,
            'final_loss': 0.1,
            'accuracy': 0.95
        })
    
    def _update_job_progress(
        self,
        job_id: str,
        _user_id: str,
        tenant_id: str,
        progress: float,
        metrics: Dict = None
    ):
        """更新任务进度"""
        update_data = {'progress': progress}
        if metrics:
            update_data['metrics'] = metrics
            if 'epoch' in metrics:
                update_data['current_epoch'] = metrics['epoch']
        
        # 更新数据库
        if self._repo is not None:
            try:
                # 使用 update_progress
                self._repo.update_progress(
                    job_id=job_id,
                    progress=progress,
                    metrics=metrics,
                    tenant_id=tenant_id,
                    current_epoch=update_data.get('current_epoch')
                )
            except Exception as e:
                logger.error("Failed to update progress: %s", e)
        
        # 更新内存
        if job_id in self._memory_store:
            self._memory_store[job_id].update(update_data)
        
        # 记录指标日志
        if metrics:
            self._add_job_log(job_id, 'metric', "Progress: %.1f%%" % progress, metrics=metrics)
    
    def _complete_job(
        self,
        job_id: str,
        _user_id: str,
        tenant_id: str,
        result: Dict[str, Any]
    ):
        """完成任务"""
        update_data = {
            'status': 'completed',
            'progress': 100.0,
            'result': result,
            'completed_at': datetime.utcnow()
        }
        
        if self._repo is not None:
            try:
                # 使用 update_status
                self._repo.update_status(
                    job_id=job_id,
                    tenant_id=tenant_id,
                    **update_data
                )
            except Exception as e:
                logger.error("Failed to complete job: %s", e)
        
        if job_id in self._memory_store:
            self._memory_store[job_id].update(update_data)
        
        if job_id in self._running_jobs:
            del self._running_jobs[job_id]
        
        self._add_job_log(job_id, 'status_change', "Training completed")
        logger.info("Training completed for job %s", job_id)
    
    def _fail_job(
        self,
        job_id: str,
        _user_id: str,
        tenant_id: str,
        error_message: str
    ):
        """任务失败"""
        self._update_job_status(job_id, _user_id, tenant_id, 'failed', error_message)
        
        if job_id in self._running_jobs:
            del self._running_jobs[job_id]
        
        self._add_job_log(job_id, 'error', "Training failed: %s" % error_message)
        logger.error("Training failed for job %s: %s", job_id, error_message)
    
    def _pause_running_job(self, job_id: str) -> Optional[str]:
        """暂停运行中的任务"""
        if job_id in self._running_jobs:
            self._running_jobs[job_id]['stop_flag'] = True
        
        # 返回检查点路径（如果有）
        return None
    
    def _stop_running_job(self, job_id: str):
        """停止运行中的任务"""
        if job_id in self._running_jobs:
            self._running_jobs[job_id]['stop_flag'] = True
            # 不直接删除，等待线程退出或后续清理
            # del self._running_jobs[job_id] 
            
    def _cleanup_job_files(self, job_id: str):
        """清理任务相关文件"""
        logger.info("Cleaning up files for job %s", job_id)
    
    def _format_job_detail(self, job) -> Dict[str, Any]:
        """格式化任务详情"""
        if isinstance(job, dict):
            return {
                'job_id': job.get('job_id'),
                'name': job.get('name'),
                'description': job.get('description'),
                'status': job.get('status'),
                'scenario_type': job.get('scenario_type'),
                'training_mode': job.get('training_mode'),
                'model_name': job.get('model_name'),
                'progress': job.get('progress', 0),
                'current_epoch': job.get('current_epoch', 0),
                'total_epochs': job.get('total_epochs', 0),
                'metrics': job.get('metrics', {}),
                'config': job.get('config', {}),
                'result': job.get('result'),
                'error_message': job.get('error_message'),
                'created_at': self._format_datetime(job.get('created_at')),
                'started_at': self._format_datetime(job.get('started_at')),
                'completed_at': self._format_datetime(job.get('completed_at'))
            }
        
        return {
            'job_id': getattr(job, 'job_id', None),
            'name': getattr(job, 'name', None),
            'description': getattr(job, 'description', None),
            'status': getattr(job, 'status', None),
            'scenario_type': getattr(job, 'scenario_type', None),
            'training_mode': getattr(job, 'training_mode', None),
            'model_name': getattr(job, 'model_name', None),
            'progress': getattr(job, 'progress', 0),
            'current_epoch': getattr(job, 'current_epoch', 0),
            'total_epochs': getattr(job, 'total_epochs', 0),
            'metrics': getattr(job, 'metrics', {}),
            'config': getattr(job, 'config', {}),
            'result': getattr(job, 'result', None),
            'error_message': getattr(job, 'error_message', None),
            'created_at': self._format_datetime(getattr(job, 'created_at', None)),
            'started_at': self._format_datetime(getattr(job, 'started_at', None)),
            'completed_at': self._format_datetime(getattr(job, 'completed_at', None))
        }
    
    def _format_job_for_list(self, job) -> Dict[str, Any]:
        """格式化任务列表项"""
        if isinstance(job, dict):
            return {
                'job_id': job.get('job_id'),
                'name': job.get('name'),
                'status': job.get('status'),
                'scenario_type': job.get('scenario_type'),
                'progress': job.get('progress', 0),
                'created_at': self._format_datetime(job.get('created_at'))
            }
        
        return {
            'job_id': getattr(job, 'job_id', None),
            'name': getattr(job, 'name', None),
            'status': getattr(job, 'status', None),
            'scenario_type': getattr(job, 'scenario_type', None),
            'progress': getattr(job, 'progress', 0),
            'created_at': self._format_datetime(getattr(job, 'created_at', None))
        }
    
    def _format_log(self, log) -> Dict[str, Any]:
        """格式化日志"""
        if isinstance(log, dict):
            return log
        
        return {
            'log_type': getattr(log, 'log_type', None),
            'log_level': getattr(log, 'log_level', None),
            'message': getattr(log, 'message', None),
            'details': getattr(log, 'details', None),
            'epoch': getattr(log, 'epoch', None),
            'step': getattr(log, 'step', None),
            'metrics': getattr(log, 'metrics', None),
            'created_at': self._format_datetime(getattr(log, 'created_at', None))
        }
    
    def _format_datetime(self, dt) -> Optional[str]:
        """格式化日期时间"""
        if dt is None:
            return None
        if isinstance(dt, str):
            return dt
        return dt.isoformat() if isinstance(dt, datetime) else str(dt)
    
    def _list_from_memory(
        self,
        user_id: str,
        status: Optional[str],
        scenario_type: Optional[str],
        page: int,
        per_page: int
    ) -> tuple:
        """从内存获取任务列表"""
        jobs = [
            j for j in self._memory_store.values()
            if j.get('user_id') == user_id
        ]
        
        if status:
            jobs = [j for j in jobs if j.get('status') == status]
        if scenario_type:
            jobs = [j for j in jobs if j.get('scenario_type') == scenario_type]
        
        total = len(jobs)
        offset = (page - 1) * per_page
        paginated = jobs[offset:offset + per_page]
        
        return [self._format_job_for_list(j) for j in paginated], total
    
    def _calculate_memory_statistics(self, user_id: str) -> Dict[str, Any]:
        """计算内存存储的统计信息"""
        jobs = [
            j for j in self._memory_store.values()
            if j.get('user_id') == user_id
        ]
        
        return {
            'total_jobs': len(jobs),
            'running_jobs': len([j for j in jobs if j.get('status') == 'running']),
            'pending_jobs': len([j for j in jobs if j.get('status') == 'pending']),
            'completed_jobs': len([j for j in jobs if j.get('status') == 'completed']),
            'failed_jobs': len([j for j in jobs if j.get('status') == 'failed']),
            'cancelled_jobs': len([j for j in jobs if j.get('status') == 'cancelled']),
            'success_rate': 0.0
        }


# ==================== 全局服务实例 ====================

_global_training_jobs_service: Optional[TrainingJobsService] = None


def get_training_jobs_service(use_memory_storage: bool = False) -> TrainingJobsService:
    """
    获取训练任务服务实例
    
    Args:
        use_memory_storage: 是否使用内存存储
        
    Returns:
        TrainingJobsService 实例
    """
    global _global_training_jobs_service
    
    if _global_training_jobs_service is None:
        _global_training_jobs_service = TrainingJobsService(use_memory_storage)
    
    return _global_training_jobs_service

