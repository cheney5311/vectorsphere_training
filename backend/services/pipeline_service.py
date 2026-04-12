"""训练流水线服务

提供训练流水线的业务逻辑，包括：
- 流水线创建、更新、删除
- 流水线执行、暂停、恢复、回滚
- 步骤执行管理
- 模板管理
"""

import logging
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import uuid

from backend.modules.training.pipeline.pipeline_definition import PipelineDefinition, PipelineStep
from backend.modules.training.pipeline.pipeline_executor import PipelineExecutor

logger = logging.getLogger(__name__)


class PipelineService:
    """训练流水线服务
    
    提供流水线的完整生命周期管理，支持数据库持久化
    """
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化流水线服务
        
        Args:
            use_memory_storage: 是否使用内存存储（用于测试）
        """
        self._use_memory_storage = use_memory_storage
        self.logger = logging.getLogger(__name__)
        
        # 初始化仓库层
        self._init_repositories(use_memory_storage)
        
        # 运行中的任务线程缓存
        self._running_tasks: Dict[str, threading.Thread] = {}
        
        # 初始化关联服务
        self._init_services()
    
    def _init_repositories(self, use_memory_storage: bool = False):
        """初始化仓库层"""
        try:
            from backend.repositories.pipeline_repository import (
                get_pipeline_repository,
                get_execution_repository,
                get_step_execution_repository,
                get_template_repository
            )
            
            self._pipeline_repo = get_pipeline_repository(use_memory_storage)
            self._execution_repo = get_execution_repository(use_memory_storage)
            self._step_repo = get_step_execution_repository(use_memory_storage)
            self._template_repo = get_template_repository(use_memory_storage)
            
            self.logger.info("Pipeline repositories initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize pipeline repositories: {e}")
            self._pipeline_repo = None
            self._execution_repo = None
            self._step_repo = None
            self._template_repo = None
    
    def _init_services(self):
        """初始化关联服务"""
        try:
            from backend.services.training_execution_service import get_training_execution_service
            from backend.modules.training.progress import get_progress_manager
            from backend.modules.training.pipeline.task_registry_interface import get_task_registry
            
            self._exec_service = get_training_execution_service()
            self._progress_mgr = get_progress_manager()
            self._task_registry = get_task_registry()
        except Exception as e:
            self.logger.warning(f"Failed to initialize related services: {e}")
            self._exec_service = None
            self._progress_mgr = None
            self._task_registry = None
    
    # ==========================================================================
    # 流水线管理
    # ==========================================================================
    
    def create_pipeline(self, name: str, steps_config: List[Dict], 
                       tenant_id: str, user_id: str,
                       description: Optional[str] = None,
                       model_name: Optional[str] = None,
                       model_id: Optional[str] = None,
                       dataset_id: Optional[str] = None,
                       global_config: Optional[Dict] = None,
                       enable_rollback: bool = True,
                       tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """创建流水线
        
        Args:
            name: 流水线名称
            steps_config: 步骤配置列表
            tenant_id: 租户ID
            user_id: 用户ID
            description: 描述
            model_name: 模型名称
            model_id: 模型ID
            dataset_id: 数据集ID
            global_config: 全局配置
            enable_rollback: 是否启用回滚
            tags: 标签列表
            
        Returns:
            创建的流水线信息
        """
        try:
            self.logger.info(f"Creating pipeline: {name}")
            
            # 验证步骤配置
            validated_steps = self._validate_steps_config(steps_config)
            
            pipeline_data = {
                'tenant_id': tenant_id,
                'name': name,
                'description': description,
                'user_id': user_id,
                'status': 'created',
                'model_name': model_name,
                'model_id': model_id,
                'dataset_id': dataset_id,
                'steps_config': validated_steps,
                'global_config': global_config,
                'enable_rollback': enable_rollback,
                'tags': tags
            }
            
            if self._pipeline_repo:
                result = self._pipeline_repo.create(pipeline_data)
                self.logger.info(f"Pipeline created: {result.get('pipeline_id')}")
                return result
            
            # 回退到返回基本数据
            pipeline_data['pipeline_id'] = f"pipe_{uuid.uuid4().hex[:8]}"
            pipeline_data['created_at'] = datetime.utcnow().isoformat()
            return pipeline_data
            
        except Exception as e:
            self.logger.error(f"Failed to create pipeline: {e}")
            raise
    
    def get_pipeline(self, pipeline_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """获取流水线详情
        
        Args:
            pipeline_id: 流水线ID
            tenant_id: 租户ID
            
        Returns:
            流水线信息
        """
        try:
            if self._pipeline_repo:
                return self._pipeline_repo.get_by_id(pipeline_id, tenant_id)
            return None
        except Exception as e:
            self.logger.error(f"Failed to get pipeline: {e}")
            return None
    
    def get_pipeline_by_name(self, name: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """通过名称获取流水线
        
        Args:
            name: 流水线名称
            tenant_id: 租户ID
            
        Returns:
            流水线信息
        """
        try:
            if self._pipeline_repo:
                return self._pipeline_repo.get_by_name(name, tenant_id)
            return None
        except Exception as e:
            self.logger.error(f"Failed to get pipeline by name: {e}")
            return None
    
    def list_pipelines(self, tenant_id: str, status: Optional[str] = None,
                      user_id: Optional[str] = None,
                      model_id: Optional[str] = None,
                      limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """获取流水线列表
        
        Args:
            tenant_id: 租户ID
            status: 状态过滤
            user_id: 用户ID过滤
            model_id: 模型ID过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            流水线列表和总数
        """
        try:
            if self._pipeline_repo:
                pipelines, total = self._pipeline_repo.list_by_tenant(
                    tenant_id=tenant_id,
                    status=status,
                    user_id=user_id,
                    model_id=model_id,
                    limit=limit,
                    offset=offset
                )
                return {'pipelines': pipelines, 'total': total, 'limit': limit, 'offset': offset}
            return {'pipelines': [], 'total': 0, 'limit': limit, 'offset': offset}
        except Exception as e:
            self.logger.error(f"Failed to list pipelines: {e}")
            return {'pipelines': [], 'total': 0, 'limit': limit, 'offset': offset}
    
    def update_pipeline(self, pipeline_id: str, tenant_id: str,
                       updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新流水线
        
        Args:
            pipeline_id: 流水线ID
            tenant_id: 租户ID
            updates: 更新内容
            
        Returns:
            更新后的流水线信息
        """
        try:
            if self._pipeline_repo:
                # 验证步骤配置（如果有更新）
                if 'steps_config' in updates:
                    updates['steps_config'] = self._validate_steps_config(updates['steps_config'])
                
                return self._pipeline_repo.update(pipeline_id, tenant_id, updates)
            return None
        except Exception as e:
            self.logger.error(f"Failed to update pipeline: {e}")
            return None
    
    def delete_pipeline(self, pipeline_id: str, tenant_id: str) -> bool:
        """删除流水线
        
        Args:
            pipeline_id: 流水线ID
            tenant_id: 租户ID
            
        Returns:
            是否删除成功
        """
        try:
            if self._pipeline_repo:
                return self._pipeline_repo.delete(pipeline_id, tenant_id)
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete pipeline: {e}")
            return False
    
    def get_pipeline_statistics(self, tenant_id: str, 
                               user_id: Optional[str] = None) -> Dict[str, Any]:
        """获取流水线统计信息
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            统计信息
        """
        try:
            if self._pipeline_repo:
                return self._pipeline_repo.get_statistics(tenant_id, user_id)
            return {'total': 0, 'by_status': {}}
        except Exception as e:
            self.logger.error(f"Failed to get pipeline statistics: {e}")
            return {'total': 0, 'by_status': {}}
    
    # ==========================================================================
    # 流水线执行
    # ==========================================================================
    
    def start_pipeline(self, pipeline_id: str, tenant_id: str, user_id: str,
                      session_id: Optional[str] = None,
                      runtime_config: Optional[Dict] = None) -> Dict[str, Any]:
        """启动流水线执行
        
        Args:
            pipeline_id: 流水线ID
            tenant_id: 租户ID
            user_id: 用户ID
            session_id: 自定义会话ID（可选）
            runtime_config: 运行时配置（可选）
            
        Returns:
            执行信息
        """
        try:
            self.logger.info(f"Starting pipeline: {pipeline_id}")
            
            # 获取流水线定义
            pipeline = self.get_pipeline(pipeline_id, tenant_id)
            if not pipeline:
                # 尝试通过名称获取
                pipeline = self.get_pipeline_by_name(pipeline_id, tenant_id)
            
            if not pipeline:
                return {'success': False, 'error': 'Pipeline not found'}
            
            # 检查是否已有正在运行的执行
            if self._execution_repo:
                running, _ = self._execution_repo.list_by_pipeline(
                    pipeline_id=pipeline.get('pipeline_id', pipeline_id),
                    tenant_id=tenant_id,
                    status='running'
                )
                if running:
                    return {'success': False, 'error': 'Pipeline already running', 
                           'execution_id': running[0].get('execution_id')}
            
            # 创建执行记录
            steps_config = pipeline.get('steps_config', [])
            execution_data = {
                'tenant_id': tenant_id,
                'pipeline_id': pipeline.get('pipeline_id', pipeline_id),
                'session_id': session_id,
                'user_id': user_id,
                'status': 'queued',
                'total_steps': len(steps_config),
                'pipeline_snapshot': pipeline,
                'runtime_config': runtime_config
            }
            
            execution = None
            if self._execution_repo:
                execution = self._execution_repo.create(execution_data)
            else:
                execution = {
                    'execution_id': f"exec_{uuid.uuid4().hex[:8]}",
                    'session_id': session_id or f"sess_{uuid.uuid4().hex[:8]}",
                    'status': 'queued',
                    **execution_data
                }
            
            actual_session_id = execution.get('session_id')
            
            # 创建步骤执行记录
            if self._step_repo:
                for idx, step_config in enumerate(steps_config):
                    step_data = {
                        'tenant_id': tenant_id,
                        'execution_id': execution.get('execution_id'),
                        'step_index': idx,
                        'step_name': step_config.get('name', f'step_{idx}'),
                        'step_type': step_config.get('type', 'custom'),
                        'status': 'pending',
                        'step_config': step_config,
                        'failure_policy': step_config.get('on_fail', 'rollback')
                    }
                    self._step_repo.create(step_data)
            
            # 初始化进度跟踪器
            if self._progress_mgr:
                try:
                    self._progress_mgr.create_progress_tracker(actual_session_id, total_steps=len(steps_config))
                except Exception:
                    pass
            
            # 注册任务
            if self._task_registry:
                try:
                    self._task_registry.ensure_task(actual_session_id)
                except Exception:
                    pass
            
            # 更新状态为运行中
            if self._execution_repo:
                self._execution_repo.update_status(
                    execution.get('execution_id'),
                    tenant_id,
                    'running'
                )
            
            if self._pipeline_repo:
                self._pipeline_repo.update_status(
                    pipeline.get('pipeline_id', pipeline_id),
                    tenant_id,
                    'running'
                )
            
            # 启动后台执行线程
            t = threading.Thread(
                target=self._run_pipeline_async,
                args=(pipeline, execution, tenant_id),
                daemon=True
            )
            self._running_tasks[execution.get('execution_id')] = t
            t.start()
            
            self.logger.info(f"Pipeline started: execution_id={execution.get('execution_id')}, session_id={actual_session_id}")
            
            return {
                'success': True,
                'execution_id': execution.get('execution_id'),
                'session_id': actual_session_id,
                'status': 'running',
                'pipeline_id': pipeline.get('pipeline_id', pipeline_id),
                'total_steps': len(steps_config)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to start pipeline: {e}")
            return {'success': False, 'error': str(e)}
    
    def _run_pipeline_async(self, pipeline: Dict, execution: Dict, tenant_id: str):
        """异步执行流水线
        
        Args:
            pipeline: 流水线配置
            execution: 执行记录
            tenant_id: 租户ID
        """
        execution_id = execution.get('execution_id')
        session_id = execution.get('session_id')
        
        try:
            # 构建 PipelineDefinition
            steps_config = pipeline.get('steps_config', [])
            pd = PipelineDefinition(
                name=pipeline.get('name'),
                tenant_id=tenant_id,
                session_id=session_id,
                model_name=pipeline.get('model_name'),
                steps=[
                    PipelineStep(
                        name=s.get('name', s.get('type', 'step')),
                        type=s.get('type', 'custom'),
                        on_fail=s.get('on_fail', 'rollback')
                    ) for s in steps_config
                ],
                enable_rollback=pipeline.get('enable_rollback', True)
            )
            
            # 执行流水线
            executor = PipelineExecutor(session_id=session_id)
            result = executor.execute(pd)
            
            # 判断执行结果
            all_success = all(r.get('status') == 'success' for r in result.get('result', []))
            final_status = 'completed' if all_success else 'failed'
            
            # 更新执行记录
            if self._execution_repo:
                self._execution_repo.update_status(
                    execution_id, tenant_id, final_status,
                    result=result
                )
            
            # 更新流水线状态
            if self._pipeline_repo:
                self._pipeline_repo.update_status(
                    pipeline.get('pipeline_id'),
                    tenant_id,
                    final_status
                )
            
            # 更新步骤执行记录
            if self._step_repo:
                steps = self._step_repo.list_by_execution(execution_id, tenant_id)
                for idx, step_result in enumerate(result.get('result', [])):
                    if idx < len(steps):
                        self._step_repo.update_status(
                            steps[idx].get('step_execution_id'),
                            tenant_id,
                            step_result.get('status', 'completed'),
                            output_data=step_result.get('details'),
                            metrics=step_result.get('metrics')
                        )
            
            self.logger.info(f"Pipeline execution completed: {execution_id}, status={final_status}")
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {execution_id}, error={e}")
            
            # 更新状态为失败
            if self._execution_repo:
                self._execution_repo.update_status(
                    execution_id, tenant_id, 'failed',
                    error_message=str(e)
                )
            
            if self._pipeline_repo:
                self._pipeline_repo.update_status(
                    pipeline.get('pipeline_id'),
                    tenant_id,
                    'failed'
                )
    
    def pause_execution(self, session_id: str, tenant_id: str) -> Dict[str, Any]:
        """暂停流水线执行
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID
            
        Returns:
            操作结果
        """
        try:
            self.logger.info(f"Pausing execution: session_id={session_id}")
            
            # 通过训练执行服务暂停
            if self._exec_service:
                try:
                    self._exec_service.pause_training(session_id)
                except Exception as e:
                    self.logger.warning(f"Failed to pause via exec service: {e}")
            
            # 通过任务注册表暂停
            if self._task_registry:
                try:
                    self._task_registry.pause(session_id)
                except Exception:
                    pass
            
            # 更新进度状态
            if self._progress_mgr:
                try:
                    self._progress_mgr.update_progress(session_id, status="paused")
                except Exception:
                    pass
            
            # 更新数据库中的执行状态
            if self._execution_repo:
                execution = self._execution_repo.get_by_session_id(session_id, tenant_id)
                if execution:
                    self._execution_repo.update_status(
                        execution.get('execution_id'),
                        tenant_id,
                        'paused'
                    )
            
            return {'success': True, 'message': 'Pipeline paused', 'session_id': session_id}
            
        except Exception as e:
            self.logger.error(f"Failed to pause execution: {e}")
            return {'success': False, 'error': str(e)}
    
    def resume_execution(self, session_id: str, tenant_id: str) -> Dict[str, Any]:
        """恢复流水线执行
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID
            
        Returns:
            操作结果
        """
        try:
            self.logger.info(f"Resuming execution: session_id={session_id}")
            
            # 通过训练执行服务恢复
            if self._exec_service:
                try:
                    self._exec_service.resume_training(session_id)
                except Exception as e:
                    self.logger.warning(f"Failed to resume via exec service: {e}")
            
            # 通过任务注册表恢复
            if self._task_registry:
                try:
                    self._task_registry.resume(session_id)
                except Exception:
                    pass
            
            # 更新进度状态
            if self._progress_mgr:
                try:
                    self._progress_mgr.start_training(session_id)
                except Exception:
                    pass
            
            # 更新数据库中的执行状态
            if self._execution_repo:
                execution = self._execution_repo.get_by_session_id(session_id, tenant_id)
                if execution:
                    self._execution_repo.update_status(
                        execution.get('execution_id'),
                        tenant_id,
                        'running'
                    )
            
            return {'success': True, 'message': 'Pipeline resumed', 'session_id': session_id}
            
        except Exception as e:
            self.logger.error(f"Failed to resume execution: {e}")
            return {'success': False, 'error': str(e)}
    
    def rollback_execution(self, pipeline_id: str, tenant_id: str, 
                          session_id: Optional[str] = None) -> Dict[str, Any]:
        """回滚流水线执行
        
        Args:
            pipeline_id: 流水线ID/名称
            tenant_id: 租户ID
            session_id: 会话ID（可选）
            
        Returns:
            操作结果
        """
        try:
            self.logger.info(f"Rolling back pipeline: {pipeline_id}")
            
            rollback_event = {
                'event': 'rollback',
                'timestamp': datetime.utcnow().isoformat(),
                'session_id': session_id
            }
            
            # 停止训练
            if session_id and self._exec_service:
                try:
                    self._exec_service.stop_training(session_id)
                except Exception:
                    pass
            
            # 取消任务
            if session_id and self._task_registry:
                try:
                    self._task_registry.cancel(session_id)
                except Exception:
                    pass
            
            # 取消进度
            if session_id and self._progress_mgr:
                try:
                    self._progress_mgr.cancel_training(session_id)
                except Exception:
                    pass
            
            # 更新数据库状态
            if self._execution_repo and session_id:
                execution = self._execution_repo.get_by_session_id(session_id, tenant_id)
                if execution:
                    self._execution_repo.update_status(
                        execution.get('execution_id'),
                        tenant_id,
                        'cancelled',
                        result={'rollback_event': rollback_event}
                    )
            
            if self._pipeline_repo:
                self._pipeline_repo.update_status(pipeline_id, tenant_id, 'rolled_back')
            
            return {
                'success': True,
                'message': 'Pipeline rolled back',
                'pipeline_id': pipeline_id,
                'event': rollback_event
            }
            
        except Exception as e:
            self.logger.error(f"Failed to rollback pipeline: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_execution_status(self, execution_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """获取执行状态
        
        Args:
            execution_id: 执行ID
            tenant_id: 租户ID
            
        Returns:
            执行状态信息
        """
        try:
            if self._execution_repo:
                execution = self._execution_repo.get_by_id(execution_id, tenant_id)
                if execution and self._step_repo:
                    # 获取步骤详情
                    steps = self._step_repo.list_by_execution(execution_id, tenant_id)
                    execution['steps'] = steps
                return execution
            return None
        except Exception as e:
            self.logger.error(f"Failed to get execution status: {e}")
            return None
    
    def list_executions(self, tenant_id: str, pipeline_id: Optional[str] = None,
                       status: Optional[str] = None,
                       user_id: Optional[str] = None,
                       limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """获取执行记录列表
        
        Args:
            tenant_id: 租户ID
            pipeline_id: 流水线ID过滤
            status: 状态过滤
            user_id: 用户ID过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            执行记录列表和总数
        """
        try:
            if self._execution_repo:
                if pipeline_id:
                    executions, total = self._execution_repo.list_by_pipeline(
                        pipeline_id=pipeline_id,
                        tenant_id=tenant_id,
                        status=status,
                        limit=limit,
                        offset=offset
                    )
                else:
                    executions, total = self._execution_repo.list_by_tenant(
                        tenant_id=tenant_id,
                        status=status,
                        user_id=user_id,
                        limit=limit,
                        offset=offset
                    )
                return {'executions': executions, 'total': total, 'limit': limit, 'offset': offset}
            return {'executions': [], 'total': 0, 'limit': limit, 'offset': offset}
        except Exception as e:
            self.logger.error(f"Failed to list executions: {e}")
            return {'executions': [], 'total': 0, 'limit': limit, 'offset': offset}
    
    # ==========================================================================
    # 模板管理
    # ==========================================================================
    
    def create_template(self, name: str, steps_template: List[Dict],
                       tenant_id: str, user_id: str,
                       description: Optional[str] = None,
                       category: Optional[str] = None,
                       default_config: Optional[Dict] = None,
                       required_params: Optional[List[str]] = None,
                       tags: Optional[List[str]] = None,
                       is_public: bool = False) -> Dict[str, Any]:
        """创建流水线模板
        
        Args:
            name: 模板名称
            steps_template: 步骤模板配置
            tenant_id: 租户ID
            user_id: 用户ID
            description: 描述
            category: 分类
            default_config: 默认配置
            required_params: 必需参数
            tags: 标签
            is_public: 是否公开
            
        Returns:
            创建的模板信息
        """
        try:
            template_data = {
                'tenant_id': tenant_id,
                'name': name,
                'description': description,
                'user_id': user_id,
                'template_type': 'custom',
                'category': category,
                'steps_template': steps_template,
                'default_config': default_config,
                'required_params': required_params,
                'tags': tags,
                'is_public': is_public
            }
            
            if self._template_repo:
                return self._template_repo.create(template_data)
            
            template_data['template_id'] = f"tmpl_{uuid.uuid4().hex[:8]}"
            return template_data
            
        except Exception as e:
            self.logger.error(f"Failed to create template: {e}")
            raise
    
    def get_template(self, template_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """获取模板详情"""
        try:
            if self._template_repo:
                return self._template_repo.get_by_id(template_id, tenant_id)
            return None
        except Exception as e:
            self.logger.error(f"Failed to get template: {e}")
            return None
    
    def list_templates(self, tenant_id: str, category: Optional[str] = None,
                      include_system: bool = True,
                      limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """获取模板列表"""
        try:
            if self._template_repo:
                templates, total = self._template_repo.list_templates(
                    tenant_id=tenant_id,
                    category=category,
                    include_system=include_system,
                    limit=limit,
                    offset=offset
                )
                return {'templates': templates, 'total': total, 'limit': limit, 'offset': offset}
            return {'templates': [], 'total': 0, 'limit': limit, 'offset': offset}
        except Exception as e:
            self.logger.error(f"Failed to list templates: {e}")
            return {'templates': [], 'total': 0, 'limit': limit, 'offset': offset}
    
    def create_pipeline_from_template(self, template_id: str, name: str,
                                     tenant_id: str, user_id: str,
                                     params: Optional[Dict] = None,
                                     **kwargs) -> Dict[str, Any]:
        """从模板创建流水线
        
        Args:
            template_id: 模板ID
            name: 流水线名称
            tenant_id: 租户ID
            user_id: 用户ID
            params: 模板参数
            **kwargs: 其他流水线参数
            
        Returns:
            创建的流水线信息
        """
        try:
            # 获取模板
            template = self.get_template(template_id, tenant_id)
            if not template:
                return {'success': False, 'error': 'Template not found'}
            
            # 合并配置
            steps_config = template.get('steps_template', [])
            default_config = template.get('default_config') or {}
            global_config = {**default_config, **(params or {})}
            
            # 创建流水线
            result = self.create_pipeline(
                name=name,
                steps_config=steps_config,
                tenant_id=tenant_id,
                user_id=user_id,
                global_config=global_config,
                **kwargs
            )
            
            # 增加模板使用次数
            if self._template_repo:
                self._template_repo.increment_usage(template_id, tenant_id)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to create pipeline from template: {e}")
            return {'success': False, 'error': str(e)}
    
    # ==========================================================================
    # 辅助方法
    # ==========================================================================
    
    def _validate_steps_config(self, steps_config: List[Dict]) -> List[Dict]:
        """验证和规范化步骤配置
        
        Args:
            steps_config: 步骤配置列表
            
        Returns:
            验证后的步骤配置
        """
        validated = []
        valid_types = ['pretrain', 'finetune', 'sft', 'preference_optim', 
                      'evaluation', 'validation', 'data_processing', 
                      'model_export', 'deployment', 'checkpoint', 'custom']
        valid_policies = ['continue', 'stop', 'rollback', 'retry']
        
        for idx, step in enumerate(steps_config):
            validated_step = {
                'name': step.get('name', f'step_{idx}'),
                'type': step.get('type', 'custom'),
                'on_fail': step.get('on_fail', 'rollback'),
                'params': step.get('params', {})
            }
            
            # 验证类型
            if validated_step['type'] not in valid_types:
                validated_step['type'] = 'custom'
            
            # 验证失败策略
            if validated_step['on_fail'] not in valid_policies:
                validated_step['on_fail'] = 'rollback'
            
            validated.append(validated_step)
        
        return validated


# ==============================================================================
# 获取服务实例的辅助函数
# ==============================================================================

_pipeline_service: Optional[PipelineService] = None


def get_pipeline_service(use_memory_storage: bool = False) -> PipelineService:
    """获取流水线服务实例
    
    Args:
        use_memory_storage: 是否使用内存存储
        
    Returns:
        PipelineService实例
    """
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = PipelineService(use_memory_storage=use_memory_storage)
    return _pipeline_service

