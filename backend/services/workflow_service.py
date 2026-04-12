"""工作流服务

实现工作流相关的业务逻辑，支持数据库持久化和租户隔离。
"""

import sys
import os
import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

try:
    from backend.core.exceptions import ValidationError, BusinessLogicError
except ImportError:
    from backend.core.exceptions import ValidationError, BusinessLogicError

logger = logging.getLogger(__name__)


class WorkflowService:
    """工作流服务
    
    支持两种模式：
    1. 内存模式 (use_memory_storage=True): 数据存储在内存中，适用于测试
    2. 数据库模式 (use_memory_storage=False): 数据持久化到数据库
    
    所有操作都基于租户维度进行隔离。
    """
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化工作流服务
        
        Args:
            use_memory_storage: 是否使用内存存储
        """
        self._use_memory_storage = use_memory_storage
        self._init_repositories()
    
    def _init_repositories(self):
        """初始化数据访问层"""
        try:
            from backend.repositories.workflow_repository import (
                WorkflowRepository,
                WorkflowExecutionRepository,
                WorkflowStepRepository,
                WorkflowTemplateRepository,
                WorkflowLogRepository
            )
            
            self.workflow_repo = WorkflowRepository(use_memory_storage=self._use_memory_storage)
            self.execution_repo = WorkflowExecutionRepository(use_memory_storage=self._use_memory_storage)
            self.step_repo = WorkflowStepRepository(use_memory_storage=self._use_memory_storage)
            self.template_repo = WorkflowTemplateRepository(use_memory_storage=self._use_memory_storage)
            self.log_repo = WorkflowLogRepository(use_memory_storage=self._use_memory_storage)
            
            logger.info(f"WorkflowService repositories initialized (memory_storage={self._use_memory_storage})")
            
        except ImportError as e:
            logger.error(f"Failed to import workflow repositories: {e}")
            raise
    
    # ========== 工作流管理 ==========
    
    def create_workflow(self, tenant_id: str, user_id: str, 
                       workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建工作流
        
        Args:
            tenant_id: 租户ID
            user_id: 创建者用户ID
            workflow_data: 工作流数据
        
        Returns:
            创建的工作流信息
        """
        try:
            # 验证必填字段
            if not workflow_data.get('name'):
                raise ValidationError("Workflow name is required")
            if not workflow_data.get('workflow_type'):
                raise ValidationError("Workflow type is required")
            
            # 准备数据
            workflow_data['tenant_id'] = tenant_id
            workflow_data['created_by'] = user_id
            workflow_data['updated_by'] = user_id
            
            # 如果使用模板创建，合并模板配置
            template_id = workflow_data.get('template_id')
            if template_id:
                template = self.template_repo.get_by_id(template_id)
                if template:
                    template_dict = self._to_dict(template)
                    # 合并配置
                    if not workflow_data.get('config'):
                        workflow_data['config'] = template_dict.get('config', {})
                    if not workflow_data.get('steps_config'):
                        workflow_data['steps_config'] = template_dict.get('steps_config', [])
                    # 增加模板使用计数
                    self.template_repo.increment_use_count(template_id)
            
            # 创建工作流
            workflow = self.workflow_repo.create(workflow_data)
            
            logger.info(f"Created workflow: {workflow_data.get('name')} for tenant {tenant_id}")
            return self._workflow_to_dict(workflow)
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            raise BusinessLogicError(f"Failed to create workflow: {e}")
    
    def get_workflow(self, workflow_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """获取工作流详情
        
        Args:
            workflow_id: 工作流ID
            tenant_id: 租户ID
        
        Returns:
            工作流信息
        """
        try:
            workflow = self.workflow_repo.get_by_id(workflow_id)
            if not workflow:
                return None
            
            # 租户隔离检查
            wf_tenant = workflow.get('tenant_id') if isinstance(workflow, dict) else workflow.tenant_id
            if wf_tenant != tenant_id:
                return None
            
            return self._workflow_to_dict(workflow)
            
        except Exception as e:
            logger.error(f"Failed to get workflow: {e}")
            return None
    
    def list_workflows(self, tenant_id: str, workflow_type: Optional[str] = None,
                      status: Optional[str] = None, search: Optional[str] = None,
                      page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """获取工作流列表
        
        Args:
            tenant_id: 租户ID
            workflow_type: 工作流类型过滤
            status: 状态过滤
            search: 搜索关键字
            page: 页码
            page_size: 每页数量
        
        Returns:
            包含工作流列表和分页信息的字典
        """
        try:
            offset = (page - 1) * page_size
            workflows, total = self.workflow_repo.get_by_tenant(
                tenant_id, workflow_type, status, search, page_size, offset
            )
            
            return {
                'items': [self._workflow_to_dict(w) for w in workflows],
                'total': total,
                'page': page,
                'page_size': page_size,
                'total_pages': (total + page_size - 1) // page_size
            }
            
        except Exception as e:
            logger.error(f"Failed to list workflows: {e}")
            return {'items': [], 'total': 0, 'page': page, 'page_size': page_size, 'total_pages': 0}
    
    def update_workflow(self, workflow_id: str, tenant_id: str, user_id: str,
                       update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新工作流
        
        Args:
            workflow_id: 工作流ID
            tenant_id: 租户ID
            user_id: 操作用户ID
            update_data: 更新数据
        
        Returns:
            更新后的工作流信息
        """
        try:
            # 检查工作流是否存在且属于租户
            workflow = self.get_workflow(workflow_id, tenant_id)
            if not workflow:
                return None
            
            # 更新修改者
            update_data['updated_by'] = user_id
            
            # 版本递增
            if 'version' not in update_data:
                update_data['version'] = workflow.get('version', 1) + 1
            
            updated_workflow = self.workflow_repo.update(workflow_id, update_data)
            if not updated_workflow:
                return None
            
            logger.info(f"Updated workflow: {workflow_id}")
            return self._workflow_to_dict(updated_workflow)
            
        except Exception as e:
            logger.error(f"Failed to update workflow: {e}")
            raise BusinessLogicError(f"Failed to update workflow: {e}")
    
    def delete_workflow(self, workflow_id: str, tenant_id: str) -> bool:
        """删除工作流
        
        Args:
            workflow_id: 工作流ID
            tenant_id: 租户ID
        
        Returns:
            是否删除成功
        """
        try:
            # 检查工作流是否存在且属于租户
            workflow = self.get_workflow(workflow_id, tenant_id)
            if not workflow:
                return False
            
            # 检查是否有运行中的执行
            running = self.execution_repo.get_running_executions(tenant_id)
            for exe in running:
                exe_wf_id = exe.get('workflow_id') if isinstance(exe, dict) else exe.workflow_id
                if exe_wf_id == workflow_id:
                    raise BusinessLogicError("Cannot delete workflow with running executions")
            
            result = self.workflow_repo.delete(workflow_id)
            if result:
                logger.info(f"Deleted workflow: {workflow_id}")
            return result
            
        except BusinessLogicError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete workflow: {e}")
            return False
    
    # ========== 工作流执行 ==========
    
    def execute_workflow(self, workflow_id: str, tenant_id: str, user_id: str,
                        input_data: Optional[Dict] = None,
                        trigger_type: str = 'manual') -> Dict[str, Any]:
        """执行工作流
        
        Args:
            workflow_id: 工作流ID
            tenant_id: 租户ID
            user_id: 触发者用户ID
            input_data: 输入数据
            trigger_type: 触发类型
        
        Returns:
            执行记录信息
        """
        try:
            # 获取工作流
            workflow = self.get_workflow(workflow_id, tenant_id)
            if not workflow:
                raise ValidationError("Workflow not found")
            
            # 检查工作流状态
            if workflow.get('status') not in ('active', 'draft'):
                raise ValidationError(f"Workflow is not active: {workflow.get('status')}")
            
            # 解析步骤配置
            steps_config = workflow.get('steps_config', [])
            if isinstance(steps_config, str):
                steps_config = json.loads(steps_config)
            
            # 创建执行记录
            execution_data = {
                'tenant_id': tenant_id,
                'workflow_id': workflow_id,
                'workflow_name': workflow.get('name'),
                'workflow_version': workflow.get('version'),
                'status': 'pending',
                'triggered_by': user_id,
                'trigger_type': trigger_type,
                'input_data': input_data or {},
                'total_steps': len(steps_config),
                'started_at': datetime.utcnow()
            }
            
            execution = self.execution_repo.create(execution_data)
            execution_id = execution.get('id') if isinstance(execution, dict) else execution.id
            
            # 创建步骤记录
            for idx, step_config in enumerate(steps_config):
                step_data = {
                    'execution_id': execution_id,
                    'workflow_id': workflow_id,
                    'step_name': step_config.get('name', f'step_{idx}'),
                    'step_type': step_config.get('type'),
                    'step_index': idx,
                    'config': step_config
                }
                self.step_repo.create(step_data)
            
            # 更新执行状态为运行中
            self.execution_repo.update_status(execution_id, 'running')
            
            # 增加工作流执行计数
            self.workflow_repo.increment_execution_count(workflow_id, success=False)
            
            # 记录日志
            self._log(execution_id, workflow_id, 'info', f'Workflow execution started: {workflow.get("name")}')
            
            logger.info(f"Started workflow execution: {execution_id} for workflow {workflow_id}")
            return self._execution_to_dict(self.execution_repo.get_by_id(execution_id))
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to execute workflow: {e}")
            raise BusinessLogicError(f"Failed to execute workflow: {e}")
    
    def get_execution(self, execution_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """获取执行记录详情
        
        Args:
            execution_id: 执行ID
            tenant_id: 租户ID
        
        Returns:
            执行记录信息
        """
        try:
            execution = self.execution_repo.get_by_id(execution_id)
            if not execution:
                return None
            
            # 租户隔离检查
            exe_tenant = execution.get('tenant_id') if isinstance(execution, dict) else execution.tenant_id
            if exe_tenant != tenant_id:
                return None
            
            result = self._execution_to_dict(execution)
            
            # 获取步骤信息
            steps = self.step_repo.get_by_execution(execution_id)
            result['steps'] = [self._step_to_dict(s) for s in steps]
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get execution: {e}")
            return None
    
    def list_executions(self, tenant_id: str, workflow_id: Optional[str] = None,
                       status: Optional[str] = None, triggered_by: Optional[str] = None,
                       page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """获取执行记录列表
        
        Args:
            tenant_id: 租户ID
            workflow_id: 工作流ID过滤
            status: 状态过滤
            triggered_by: 触发者过滤
            page: 页码
            page_size: 每页数量
        
        Returns:
            包含执行记录列表和分页信息的字典
        """
        try:
            offset = (page - 1) * page_size
            executions, total = self.execution_repo.get_by_tenant(
                tenant_id, workflow_id, status, triggered_by, page_size, offset
            )
            
            return {
                'items': [self._execution_to_dict(e) for e in executions],
                'total': total,
                'page': page,
                'page_size': page_size,
                'total_pages': (total + page_size - 1) // page_size
            }
            
        except Exception as e:
            logger.error(f"Failed to list executions: {e}")
            return {'items': [], 'total': 0, 'page': page, 'page_size': page_size, 'total_pages': 0}
    
    def cancel_execution(self, execution_id: str, tenant_id: str, user_id: str,
                        reason: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """取消执行
        
        Args:
            execution_id: 执行ID
            tenant_id: 租户ID
            user_id: 操作用户ID
            reason: 取消原因
        
        Returns:
            更新后的执行记录
        """
        try:
            execution = self.get_execution(execution_id, tenant_id)
            if not execution:
                return None
            
            # 检查状态是否可取消
            if execution.get('status') in ('completed', 'failed', 'cancelled', 'timeout'):
                raise ValidationError("Execution already finished")
            
            # 更新执行状态
            update_data = {
                'status': 'cancelled',
                'cancelled_by': user_id,
                'cancel_reason': reason or 'User cancelled',
                'completed_at': datetime.utcnow()
            }
            
            updated = self.execution_repo.update(execution_id, update_data)
            
            # 记录日志
            self._log(execution_id, execution.get('workflow_id'), 'warning', 
                     f'Execution cancelled by user: {reason or "No reason provided"}')
            
            logger.info(f"Cancelled execution: {execution_id}")
            return self._execution_to_dict(updated)
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to cancel execution: {e}")
            raise BusinessLogicError(f"Failed to cancel execution: {e}")
    
    def update_execution_progress(self, execution_id: str, progress: float,
                                  current_step: Optional[str] = None,
                                  current_step_index: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """更新执行进度（内部使用）"""
        try:
            updated = self.execution_repo.update_progress(execution_id, progress, current_step, current_step_index)
            return self._execution_to_dict(updated) if updated else None
        except Exception as e:
            logger.error(f"Failed to update execution progress: {e}")
            return None
    
    def complete_execution(self, execution_id: str, success: bool,
                          output_data: Optional[Dict] = None,
                          error_message: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """完成执行（内部使用）"""
        try:
            status = 'completed' if success else 'failed'
            updated = self.execution_repo.update_status(execution_id, status, error_message, output_data)
            
            if updated:
                # 更新工作流统计
                wf_id = updated.get('workflow_id') if isinstance(updated, dict) else updated.workflow_id
                self.workflow_repo.increment_execution_count(wf_id, success)
                
                # 记录日志
                log_level = 'info' if success else 'error'
                log_msg = f'Execution {status}' + (f': {error_message}' if error_message else '')
                self._log(execution_id, wf_id, log_level, log_msg)
            
            return self._execution_to_dict(updated) if updated else None
        except Exception as e:
            logger.error(f"Failed to complete execution: {e}")
            return None
    
    # ========== 执行日志 ==========
    
    def get_execution_logs(self, execution_id: str, tenant_id: str,
                          level: Optional[str] = None,
                          page: int = 1, page_size: int = 100) -> Dict[str, Any]:
        """获取执行日志
        
        Args:
            execution_id: 执行ID
            tenant_id: 租户ID
            level: 日志级别过滤
            page: 页码
            page_size: 每页数量
        
        Returns:
            日志列表和分页信息
        """
        try:
            # 验证执行记录属于租户
            execution = self.get_execution(execution_id, tenant_id)
            if not execution:
                return {'items': [], 'total': 0, 'page': page, 'page_size': page_size, 'total_pages': 0}
            
            offset = (page - 1) * page_size
            logs, total = self.log_repo.get_by_execution(execution_id, level, page_size, offset)
            
            return {
                'items': [self._log_to_dict(l) for l in logs],
                'total': total,
                'page': page,
                'page_size': page_size,
                'total_pages': (total + page_size - 1) // page_size
            }
            
        except Exception as e:
            logger.error(f"Failed to get execution logs: {e}")
            return {'items': [], 'total': 0, 'page': page, 'page_size': page_size, 'total_pages': 0}
    
    def _log(self, execution_id: str, workflow_id: str, level: str, message: str,
            step_id: Optional[str] = None, details: Optional[Dict] = None):
        """记录日志（内部使用）"""
        try:
            self.log_repo.create({
                'execution_id': execution_id,
                'workflow_id': workflow_id,
                'step_id': step_id,
                'level': level,
                'message': message,
                'details': details,
                'source': 'workflow_service'
            })
        except Exception as e:
            logger.error(f"Failed to create workflow log: {e}")
    
    # ========== 模板管理 ==========
    
    def list_templates(self, tenant_id: str, workflow_type: Optional[str] = None,
                      category: Optional[str] = None,
                      page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """获取可用模板列表
        
        Args:
            tenant_id: 租户ID
            workflow_type: 工作流类型过滤
            category: 分类过滤
            page: 页码
            page_size: 每页数量
        
        Returns:
            模板列表和分页信息
        """
        try:
            offset = (page - 1) * page_size
            templates, total = self.template_repo.get_available_templates(
                tenant_id, workflow_type, category, page_size, offset
            )
            
            return {
                'items': [self._template_to_dict(t) for t in templates],
                'total': total,
                'page': page,
                'page_size': page_size,
                'total_pages': (total + page_size - 1) // page_size
            }
            
        except Exception as e:
            logger.error(f"Failed to list templates: {e}")
            return {'items': [], 'total': 0, 'page': page, 'page_size': page_size, 'total_pages': 0}
    
    def get_template(self, template_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """获取模板详情
        
        Args:
            template_id: 模板ID
            tenant_id: 租户ID
        
        Returns:
            模板信息
        """
        try:
            template = self.template_repo.get_by_id(template_id)
            if not template:
                return None
            
            # 检查访问权限：公开模板或租户自己的模板
            tpl_tenant = template.get('tenant_id') if isinstance(template, dict) else template.tenant_id
            is_public = template.get('is_public') if isinstance(template, dict) else template.is_public
            
            if not is_public and tpl_tenant != tenant_id:
                return None
            
            return self._template_to_dict(template)
            
        except Exception as e:
            logger.error(f"Failed to get template: {e}")
            return None
    
    def create_template(self, tenant_id: str, user_id: str,
                       template_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建模板
        
        Args:
            tenant_id: 租户ID
            user_id: 创建者用户ID
            template_data: 模板数据
        
        Returns:
            创建的模板信息
        """
        try:
            if not template_data.get('name'):
                raise ValidationError("Template name is required")
            if not template_data.get('workflow_type'):
                raise ValidationError("Workflow type is required")
            
            template_data['tenant_id'] = tenant_id
            template_data['created_by'] = user_id
            
            template = self.template_repo.create(template_data)
            
            logger.info(f"Created template: {template_data.get('name')} for tenant {tenant_id}")
            return self._template_to_dict(template)
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to create template: {e}")
            raise BusinessLogicError(f"Failed to create template: {e}")
    
    def create_workflow_from_template(self, template_id: str, tenant_id: str, user_id: str,
                                     workflow_name: str, workflow_description: Optional[str] = None,
                                     override_config: Optional[Dict] = None) -> Dict[str, Any]:
        """从模板创建工作流
        
        Args:
            template_id: 模板ID
            tenant_id: 租户ID
            user_id: 创建者用户ID
            workflow_name: 工作流名称
            workflow_description: 工作流描述
            override_config: 覆盖配置
        
        Returns:
            创建的工作流信息
        """
        try:
            template = self.get_template(template_id, tenant_id)
            if not template:
                raise ValidationError("Template not found or not accessible")
            
            # 准备工作流数据
            workflow_data = {
                'name': workflow_name,
                'description': workflow_description or template.get('description'),
                'workflow_type': template.get('workflow_type'),
                'template_id': template_id,
                'config': template.get('config', {}),
                'steps_config': template.get('steps_config', []),
                'category': template.get('category')
            }
            
            # 合并默认参数
            default_params = template.get('default_params', {})
            if isinstance(default_params, str):
                default_params = json.loads(default_params)
            
            config = workflow_data.get('config', {})
            if isinstance(config, str):
                config = json.loads(config)
            
            if default_params:
                config = {**default_params, **config}
            
            # 应用覆盖配置
            if override_config:
                config = {**config, **override_config}
            
            workflow_data['config'] = config
            
            return self.create_workflow(tenant_id, user_id, workflow_data)
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to create workflow from template: {e}")
            raise BusinessLogicError(f"Failed to create workflow from template: {e}")
    
    # ========== 统计信息 ==========
    
    def get_workflow_statistics(self, tenant_id: str) -> Dict[str, Any]:
        """获取工作流统计信息
        
        Args:
            tenant_id: 租户ID
        
        Returns:
            统计信息
        """
        try:
            # 获取工作流统计
            workflow_stats = self.workflow_repo.get_statistics(tenant_id)
            
            # 获取执行统计
            execution_stats = self.execution_repo.get_execution_statistics(tenant_id)
            
            # 获取模板数量
            templates, template_count = self.template_repo.get_available_templates(tenant_id, limit=1)
            
            return {
                'workflow_stats': workflow_stats,
                'execution_stats': execution_stats,
                'templates_available': template_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get workflow statistics: {e}")
            return {}
    
    def get_running_executions(self, tenant_id: str) -> List[Dict[str, Any]]:
        """获取正在运行的执行记录
        
        Args:
            tenant_id: 租户ID
        
        Returns:
            运行中的执行记录列表
        """
        try:
            executions = self.execution_repo.get_running_executions(tenant_id)
            return [self._execution_to_dict(e) for e in executions]
        except Exception as e:
            logger.error(f"Failed to get running executions: {e}")
            return []
    
    # ========== 辅助方法 ==========
    
    def _to_dict(self, obj) -> Optional[Dict]:
        """通用转换方法"""
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj
        return {c.name: getattr(obj, c.name) for c in obj.__table__.columns} if hasattr(obj, '__table__') else vars(obj)
    
    def _workflow_to_dict(self, workflow) -> Optional[Dict]:
        """将工作流转换为字典"""
        if workflow is None:
            return None
        if isinstance(workflow, dict):
            # 处理JSON字段
            for field in ('config', 'steps_config', 'trigger_config', 'notification_config', 'tags'):
                if field in workflow and isinstance(workflow.get(field), str):
                    try:
                        workflow[field] = json.loads(workflow[field])
                    except:
                        pass
            return workflow
        
        result = {
            'id': str(getattr(workflow, 'id', '')),
            'tenant_id': getattr(workflow, 'tenant_id', None),
            'name': getattr(workflow, 'name', None),
            'description': getattr(workflow, 'description', None),
            'workflow_type': getattr(workflow, 'workflow_type', None),
            'status': getattr(workflow, 'status', 'draft'),
            'version': getattr(workflow, 'version', 1),
            'created_by': getattr(workflow, 'created_by', None),
            'updated_by': getattr(workflow, 'updated_by', None),
            'execution_count': getattr(workflow, 'execution_count', 0),
            'success_count': getattr(workflow, 'success_count', 0),
            'failure_count': getattr(workflow, 'failure_count', 0),
            'success_rate': self._calc_success_rate(
                getattr(workflow, 'success_count', 0),
                getattr(workflow, 'execution_count', 0)
            ),
            'avg_duration_seconds': getattr(workflow, 'avg_duration_seconds', 0),
            'schedule_enabled': getattr(workflow, 'schedule_enabled', False),
            'schedule_cron': getattr(workflow, 'schedule_cron', None),
            'timeout_seconds': getattr(workflow, 'timeout_seconds', 3600),
            'max_retries': getattr(workflow, 'max_retries', 3),
            'template_id': getattr(workflow, 'template_id', None),
            'is_template': getattr(workflow, 'is_template', False),
            'category': getattr(workflow, 'category', None),
            'created_at': self._format_datetime(getattr(workflow, 'created_at', None)),
            'updated_at': self._format_datetime(getattr(workflow, 'updated_at', None)),
            'last_run_at': self._format_datetime(getattr(workflow, 'last_run_at', None))
        }
        
        # 处理JSON字段
        for field in ('config', 'steps_config', 'trigger_config', 'notification_config', 'tags'):
            value = getattr(workflow, field, None)
            if value:
                if isinstance(value, str):
                    try:
                        result[field] = json.loads(value)
                    except:
                        result[field] = value
                else:
                    result[field] = value
            else:
                result[field] = {} if field != 'tags' and field != 'steps_config' else []
        
        return result
    
    def _execution_to_dict(self, execution) -> Optional[Dict]:
        """将执行记录转换为字典"""
        if execution is None:
            return None
        if isinstance(execution, dict):
            # 处理JSON字段
            for field in ('input_data', 'output_data', 'context_data', 'error_details'):
                if field in execution and isinstance(execution.get(field), str):
                    try:
                        execution[field] = json.loads(execution[field])
                    except:
                        pass
            return execution
        
        result = {
            'id': str(getattr(execution, 'id', '')),
            'tenant_id': getattr(execution, 'tenant_id', None),
            'workflow_id': getattr(execution, 'workflow_id', None),
            'workflow_name': getattr(execution, 'workflow_name', None),
            'workflow_version': getattr(execution, 'workflow_version', None),
            'status': getattr(execution, 'status', 'pending'),
            'progress': getattr(execution, 'progress', 0),
            'current_step': getattr(execution, 'current_step', None),
            'current_step_index': getattr(execution, 'current_step_index', 0),
            'total_steps': getattr(execution, 'total_steps', 0),
            'triggered_by': getattr(execution, 'triggered_by', None),
            'trigger_type': getattr(execution, 'trigger_type', 'manual'),
            'duration_seconds': getattr(execution, 'duration_seconds', None),
            'error_message': getattr(execution, 'error_message', None),
            'error_step': getattr(execution, 'error_step', None),
            'retry_count': getattr(execution, 'retry_count', 0),
            'priority': getattr(execution, 'priority', 5),
            'cancelled_by': getattr(execution, 'cancelled_by', None),
            'cancel_reason': getattr(execution, 'cancel_reason', None),
            'started_at': self._format_datetime(getattr(execution, 'started_at', None)),
            'completed_at': self._format_datetime(getattr(execution, 'completed_at', None)),
            'created_at': self._format_datetime(getattr(execution, 'created_at', None)),
            'updated_at': self._format_datetime(getattr(execution, 'updated_at', None))
        }
        
        # 处理JSON字段
        for field in ('input_data', 'output_data', 'context_data', 'error_details'):
            value = getattr(execution, field, None)
            if value:
                if isinstance(value, str):
                    try:
                        result[field] = json.loads(value)
                    except:
                        result[field] = value
                else:
                    result[field] = value
            else:
                result[field] = {}
        
        return result
    
    def _step_to_dict(self, step) -> Optional[Dict]:
        """将步骤转换为字典"""
        if step is None:
            return None
        if isinstance(step, dict):
            for field in ('input_data', 'output_data', 'config', 'error_details'):
                if field in step and isinstance(step.get(field), str):
                    try:
                        step[field] = json.loads(step[field])
                    except:
                        pass
            return step
        
        result = {
            'id': str(getattr(step, 'id', '')),
            'execution_id': getattr(step, 'execution_id', None),
            'workflow_id': getattr(step, 'workflow_id', None),
            'step_name': getattr(step, 'step_name', None),
            'step_type': getattr(step, 'step_type', None),
            'step_index': getattr(step, 'step_index', 0),
            'status': getattr(step, 'status', 'pending'),
            'progress': getattr(step, 'progress', 0),
            'duration_seconds': getattr(step, 'duration_seconds', None),
            'error_message': getattr(step, 'error_message', None),
            'retry_count': getattr(step, 'retry_count', 0),
            'started_at': self._format_datetime(getattr(step, 'started_at', None)),
            'completed_at': self._format_datetime(getattr(step, 'completed_at', None))
        }
        
        for field in ('input_data', 'output_data', 'config', 'error_details'):
            value = getattr(step, field, None)
            if value:
                if isinstance(value, str):
                    try:
                        result[field] = json.loads(value)
                    except:
                        result[field] = value
                else:
                    result[field] = value
            else:
                result[field] = {}
        
        return result
    
    def _template_to_dict(self, template) -> Optional[Dict]:
        """将模板转换为字典"""
        if template is None:
            return None
        if isinstance(template, dict):
            for field in ('config', 'steps_config', 'default_params', 'tags'):
                if field in template and isinstance(template.get(field), str):
                    try:
                        template[field] = json.loads(template[field])
                    except:
                        pass
            return template
        
        result = {
            'id': str(getattr(template, 'id', '')),
            'tenant_id': getattr(template, 'tenant_id', None),
            'name': getattr(template, 'name', None),
            'description': getattr(template, 'description', None),
            'workflow_type': getattr(template, 'workflow_type', None),
            'created_by': getattr(template, 'created_by', None),
            'is_public': getattr(template, 'is_public', False),
            'is_system': getattr(template, 'is_system', False),
            'use_count': getattr(template, 'use_count', 0),
            'version': getattr(template, 'version', '1.0.0'),
            'category': getattr(template, 'category', None),
            'icon': getattr(template, 'icon', None),
            'thumbnail': getattr(template, 'thumbnail', None),
            'created_at': self._format_datetime(getattr(template, 'created_at', None)),
            'updated_at': self._format_datetime(getattr(template, 'updated_at', None))
        }
        
        for field in ('config', 'steps_config', 'default_params', 'tags'):
            value = getattr(template, field, None)
            if value:
                if isinstance(value, str):
                    try:
                        result[field] = json.loads(value)
                    except:
                        result[field] = value
                else:
                    result[field] = value
            else:
                result[field] = {} if field not in ('steps_config', 'tags') else []
        
        return result
    
    def _log_to_dict(self, log) -> Optional[Dict]:
        """将日志转换为字典"""
        if log is None:
            return None
        if isinstance(log, dict):
            if 'details' in log and isinstance(log.get('details'), str):
                try:
                    log['details'] = json.loads(log['details'])
                except:
                    pass
            return log
        
        result = {
            'id': str(getattr(log, 'id', '')),
            'execution_id': getattr(log, 'execution_id', None),
            'workflow_id': getattr(log, 'workflow_id', None),
            'step_id': getattr(log, 'step_id', None),
            'level': getattr(log, 'level', 'info'),
            'message': getattr(log, 'message', None),
            'source': getattr(log, 'source', None),
            'timestamp': self._format_datetime(getattr(log, 'timestamp', None))
        }
        
        details = getattr(log, 'details', None)
        if details:
            if isinstance(details, str):
                try:
                    result['details'] = json.loads(details)
                except:
                    result['details'] = details
            else:
                result['details'] = details
        else:
            result['details'] = {}
        
        return result
    
    def _format_datetime(self, dt) -> Optional[str]:
        """格式化日期时间"""
        if dt is None:
            return None
        if isinstance(dt, str):
            return dt
        return dt.isoformat() if hasattr(dt, 'isoformat') else str(dt)
    
    def _calc_success_rate(self, success: int, total: int) -> float:
        """计算成功率"""
        if total == 0:
            return 0.0
        return round(success / total * 100, 2)


# ============================================================================
# 全局实例和工厂函数
# ============================================================================

_global_workflow_service: Optional[WorkflowService] = None


def get_workflow_service(use_memory_storage: bool = False) -> WorkflowService:
    """获取全局工作流服务实例
    
    Args:
        use_memory_storage: 是否使用内存存储（仅在首次创建时生效）
    
    Returns:
        WorkflowService 实例
    """
    global _global_workflow_service
    if _global_workflow_service is None:
        _global_workflow_service = WorkflowService(use_memory_storage=use_memory_storage)
    return _global_workflow_service


def reset_workflow_service():
    """重置全局工作流服务实例（主要用于测试）"""
    global _global_workflow_service
    _global_workflow_service = None

