# -*- coding: utf-8 -*-
"""训练历史服务

提供训练历史记录管理的核心业务逻辑。

架构调用关系：
API层 (training_history_api.py) 
    -> Service层 (本模块)
        -> Launcher层 (training_launcher.py)
        -> Repository层 (training_history_repository.py)
        -> Training模块 (backend/modules/training)
"""

import logging
import os
import uuid
from typing import Optional, List, Dict, Any
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
    ValidationError, BusinessLogicError
)


class TrainingHistoryService:
    """
    训练历史服务
    
    提供训练历史记录的完整业务逻辑：
    - 获取训练历史列表（分页、过滤）
    - 获取训练详情
    - 下载训练模型
    - 重新开始训练（调用训练模块）
    - 删除训练记录
    - 获取训练统计
    
    调用关系：
    - 使用 TrainingHistoryRepository 进行数据持久化
    - 使用 TrainingLauncher 执行重新训练
    """
    
    def __init__(self, use_memory_storage: bool = False):
        """
        初始化训练历史服务
        
        Args:
            use_memory_storage: 是否使用内存存储（用于测试）
        """
        self._use_memory_storage = use_memory_storage
        self._repo = None
        self._launcher = None
        self._memory_store: Dict[str, Dict] = {}
        
        # 初始化仓库层
        self._init_repository()
        
        # 初始化训练启动器
        self._init_launcher()
    
    def _init_repository(self):
        """初始化仓库层"""
        if self._use_memory_storage:
            self._repo = None
            return
        
        try:
            from backend.repositories.training_history_repository import get_training_history_repository
            self._repo = get_training_history_repository()
            logger.info("TrainingHistoryRepository initialized")
        except ImportError as e:
            logger.warning("Failed to import TrainingHistoryRepository: %s", e)
            self._repo = None
    
    def _init_launcher(self):
        """初始化训练启动器"""
        self._launcher_class = TrainingSystemLauncher
        logger.info("TrainingSystemLauncher available")
        try:
            availability = get_module_availability()
            logger.info("Launcher module availability: %s", availability)
        except Exception as e:
            logger.warning("Failed to check launcher availability: %s", e)
    
    # ==================== 核心业务方法 ====================
    
    def get_training_history(
        self, 
        user_id: str, 
        page: int = 1, 
        limit: int = 10, 
        status: Optional[str] = None,
        training_type: Optional[str] = None,
        model_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        _tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取训练历史记录（支持分页和筛选）
        
        Args:
            user_id: 用户ID
            page: 页码 (默认: 1)
            limit: 每页数量 (默认: 10, 最大: 50)
            status: 状态过滤 (completed, failed, cancelled, running, pending)
            training_type: 训练类型过滤 (PT, SFT, DPO, standard, distributed, etc.)
            model_name: 模型名称搜索
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            tenant_id: 租户ID（可选）
            
        Returns:
            {
                'sessions': [...],
                'total': int,
                'page': int,
                'limit': int
            }
        """
        # 参数验证和规范化
        page = max(page, 1)
        limit = min(max(limit, 1), 50)
        
        try:
            if self._repo is not None:
                # 使用仓库层
                sessions, total = self._repo.get_training_sessions_paginated(
                    user_id=user_id,
                    page=page,
                    limit=limit,
                    status=status,
                    training_type=training_type,
                    model_name=model_name,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # 转换为前端格式
                session_list = [self._format_session_for_list(s) for s in sessions]
                
            else:
                # 使用内存存储（用于测试或fallback）
                session_list, total = self._get_from_memory(
                    user_id, page, limit, status, training_type, model_name, start_date, end_date
                )
            
            return {
                'sessions': session_list,
                'total': total,
                'page': page,
                'limit': limit
            }
            
        except Exception as e:
            logger.error("获取训练历史记录失败: %s", e)
            raise BusinessLogicError(f"获取训练历史记录失败: {e}", operation="get_training_history") from e
    
    def get_training_history_detail(
        self, 
        session_id: str, 
        user_id: str,
        _tenant_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        获取训练历史记录详情
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            tenant_id: 租户ID（可选）
            
        Returns:
            训练详情字典，不存在则返回 None
        """
        try:
            if self._repo is not None:
                session = self._repo.get_training_session_by_id(session_id, user_id)
                if not session:
                    return None
                return self._format_session_detail(session)
            else:
                # 内存存储
                key = f"{user_id}:{session_id}"
                if key in self._memory_store:
                    return self._memory_store[key].copy()
                return None
                
        except Exception as e:
            logger.error("获取训练记录详情失败: %s", e)
            raise BusinessLogicError(f"获取训练记录详情失败: {e}", operation="get_training_history_detail") from e
    
    def download_training_model(
        self, 
        session_id: str, 
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> Optional[str]:
        """
        获取训练模型下载URL
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            tenant_id: 租户ID（可选）
            
        Returns:
            下载URL字符串，失败返回 None
        """
        try:
            # 获取训练详情
            detail = self.get_training_history_detail(session_id, user_id, tenant_id)
            
            if not detail:
                return None
            
            # 检查状态
            if detail.get('status') != 'completed':
                raise BusinessLogicError(
                    "只有已完成的训练任务才能下载模型", 
                    operation="download_training_model"
                )
            
            # 获取模型路径
            result = detail.get('result', {})
            model_path = result.get('model_path') or result.get('output_path')
            
            if not model_path:
                raise BusinessLogicError("模型路径不存在", operation="download_training_model")
            
            # 生成下载URL
            # 实际生产环境应生成带签名的临时URL
            download_url = self._generate_download_url(session_id, model_path)
            
            logger.info("Generated download URL for session %s", session_id)
            return download_url
            
        except BusinessLogicError:
            raise
        except Exception as e:
            logger.error("获取下载链接失败: %s", e)
            raise BusinessLogicError(f"获取下载链接失败: {e}", operation="download_training_model") from e
    
    def restart_training(
        self, 
        session_id: str, 
        user_id: str,
        tenant_id: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        重新开始训练
        
        基于历史训练配置创建新的训练任务，并调用训练模块执行。
        
        Args:
            session_id: 原会话ID
            user_id: 用户ID
            tenant_id: 租户ID（可选）
            config_overrides: 配置覆盖（可选）
            
        Returns:
            新的会话ID
        """
        try:
            # 获取原训练详情
            detail = self.get_training_history_detail(session_id, user_id, tenant_id)
            
            if not detail:
                raise ValidationError("训练记录不存在", field="session_id")
            
            # 获取原配置
            original_config = detail.get('config', {})
            
            # 合并配置覆盖
            new_config = {**original_config}
            if config_overrides:
                new_config.update(config_overrides)
            
            # 生成新会话ID
            new_session_id = str(uuid.uuid4())
            
            # 创建新的训练会话记录
            new_session_data = {
                'session_id': new_session_id,
                'user_id': user_id,
                'tenant_id': tenant_id,
                'model_id': detail.get('modelName', 'unknown'),
                'dataset_id': new_config.get('dataset_id'),
                'training_type': detail.get('trainingType', 'standard'),
                'status': 'pending',
                'config': new_config,
                'created_at': datetime.utcnow().isoformat(),
                'parent_session_id': session_id  # 关联原会话
            }
            
            # 保存到数据库
            if self._repo is not None:
                self._repo.create_training_session(**new_session_data)
            else:
                # 内存存储
                key = f"{user_id}:{new_session_id}"
                self._memory_store[key] = new_session_data
            
            # 尝试启动训练任务
            self._start_training_async(new_session_id, user_id, new_config, detail.get('trainingType'))
            
            logger.info("Restarted training: %s -> %s", session_id, new_session_id)
            return new_session_id
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error("重新开始训练失败: %s", e)
            raise BusinessLogicError(f"重新开始训练失败: {e}", operation="restart_training") from e
    
    def delete_training_record(
        self, 
        session_id: str, 
        user_id: str,
        _tenant_id: Optional[str] = None
    ) -> bool:
        """
        删除训练记录
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            tenant_id: 租户ID（可选）
            
        Returns:
            是否删除成功
        """
        try:
            if self._repo is not None:
                success = self._repo.delete_training_session(session_id, user_id)
            else:
                # 内存存储
                key = f"{user_id}:{session_id}"
                if key in self._memory_store:
                    del self._memory_store[key]
                    success = True
                else:
                    success = False
            
            if success:
                logger.info("Deleted training record: %s", session_id)
                # 清理相关文件（模型、日志等）
                self._cleanup_training_files(session_id)
            
            return success
            
        except Exception as e:
            logger.error("删除训练记录失败: %s", e)
            raise BusinessLogicError(f"删除训练记录失败: {e}", operation="delete_training_record") from e
    
    def get_training_statistics(
        self, 
        user_id: str,
        _tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取训练统计信息
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID（可选）
            
        Returns:
            统计信息字典
        """
        try:
            if self._repo is not None:
                stats = self._repo.get_user_training_statistics(user_id)
                # 转换为前端格式
                return {
                    'totalTrainings': stats.get('total_trainings', 0),
                    'completedTrainings': stats.get('completed_trainings', 0),
                    'failedTrainings': stats.get('failed_trainings', 0),
                    'cancelledTrainings': stats.get('cancelled_trainings', 0),
                    'averageTrainingTime': stats.get('average_training_time', 0.0),
                    'averageAccuracy': stats.get('average_accuracy', 0.0),
                    'averageLoss': stats.get('average_loss', 0.0),
                    'mostUsedModel': stats.get('most_used_model', 'unknown'),
                    'mostUsedTrainingType': stats.get('most_used_training_type', 'unknown')
                }
            else:
                # 内存存储统计
                return self._calculate_memory_statistics(user_id)
                
        except Exception as e:
            logger.error("获取训练统计信息失败: %s", e)
            raise BusinessLogicError(f"获取训练统计信息失败: {e}", operation="get_training_statistics") from e
    
    # ==================== 扩展业务方法 ====================
    
    def get_training_logs(
        self, 
        session_id: str, 
        user_id: str,
        lines: int = 100,
        tenant_id: Optional[str] = None
    ) -> List[str]:
        """
        获取训练日志
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            lines: 返回日志行数
            tenant_id: 租户ID（可选）
            
        Returns:
            日志行列表
        """
        try:
            detail = self.get_training_history_detail(session_id, user_id, tenant_id)
            if not detail:
                return []
            
            log_path = detail.get('logPath')
            if not log_path or not os.path.exists(log_path):
                return [f"日志文件不存在: {log_path}"]
            
            # 读取日志文件的最后N行
            with open(log_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                return all_lines[-lines:]
                
        except Exception as e:
            logger.error("获取训练日志失败: %s", e)
            return [f"获取日志失败: {e}"]
    
    def compare_training_sessions(
        self,
        session_ids: List[str],
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        比较多个训练会话
        
        Args:
            session_ids: 会话ID列表
            user_id: 用户ID
            tenant_id: 租户ID（可选）
            
        Returns:
            比较结果
        """
        sessions = []
        for sid in session_ids[:5]:  # 最多比较5个
            detail = self.get_training_history_detail(sid, user_id, tenant_id)
            if detail:
                sessions.append(detail)
        
        if not sessions:
            return {'sessions': [], 'comparison': {}}
        
        # 提取可比较的指标
        comparison = {
            'loss': [s.get('finalLoss', 0) for s in sessions],
            'accuracy': [s.get('accuracy', 0) for s in sessions],
            'duration': [s.get('duration', 0) for s in sessions],
            'epochs': [s.get('epochs', 0) for s in sessions]
        }
        
        return {
            'sessions': sessions,
            'comparison': comparison
        }
    
    # ==================== 私有辅助方法 ====================
    
    def _format_session_for_list(self, session) -> Dict[str, Any]:
        """格式化会话数据用于列表显示"""
        # 计算持续时间
        duration = 0
        started_at = getattr(session, 'started_at', None)
        completed_at = getattr(session, 'completed_at', None)
        created_at = getattr(session, 'created_at', None)
        
        if started_at and completed_at:
            duration = int((completed_at - started_at).total_seconds())
        elif started_at:
            duration = int((datetime.utcnow() - started_at).total_seconds())
        
        # 从配置和结果中提取信息
        config = getattr(session, 'config', None) or {}
        result = getattr(session, 'result', None) or {}
        
        session_id = getattr(session, 'session_id', None) or getattr(session, 'id', 'unknown')
        
        return {
            'id': session_id,
            'name': config.get('session_name', f'Training Task {str(session_id)[:8]}'),
            'status': getattr(session, 'status', 'unknown'),
            'trainingType': getattr(session, 'training_type', 'unknown'),
            'modelName': getattr(session, 'model_id', 'unknown'),
            'startTime': started_at.isoformat() if started_at else (created_at.isoformat() if created_at else None),
            'endTime': completed_at.isoformat() if completed_at else None,
            'duration': duration,
            'finalLoss': result.get('final_loss', 0.0),
            'accuracy': result.get('accuracy', 0.0),
            'epochs': config.get('epochs', result.get('completed_epochs', 0)),
            'datasetSize': config.get('dataset_size', 0),
            'outputPath': result.get('output_path', ''),
            'configPath': result.get('config_path', ''),
            'logPath': result.get('log_path', '')
        }
    
    def _format_session_detail(self, session) -> Dict[str, Any]:
        """格式化会话数据用于详情显示"""
        base = self._format_session_for_list(session)
        
        config = getattr(session, 'config', None) or {}
        result = getattr(session, 'result', None) or {}
        
        base.update({
            'config': config,
            'result': result,
            'errorMessage': getattr(session, 'error_message', None),
            'progress': getattr(session, 'progress', 0.0),
            'datasetId': getattr(session, 'dataset_id', None)
        })
        
        return base
    
    def _generate_download_url(self, session_id: str, _model_path: str) -> str:
        """生成下载URL"""
        # 实际生产环境应使用签名URL或临时token
        return f"/api/v1/training/models/download?session_id={session_id}"
    
    def _start_training_async(
        self, 
        session_id: str, 
        user_id: str, 
        config: Dict[str, Any],
        training_type: str
    ):
        """异步启动训练任务 (通过 Launcher 统一调度)"""
        
        try:
            import threading
            
            def run_training():
                try:
                    # 构建启动器配置
                    launcher_config = self._build_launcher_config(config, training_type)
                    launcher = TrainingSystemLauncher(launcher_config)
                    
                    # 分析配置并选择训练器
                    analysis = launcher.analyze_config()
                    logger.info("Launcher analysis for session %s: %s", session_id, analysis)
                    
                    trainer = launcher.select_trainer(analysis)
                    
                    if trainer is None:
                        raise BusinessLogicError("Launcher returned no trainer", operation="start_training")
                    
                    # 更新状态为运行中
                    self._update_session_status(session_id, user_id, 'running')
                    
                    # 初始化进度跟踪
                    try:
                        progress_manager = get_progress_manager()
                        total_steps = config.get('total_steps', 1000)
                        progress_manager.create_progress_tracker(session_id, total_steps=total_steps)
                        progress_manager.set_status(session_id, 'running')
                    except Exception as e:
                        logger.warning("Failed to initialize progress tracking: %s", e)
                    
                    # 设置进度回调
                    def progress_callback(progress: float, metrics: Dict[str, Any] = None):
                        try:
                            if metrics:
                                progress_manager = get_progress_manager()
                                progress_manager.update_progress(
                                    session_id,
                                    progress=progress,
                                    metrics=metrics
                                )
                        except Exception:
                            pass
                    
                    if hasattr(trainer, 'set_progress_callback'):
                        trainer.set_progress_callback(progress_callback)
                    
                    # 执行训练
                    result = trainer.train()
                    
                    # 更新结果
                    if result.get('success'):
                        self._update_session_result(session_id, user_id, 'completed', result)
                    else:
                        self._update_session_result(
                            session_id, user_id, 'failed', 
                            result, result.get('error', 'Training failed')
                        )
                        
                except Exception as e:
                    logger.error("Training failed for session %s: %s", session_id, e)
                    self._update_session_result(session_id, user_id, 'failed', {}, str(e))
            
            # 启动后台线程
            thread = threading.Thread(target=run_training, daemon=True)
            thread.start()
            
            logger.info("Started training thread for session %s", session_id)
            
        except Exception as e:
            logger.error("Failed to start training async: %s", e)
    
    def launch_production_training(
        self,
        session_id: str,
        user_id: str,
        training_type: str = 'standard',
        **kwargs
    ) -> Dict[str, Any]:
        """使用生产级启动器启动训练
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            training_type: 训练类型
            **kwargs: 额外配置
            
        Returns:
            启动结果
        """
        try:
            # 获取会话详情
            detail = self.get_training_history_detail(session_id, user_id)
            if not detail:
                return {'success': False, 'error': 'Session not found'}
            
            config = detail.get('config', {})
            
            # 构建生产级配置
            production_config = create_production_training_config(
                training_type=training_type,
                output_dir=config.get('output_dir', f'./outputs/{session_id}'),
                model_name=detail.get('modelName', 'production_model'),
                num_epochs=config.get('epochs', kwargs.get('num_epochs', 10)),
                batch_size=config.get('batch_size', kwargs.get('batch_size', 16)),
                learning_rate=config.get('learning_rate', kwargs.get('learning_rate', 2e-5)),
                enable_checkpoint=kwargs.get('enable_checkpoint', True),
                enable_monitoring=kwargs.get('enable_monitoring', True),
            )
            
            # 在后台线程启动生产级训练
            import threading
            
            def run_production():
                try:
                    launcher = ProductionTrainingLauncher(production_config)
                    result = launcher.launch_training()
                    
                    if result.get('success'):
                        self._update_session_result(session_id, user_id, 'completed', result)
                    else:
                        self._update_session_result(
                            session_id, user_id, 'failed',
                            result, result.get('error', 'Training failed')
                        )
                except Exception as e:
                    logger.error("Production training failed: %s", e)
                    self._update_session_result(session_id, user_id, 'failed', {}, str(e))
            
            thread = threading.Thread(target=run_production, daemon=True)
            thread.start()
            
            return {
                'success': True,
                'session_id': session_id,
                'status': 'running',
                'training_type': training_type,
                'message': '生产级训练已启动'
            }
            
        except Exception as e:
            logger.error("Failed to launch production training: %s", e)
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
    
    def _build_launcher_config(self, config: Dict[str, Any], training_type: str) -> Dict[str, Any]:
        """构建启动器配置"""
        launcher_config = {
            'model': {
                'name': config.get('model_name', 'default_model'),
                'type': config.get('model_type', 'standard')
            },
            'training': {
                'mode': training_type,
                'num_epochs': config.get('epochs', 10),
                'batch_size': config.get('batch_size', 32),
                'learning_rate': config.get('learning_rate', 1e-4)
            },
            'data': {
                'train_path': config.get('train_data_path', './data/train')
            }
        }
        
        # 根据训练类型添加特定配置
        if training_type in ['PT', 'pretrain']:
            launcher_config['three_stage'] = {'enabled': True, 'stages': ['pretrain']}
        elif training_type in ['SFT', 'finetune']:
            launcher_config['three_stage'] = {'enabled': True, 'stages': ['finetune']}
        elif training_type in ['DPO', 'preference']:
            launcher_config['three_stage'] = {'enabled': True, 'stages': ['preference']}
        elif training_type == 'multimodal':
            launcher_config['multimodal'] = {'enabled': True}
        elif training_type == 'distillation':
            launcher_config['distillation'] = {'enabled': True}
        elif training_type == 'distributed':
            launcher_config['distributed'] = {'enabled': True}
        
        return launcher_config
    
    def _update_session_status(self, session_id: str, user_id: str, status: str):
        """更新会话状态"""
        try:
            if self._repo is not None:
                # 使用仓库更新
                pass  # Repository需要提供update方法
            else:
                key = f"{user_id}:{session_id}"
                if key in self._memory_store:
                    self._memory_store[key]['status'] = status
                    if status == 'running':
                        self._memory_store[key]['started_at'] = datetime.utcnow().isoformat()
        except Exception as e:
            logger.error("Failed to update session status: %s", e)
    
    def _update_session_result(
        self, 
        session_id: str, 
        user_id: str, 
        status: str, 
        result: Dict[str, Any],
        error_message: str = None
    ):
        """更新会话结果"""
        try:
            if self._repo is not None:
                # 使用仓库更新
                pass  # Repository需要提供update方法
            else:
                key = f"{user_id}:{session_id}"
                if key in self._memory_store:
                    self._memory_store[key]['status'] = status
                    self._memory_store[key]['result'] = result
                    self._memory_store[key]['completed_at'] = datetime.utcnow().isoformat()
                    if error_message:
                        self._memory_store[key]['error_message'] = error_message
        except Exception as e:
            logger.error("Failed to update session result: %s", e)
    
    def _cleanup_training_files(self, session_id: str):
        """清理训练相关文件"""
        # 实际实现中应该清理模型文件、日志等
        logger.info("Cleanup training files for session: %s", session_id)
    
    def _get_from_memory(
        self, 
        user_id: str, 
        page: int, 
        limit: int,
        status: Optional[str],
        training_type: Optional[str],
        model_name: Optional[str],
        _start_date: Optional[str],
        _end_date: Optional[str]
    ) -> tuple:
        """从内存存储获取数据"""
        # 过滤用户的会话
        user_sessions = [
            v for k, v in self._memory_store.items() 
            if k.startswith(f"{user_id}:")
        ]
        
        # 应用过滤器
        if status:
            user_sessions = [s for s in user_sessions if s.get('status') == status]
        if training_type:
            user_sessions = [s for s in user_sessions if s.get('training_type') == training_type]
        if model_name:
            user_sessions = [s for s in user_sessions if model_name.lower() in s.get('model_id', '').lower()]
        
        total = len(user_sessions)
        offset = (page - 1) * limit
        paginated = user_sessions[offset:offset + limit]
        
        return paginated, total
    
    def _calculate_memory_statistics(self, user_id: str) -> Dict[str, Any]:
        """计算内存存储的统计信息"""
        user_sessions = [
            v for k, v in self._memory_store.items() 
            if k.startswith(f"{user_id}:")
        ]
        
        total = len(user_sessions)
        completed = len([s for s in user_sessions if s.get('status') == 'completed'])
        failed = len([s for s in user_sessions if s.get('status') == 'failed'])
        cancelled = len([s for s in user_sessions if s.get('status') == 'cancelled'])
        
        return {
            'totalTrainings': total,
            'completedTrainings': completed,
            'failedTrainings': failed,
            'cancelledTrainings': cancelled,
            'averageTrainingTime': 0.0,
            'averageAccuracy': 0.0,
            'averageLoss': 0.0,
            'mostUsedModel': 'unknown',
            'mostUsedTrainingType': 'unknown'
        }


# ==================== 全局服务实例 ====================

_global_training_history_service: Optional[TrainingHistoryService] = None


def get_training_history_service(use_memory_storage: bool = False) -> TrainingHistoryService:
    """
    获取训练历史服务实例
    
    Args:
        use_memory_storage: 是否使用内存存储
        
    Returns:
        TrainingHistoryService 实例
    """
    global _global_training_history_service
    
    if _global_training_history_service is None:
        _global_training_history_service = TrainingHistoryService(use_memory_storage)
    
    return _global_training_history_service
