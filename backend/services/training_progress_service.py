# -*- coding: utf-8 -*-
"""训练进度服务层

提供训练进度管理的完整业务逻辑。

架构调用关系：
API层 (training_progress_api.py)
    -> Service层 (本模块)
        -> Repository层 (training_progress_repository.py)
        -> Training模块 (backend/modules/training)
"""

import logging
import os
import re
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class TrainingProgressService:
    """
    训练进度服务层
    
    提供训练进度的完整管理：
    - 获取实时进度
    - 更新进度
    - 获取日志、指标、事件
    - 检查点管理
    - 资源监控
    
    调用关系：
    - 使用 TrainingProgressRepository 进行数据持久化
    - 使用 progress_manager 获取实时进度
    """
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化服务"""
        self._use_memory_storage = use_memory_storage
        self._repo = None
        self._progress_manager = None
        
        self._init_repository()
        self._init_progress_manager()
    
    def _init_repository(self):
        """初始化仓库层"""
        try:
            from backend.repositories.training_progress_repository import get_training_progress_repository
            self._repo = get_training_progress_repository(self._use_memory_storage)
            logger.info("TrainingProgressRepository initialized")
        except ImportError as e:
            logger.warning(f"Failed to import TrainingProgressRepository: {e}")
            self._repo = None
    
    def _init_progress_manager(self):
        """初始化进度管理器"""
        try:
            from backend.modules.training.progress.progress_manager import get_progress_manager
            self._progress_manager = get_progress_manager()
            logger.info("Progress manager initialized")
        except ImportError as e:
            logger.warning(f"Progress manager not available: {e}")
            self._progress_manager = None
    
    # ==================== 进度查询 ====================
    
    def get_progress(
        self,
        session_id: str,
        user_id: str,
        tenant_id: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        获取训练进度
        
        Args:
            session_id: 训练会话ID
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            进度数据字典
        """
        if not self._repo:
            return None
        
        try:
            # 获取会话
            session = self._repo.get_session(session_id, user_id, tenant_id)
            if not session:
                return None
            
            # 获取最新进度
            progress = self._repo.get_latest_progress(session_id)
            
            return self._format_progress(session, progress)
            
        except Exception as e:
            logger.error(f"Failed to get progress for session {session_id}: {e}")
            return None
    
    def get_progress_history(
        self,
        session_id: str,
        user_id: str,
        tenant_id: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        获取进度历史记录
        
        Args:
            session_id: 训练会话ID
            user_id: 用户ID
            tenant_id: 租户ID
            limit: 返回数量
            offset: 偏移量
            
        Returns:
            进度历史列表
        """
        if not self._repo:
            return {'history': [], 'total': 0}
        
        try:
            # 验证会话所有权
            session = self._repo.get_session(session_id, user_id, tenant_id)
            if not session:
                return {'history': [], 'total': 0}
            
            # 获取历史记录
            records, total = self._repo.get_progress_history(
                session_id, limit, offset, order_desc=True
            )
            
            history = []
            for r in records:
                history.append(self._format_progress_record(r))
            
            return {'history': history, 'total': total}
            
        except Exception as e:
            logger.error(f"Failed to get progress history: {e}")
            return {'history': [], 'total': 0}
    
    def update_progress(
        self,
        session_id: str,
        user_id: str,
        tenant_id: str = None,
        progress_data: Dict[str, Any] = None,
        push_realtime: bool = True
    ) -> Dict[str, Any]:
        """
        更新训练进度
        
        Args:
            session_id: 训练会话ID
            user_id: 用户ID
            tenant_id: 租户ID
            progress_data: 进度数据
            push_realtime: 是否推送实时更新
            
        Returns:
            更新结果
        """
        if not progress_data:
            return {'success': False, 'message': 'No progress data provided'}
        
        if not self._repo:
            return {'success': False, 'message': 'Repository not available'}
        
        try:
            # 验证会话
            session = self._repo.get_session(session_id, user_id, tenant_id)
            if not session:
                return {'success': False, 'message': 'Session not found'}
            
            # 添加 session_id 到进度数据
            progress_data['session_id'] = session_id
            
            # 创建进度记录
            progress_id = self._repo.create_progress(progress_data)
            if not progress_id:
                return {'success': False, 'message': 'Failed to create progress record'}
            
            # 更新会话进度百分比
            total_epochs = progress_data.get('total_epochs', 1)
            current_epoch = progress_data.get('epoch', 0)
            if total_epochs > 0:
                progress_percentage = (current_epoch / total_epochs) * 100
                self._repo.update_session_progress(session_id, progress_percentage, tenant_id)
            
            # 实时推送进度更新
            pushed_count = 0
            if push_realtime:
                pushed_count = self._push_progress_update(session_id, progress_data)
            
            logger.info(f"Progress updated for session {session_id}, pushed to {pushed_count} subscribers")
            return {
                'success': True,
                'session_id': session_id,
                'progress_id': progress_id,
                'pushed_to': pushed_count
            }
            
        except Exception as e:
            logger.error(f"Failed to update progress: {e}")
            return {'success': False, 'message': str(e)}
    
    def _push_progress_update(self, session_id: str, progress_data: Dict[str, Any]) -> int:
        """
        推送进度更新到订阅者
        
        Args:
            session_id: 训练会话ID
            progress_data: 进度数据
            
        Returns:
            推送到的用户数量
        """
        try:
            from backend.api.training.training_progress_websocket_api import push_training_progress
            return push_training_progress(session_id, progress_data)
        except ImportError:
            logger.debug("Realtime push not available")
            return 0
        except Exception as e:
            logger.warning(f"Failed to push progress update: {e}")
            return 0
    
    # ==================== 日志管理 ====================
    
    def get_logs(
        self,
        session_id: str,
        user_id: str,
        tenant_id: str = None,
        limit: int = 100,
        level: str = None,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> Dict[str, Any]:
        """
        获取训练日志
        
        Args:
            session_id: 训练会话ID
            user_id: 用户ID
            tenant_id: 租户ID
            limit: 日志数量限制
            level: 日志级别过滤
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            日志数据
        """
        logs = []
        
        if not self._repo:
            return {'session_id': session_id, 'logs': [], 'total': 0}
        
        # 尝试从文件读取日志
        log_path = self._repo.get_session_log_path(session_id, user_id)
        if log_path and os.path.isfile(log_path):
            try:
                logs = self._read_log_file(log_path, limit, level)
            except Exception as e:
                logger.error(f"Failed to read log file: {e}")
        
        # 如果文件日志不可用，从数据库构建日志
        if not logs:
            logs = self._get_logs_from_progress(session_id, limit, level)
        
        return {
            'session_id': session_id,
            'logs': logs,
            'total': len(logs)
        }
    
    def _read_log_file(
        self,
        log_path: str,
        limit: int,
        level: str = None
    ) -> List[Dict[str, Any]]:
        """从文件读取日志"""
        logs = []
        
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # 取最后 limit 行
        tail_lines = lines[-limit:] if limit > 0 else lines
        
        for line in tail_lines:
            line = line.strip()
            if not line:
                continue
            
            # 解析日志行
            log_entry = self._parse_log_line(line)
            
            # 级别过滤
            if level and log_entry.get('level', '').upper() != level.upper():
                continue
            
            logs.append(log_entry)
        
        return logs
    
    def _parse_log_line(self, line: str) -> Dict[str, Any]:
        """解析日志行"""
        timestamp = datetime.now().isoformat()
        level = "INFO"
        message = line
        
        # 尝试提取时间戳
        ts_match = re.search(r'\[(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})', line)
        if ts_match:
            timestamp = ts_match.group(1)
        
        # 尝试提取级别
        level_match = re.search(r'\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]', line, re.IGNORECASE)
        if level_match:
            level = level_match.group(1).upper()
        
        return {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
    
    def _get_logs_from_progress(
        self,
        session_id: str,
        limit: int,
        level: str = None
    ) -> List[Dict[str, Any]]:
        """从进度记录构建日志"""
        logs = []
        
        if not self._repo:
            return logs
        
        try:
            records = self._repo.get_progress_with_logs(session_id, limit)
            
            for r in records:
                # 从进度记录构建日志
                created_at = self._get_attr(r, 'created_at')
                stage = self._get_attr(r, 'stage')
                epoch = self._get_attr(r, 'epoch')
                loss = self._get_attr(r, 'loss')
                
                if loss is not None:
                    message = f"Stage: {stage}, Epoch: {epoch}, Loss: {loss:.4f}"
                else:
                    message = f"Stage: {stage}"
                
                logs.append({
                    'timestamp': self._format_datetime(created_at),
                    'level': 'INFO',
                    'message': message
                })
                
                # 检查 metrics 中是否有日志信息
                metrics = self._get_attr(r, 'metrics')
                if metrics and isinstance(metrics, dict):
                    log_entries = metrics.get('logs', [])
                    for entry in log_entries:
                        if isinstance(entry, dict):
                            logs.append(entry)
            
        except Exception as e:
            logger.error(f"Failed to get logs from progress: {e}")
        
        return logs
    
    # ==================== 指标管理 ====================
    
    def get_metrics(
        self,
        session_id: str,
        user_id: str,
        tenant_id: str = None,
        limit: int = 100,
        metric_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        获取训练指标
        
        Args:
            session_id: 训练会话ID
            user_id: 用户ID
            tenant_id: 租户ID
            limit: 指标数量限制
            metric_names: 指定的指标名称
            
        Returns:
            指标数据
        """
        if not self._repo:
            return {'session_id': session_id, 'metrics': [], 'total': 0}
        
        try:
            # 验证会话
            session = self._repo.get_session(session_id, user_id, tenant_id)
            if not session:
                return {'session_id': session_id, 'metrics': [], 'total': 0}
            
            # 获取进度记录
            records, total = self._repo.get_progress_history(session_id, limit, 0)
            
            metrics = []
            for r in records:
                metric = self._format_metric_record(r)
                
                # 如果指定了指标名称，只返回指定的指标
                if metric_names:
                    metric = {k: v for k, v in metric.items() if k in metric_names or k == 'timestamp'}
                
                metrics.append(metric)
            
            return {
                'session_id': session_id,
                'metrics': metrics,
                'total': len(metrics)
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {'session_id': session_id, 'metrics': [], 'total': 0}
    
    def get_metric_summary(
        self,
        session_id: str,
        user_id: str,
        tenant_id: str = None
    ) -> Dict[str, Any]:
        """
        获取指标摘要统计
        
        Returns:
            指标的最小值、最大值、平均值等统计
        """
        if not self._repo:
            return {}
        
        try:
            # 验证会话（可选，因为 summary 可能用于内部调用）
            return self._repo.get_metrics_summary(session_id)
            
        except Exception as e:
            logger.error(f"Failed to get metric summary: {e}")
            return {}
    
    # ==================== 事件管理 ====================
    
    def get_events(
        self,
        session_id: str,
        user_id: str,
        tenant_id: str = None,
        event_types: List[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        获取训练事件
        
        Args:
            session_id: 训练会话ID
            user_id: 用户ID
            tenant_id: 租户ID
            event_types: 事件类型过滤
            limit: 事件数量限制
            
        Returns:
            事件列表
        """
        if not self._repo:
            return {'session_id': session_id, 'events': []}
        
        try:
            # 获取会话
            session = self._repo.get_session(session_id, user_id, tenant_id)
            if not session:
                return {'session_id': session_id, 'events': []}
            
            # 获取所有进度记录
            progress_records = self._repo.get_all_progress(session_id, limit)
            
            # 构建事件
            events = self._build_events_from_session(session, progress_records)
            
            # 过滤事件类型
            if event_types:
                events = [e for e in events if e.get('event_type') in event_types]
            
            return {'session_id': session_id, 'events': events}
            
        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            return {'session_id': session_id, 'events': []}
    
    def _build_events_from_session(
        self,
        session,
        progress_records: List
    ) -> List[Dict[str, Any]]:
        """从会话和进度记录构建事件列表"""
        events = []
        
        # 训练开始事件
        started_at = self._get_attr(session, 'started_at')
        if started_at:
            events.append({
                'timestamp': self._format_datetime(started_at),
                'event_type': 'training_started',
                'description': '训练任务已启动',
                'details': {
                    'model_id': self._get_attr(session, 'model_id'),
                    'dataset_id': self._get_attr(session, 'dataset_id'),
                    'training_type': self._get_attr(session, 'training_type')
                }
            })
        
        # 从进度记录构建事件
        last_epoch = None
        last_stage = None
        
        for r in progress_records:
            ts = self._format_datetime(self._get_attr(r, 'created_at'))
            stage = self._get_attr(r, 'stage')
            epoch = self._get_attr(r, 'epoch')
            loss = self._get_attr(r, 'loss')
            accuracy = self._get_attr(r, 'accuracy')
            metrics = self._get_attr(r, 'metrics')
            
            # 阶段变化事件
            if stage and stage != last_stage:
                events.append({
                    'timestamp': ts,
                    'event_type': 'stage_changed',
                    'description': f'进入训练阶段: {stage}',
                    'details': {
                        'stage': stage,
                        'previous_stage': last_stage
                    }
                })
                last_stage = stage
            
            # Epoch 变化事件
            if epoch is not None and epoch != last_epoch:
                if last_epoch is not None:
                    events.append({
                        'timestamp': ts,
                        'event_type': 'epoch_completed',
                        'description': f'完成第 {last_epoch} 轮训练',
                        'details': {
                            'epoch': last_epoch,
                            'loss': loss or 0.0,
                            'accuracy': accuracy or 0.0
                        }
                    })
                
                events.append({
                    'timestamp': ts,
                    'event_type': 'epoch_started',
                    'description': f'开始第 {epoch} 轮训练',
                    'details': {'epoch': epoch}
                })
                last_epoch = epoch
            
            # 检查 metrics 中的自定义事件
            if metrics and isinstance(metrics, dict):
                custom_event = metrics.get('event')
                if custom_event and isinstance(custom_event, dict):
                    custom_event.setdefault('timestamp', ts)
                    events.append(custom_event)
                
                # 检查点事件
                if metrics.get('checkpoint_saved'):
                    events.append({
                        'timestamp': ts,
                        'event_type': 'checkpoint_saved',
                        'description': f'保存检查点: epoch {epoch}',
                        'details': {
                            'epoch': epoch,
                            'path': metrics.get('checkpoint_path')
                        }
                    })
        
        # 训练完成事件
        completed_at = self._get_attr(session, 'completed_at')
        if completed_at:
            final_loss = None
            final_accuracy = None
            if progress_records:
                last_record = progress_records[-1]
                final_loss = self._get_attr(last_record, 'loss')
                final_accuracy = self._get_attr(last_record, 'accuracy')
            
            events.append({
                'timestamp': self._format_datetime(completed_at),
                'event_type': 'training_completed',
                'description': '训练任务已完成',
                'details': {
                    'status': self._get_attr(session, 'status'),
                    'final_loss': final_loss,
                    'final_accuracy': final_accuracy
                }
            })
        
        # 错误事件
        error_message = self._get_attr(session, 'error_message')
        if error_message:
            events.append({
                'timestamp': self._format_datetime(
                    completed_at or self._get_attr(session, 'updated_at')
                ),
                'event_type': 'training_error',
                'description': '训练任务出错',
                'details': {
                    'error_message': error_message
                }
            })
        
        return events
    
    # ==================== 检查点管理 ====================
    
    def get_checkpoints(
        self,
        session_id: str,
        user_id: str,
        tenant_id: str = None
    ) -> Dict[str, Any]:
        """
        获取检查点列表
        
        Returns:
            检查点列表
        """
        if not self._repo:
            return {'session_id': session_id, 'checkpoints': []}
        
        try:
            # 验证会话
            session = self._repo.get_session(session_id, user_id, tenant_id)
            if not session:
                return {'session_id': session_id, 'checkpoints': []}
            
            # 从结果中获取检查点信息
            checkpoints = []
            result = self._get_attr(session, 'result') or {}
            if 'checkpoints' in result:
                checkpoints = result['checkpoints']
            
            # 从进度记录中补充检查点信息
            progress_checkpoints = self._repo.get_checkpoints(session_id)
            checkpoints.extend(progress_checkpoints)
            
            return {'session_id': session_id, 'checkpoints': checkpoints}
            
        except Exception as e:
            logger.error(f"Failed to get checkpoints: {e}")
            return {'session_id': session_id, 'checkpoints': []}
    
    # ==================== 资源监控 ====================
    
    def get_resource_usage(
        self,
        session_id: str,
        user_id: str,
        tenant_id: str = None
    ) -> Dict[str, Any]:
        """
        获取资源使用情况
        
        Returns:
            当前资源使用数据
        """
        if not self._repo:
            return {}
        
        try:
            # 获取最新的进度记录
            latest = self._repo.get_latest_progress(session_id)
            if not latest:
                return {}
            
            return {
                'gpu': {
                    'utilization': self._get_attr(latest, 'gpu_utilization') or 0.0,
                    'memory_used': self._get_attr(latest, 'gpu_memory_used') or 0.0,
                    'memory_total': self._get_attr(latest, 'gpu_memory_total') or 0.0,
                    'temperature': self._get_attr(latest, 'gpu_temperature') or 0.0,
                    'power_draw': self._get_attr(latest, 'gpu_power_draw') or 0.0
                },
                'cpu': {
                    'utilization': self._get_attr(latest, 'cpu_utilization') or 0.0,
                    'memory_used': self._get_attr(latest, 'cpu_memory_used') or 0.0,
                    'memory_total': self._get_attr(latest, 'cpu_memory_total') or 0.0,
                    'temperature': self._get_attr(latest, 'cpu_temperature') or 0.0
                },
                'disk': {
                    'read_speed': self._get_attr(latest, 'disk_read_speed') or 0.0,
                    'write_speed': self._get_attr(latest, 'disk_write_speed') or 0.0,
                    'utilization': self._get_attr(latest, 'disk_utilization') or 0.0
                },
                'network': {
                    'download_speed': self._get_attr(latest, 'network_download_speed') or 0.0,
                    'upload_speed': self._get_attr(latest, 'network_upload_speed') or 0.0,
                    'latency': self._get_attr(latest, 'network_latency') or 0.0
                },
                'training': {
                    'samples_per_second': self._get_attr(latest, 'samples_per_second') or 0.0,
                    'tokens_per_second': self._get_attr(latest, 'tokens_per_second') or 0.0,
                    'batch_size': self._get_attr(latest, 'batch_size') or 0,
                    'gradient_norm': self._get_attr(latest, 'gradient_norm') or 0.0
                },
                'timestamp': self._format_datetime(self._get_attr(latest, 'created_at'))
            }
            
        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            return {}
    
    # ==================== 实时进度（WebSocket支持） ====================
    
    def get_realtime_progress(
        self,
        session_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        获取实时进度（用于 WebSocket 推送）
        
        优先从进度管理器获取，否则从数据库获取
        """
        # 尝试从进度管理器获取实时数据
        if self._progress_manager:
            try:
                realtime = self._progress_manager.get_progress(session_id)
                if realtime:
                    return realtime
            except Exception:
                pass
        
        # 回退到数据库
        return self.get_progress(session_id, user_id) or {}
    
    def subscribe_progress(self, session_id: str, user_id: str) -> bool:
        """
        订阅训练进度实时更新
        
        Args:
            session_id: 训练会话ID
            user_id: 用户ID
            
        Returns:
            是否订阅成功
        """
        try:
            from backend.api.training.training_progress_websocket_api import subscribe_to_progress
            return subscribe_to_progress(session_id, user_id)
        except ImportError:
            logger.warning("Realtime subscription not available")
            return False
        except Exception as e:
            logger.error(f"Failed to subscribe: {e}")
            return False
    
    def unsubscribe_progress(self, session_id: str, user_id: str) -> bool:
        """
        取消订阅训练进度实时更新
        
        Args:
            session_id: 训练会话ID
            user_id: 用户ID
            
        Returns:
            是否取消成功
        """
        try:
            from backend.api.training.training_progress_websocket_api import unsubscribe_from_progress
            return unsubscribe_from_progress(session_id, user_id)
        except ImportError:
            logger.warning("Realtime unsubscription not available")
            return False
        except Exception as e:
            logger.error(f"Failed to unsubscribe: {e}")
            return False
    
    def push_progress_event(
        self,
        session_id: str,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> int:
        """
        推送进度事件（不持久化，仅实时推送）
        
        Args:
            session_id: 训练会话ID
            event_type: 事件类型
            event_data: 事件数据
            
        Returns:
            推送到的用户数量
        """
        try:
            from backend.api.training.training_progress_websocket_api import push_training_progress
            
            message = {
                'type': event_type,
                **event_data
            }
            return push_training_progress(session_id, message)
        except ImportError:
            logger.debug("Realtime push not available")
            return 0
        except Exception as e:
            logger.warning(f"Failed to push event: {e}")
            return 0
    
    # ==================== 私有辅助方法 ====================
    
    def _get_attr(self, obj, attr: str, default=None):
        """获取对象或字典的属性值"""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)
    
    def _format_progress(self, session, progress) -> Dict[str, Any]:
        """格式化进度数据"""
        session_id = self._get_attr(session, 'session_id')
        status = self._get_attr(session, 'status')
        
        if not progress:
            return {
                'session_id': session_id,
                'current_epoch': 0,
                'total_epochs': 0,
                'current_step': 0,
                'total_steps': 0,
                'current_stage': '',
                'total_stages': 0,
                'loss': 0.0,
                'accuracy': 0.0,
                'status': status,
                'progress_percentage': 0.0,
                'stage_progress_percentage': 0.0,
                'updated_at': self._format_datetime(self._get_attr(session, 'updated_at'))
            }
        
        # 计算进度百分比
        total_epochs = self._get_attr(progress, 'total_epochs') or 0
        current_epoch = self._get_attr(progress, 'current_epoch') or self._get_attr(progress, 'epoch') or 0
        
        progress_percentage = 0.0
        if total_epochs and total_epochs > 0:
            progress_percentage = (current_epoch / total_epochs) * 100
        
        total_stages = self._get_attr(progress, 'total_stages') or 0
        stage = self._get_attr(progress, 'stage')
        current_stage_num = 0
        if stage:
            stage_map = {'pretrain': 1, 'finetune': 2, 'preference': 3}
            current_stage_num = stage_map.get(stage, 1)
        
        stage_progress_percentage = 0.0
        if total_stages and total_stages > 0:
            stage_progress_percentage = (current_stage_num / total_stages) * 100
        
        return {
            'session_id': session_id,
            'current_epoch': current_epoch,
            'total_epochs': total_epochs,
            'current_step': self._get_attr(progress, 'step') or 0,
            'total_steps': self._get_attr(progress, 'total_steps') or 0,
            'current_stage': stage or '',
            'total_stages': total_stages,
            'loss': self._get_attr(progress, 'loss') or 0.0,
            'accuracy': self._get_attr(progress, 'accuracy') or 0.0,
            'learning_rate': self._get_attr(progress, 'learning_rate') or 0.0,
            'status': status,
            'progress_percentage': round(progress_percentage, 2),
            'stage_progress_percentage': round(stage_progress_percentage, 2),
            'updated_at': self._format_datetime(
                self._get_attr(progress, 'updated_at') or self._get_attr(progress, 'created_at')
            )
        }
    
    def _format_progress_record(self, record) -> Dict[str, Any]:
        """格式化单条进度记录"""
        return {
            'timestamp': self._format_datetime(self._get_attr(record, 'created_at')),
            'stage': self._get_attr(record, 'stage'),
            'epoch': self._get_attr(record, 'epoch') or 0,
            'step': self._get_attr(record, 'step') or 0,
            'loss': self._get_attr(record, 'loss') or 0.0,
            'accuracy': self._get_attr(record, 'accuracy') or 0.0,
            'learning_rate': self._get_attr(record, 'learning_rate') or 0.0,
            'metrics': self._get_attr(record, 'metrics') or {}
        }
    
    def _format_metric_record(self, record) -> Dict[str, Any]:
        """格式化指标记录"""
        return {
            'timestamp': self._format_datetime(self._get_attr(record, 'created_at')),
            'epoch': self._get_attr(record, 'epoch') or 0,
            'step': self._get_attr(record, 'step') or 0,
            'loss': self._get_attr(record, 'loss') or 0.0,
            'accuracy': self._get_attr(record, 'accuracy') or 0.0,
            'learning_rate': self._get_attr(record, 'learning_rate') or 0.0,
            # GPU 指标
            'gpu_utilization': self._get_attr(record, 'gpu_utilization') or 0.0,
            'gpu_memory_used': self._get_attr(record, 'gpu_memory_used') or 0.0,
            'gpu_memory_total': self._get_attr(record, 'gpu_memory_total') or 0.0,
            'gpu_temperature': self._get_attr(record, 'gpu_temperature') or 0.0,
            'gpu_power_draw': self._get_attr(record, 'gpu_power_draw') or 0.0,
            # CPU 指标
            'cpu_utilization': self._get_attr(record, 'cpu_utilization') or 0.0,
            'cpu_memory_used': self._get_attr(record, 'cpu_memory_used') or 0.0,
            'cpu_memory_total': self._get_attr(record, 'cpu_memory_total') or 0.0,
            'cpu_temperature': self._get_attr(record, 'cpu_temperature') or 0.0,
            # 训练性能
            'samples_per_second': self._get_attr(record, 'samples_per_second') or 0.0,
            'tokens_per_second': self._get_attr(record, 'tokens_per_second') or 0.0,
            'batch_size': self._get_attr(record, 'batch_size') or 0,
            'gradient_norm': self._get_attr(record, 'gradient_norm') or 0.0,
            # IO 指标
            'disk_read_speed': self._get_attr(record, 'disk_read_speed') or 0.0,
            'disk_write_speed': self._get_attr(record, 'disk_write_speed') or 0.0,
            'disk_utilization': self._get_attr(record, 'disk_utilization') or 0.0,
            'network_download_speed': self._get_attr(record, 'network_download_speed') or 0.0,
            'network_upload_speed': self._get_attr(record, 'network_upload_speed') or 0.0,
            'network_latency': self._get_attr(record, 'network_latency') or 0.0
        }
    
    def _format_datetime(self, dt) -> Optional[str]:
        """格式化日期时间"""
        if dt is None:
            return None
        if isinstance(dt, str):
            return dt
        return dt.isoformat() if isinstance(dt, datetime) else str(dt)


# ==================== 全局服务实例 ====================

_global_training_progress_service: Optional[TrainingProgressService] = None


def get_training_progress_service(use_memory_storage: bool = False) -> TrainingProgressService:
    """获取训练进度服务实例"""
    global _global_training_progress_service
    
    if _global_training_progress_service is None:
        _global_training_progress_service = TrainingProgressService(use_memory_storage)
    
    return _global_training_progress_service
