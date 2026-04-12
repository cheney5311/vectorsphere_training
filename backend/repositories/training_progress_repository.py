# -*- coding: utf-8 -*-
"""训练进度仓库

提供训练进度数据访问接口，支持 CRUD 操作和高级查询。

数据模型：
- TrainingSession: 训练会话
- TrainingProgress: 训练进度记录
"""

import logging
import uuid
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from sqlalchemy import func, and_, or_, desc, asc

logger = logging.getLogger(__name__)


class TrainingProgressRepository:
    """
    训练进度仓库
    
    提供训练进度的数据访问层，封装所有数据库操作。
    """
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        """
        初始化仓库
        
        Args:
            db_service: 数据库服务实例
            use_memory_storage: 是否使用内存存储（用于测试）
        """
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        
        # 内存存储
        self._sessions: Dict[str, Dict] = {}
        self._progress_records: Dict[str, List[Dict]] = {}
        
        if not use_memory_storage:
            self._init_database(db_service)
    
    def _init_database(self, db_service=None):
        """初始化数据库连接"""
        try:
            if db_service:
                self._db_service = db_service
            else:
                from backend.modules.database.manager import get_database_manager
                self._db_service = get_database_manager()
            logger.info("Database manager initialized for TrainingProgressRepository")
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    # ==================== Session CRUD ====================
    
    def get_session(
        self,
        session_id: str,
        user_id: str = None,
        tenant_id: str = None
    ) -> Optional[Any]:
        """
        获取训练会话
        
        Args:
            session_id: 会话ID
            user_id: 用户ID（用于权限验证）
            tenant_id: 租户ID
            
        Returns:
            训练会话对象或None
        """
        if self._use_memory_storage:
            session = self._sessions.get(session_id)
            if session and (not user_id or session.get('user_id') == user_id):
                return session
            return None
        
        try:
            from backend.schemas.training_models import TrainingSession
            
            with self._db_service.get_session() as db:
                query = db.query(TrainingSession).filter(
                    TrainingSession.session_id == session_id
                )
                
                if user_id:
                    query = query.filter(TrainingSession.user_id == user_id)
                
                if tenant_id:
                    query = query.filter(TrainingSession.tenant_id == tenant_id)
                
                return query.first()
                
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    def update_session_progress(
        self,
        session_id: str,
        progress_percentage: float,
        tenant_id: str = None
    ) -> bool:
        """
        更新会话进度百分比
        
        Args:
            session_id: 会话ID
            progress_percentage: 进度百分比 (0-100)
            tenant_id: 租户ID
            
        Returns:
            是否更新成功
        """
        if self._use_memory_storage:
            if session_id in self._sessions:
                self._sessions[session_id]['progress'] = progress_percentage
                self._sessions[session_id]['updated_at'] = datetime.utcnow()
                return True
            return False
        
        try:
            from backend.schemas.training_models import TrainingSession
            
            with self._db_service.get_session() as db:
                session = db.query(TrainingSession).filter(
                    TrainingSession.session_id == session_id
                ).first()
                
                if session:
                    session.progress = progress_percentage
                    session.updated_at = datetime.utcnow()
                    db.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to update session progress: {e}")
            return False
    
    # ==================== Progress CRUD ====================
    
    def create_progress(self, progress_data: Dict[str, Any]) -> Optional[str]:
        """
        创建进度记录
        
        Args:
            progress_data: 进度数据字典
            
        Returns:
            进度记录ID或None
        """
        progress_id = str(uuid.uuid4())
        
        if self._use_memory_storage:
            session_id = progress_data.get('session_id')
            if session_id not in self._progress_records:
                self._progress_records[session_id] = []
            
            record = {
                'id': progress_id,
                'created_at': datetime.utcnow(),
                **progress_data
            }
            self._progress_records[session_id].append(record)
            return progress_id
        
        try:
            from backend.schemas.training_models import TrainingProgress
            
            with self._db_service.get_session() as db:
                progress = TrainingProgress(
                    id=progress_id,
                    session_id=progress_data.get('session_id'),
                    stage=progress_data.get('stage', 'training'),
                    epoch=progress_data.get('epoch'),
                    step=progress_data.get('step'),
                    total_steps=progress_data.get('total_steps'),
                    loss=progress_data.get('loss'),
                    accuracy=progress_data.get('accuracy'),
                    learning_rate=progress_data.get('learning_rate'),
                    metrics=progress_data.get('metrics'),
                    # GPU 监控
                    gpu_utilization=progress_data.get('gpu_utilization'),
                    gpu_memory_used=progress_data.get('gpu_memory_used'),
                    gpu_memory_total=progress_data.get('gpu_memory_total'),
                    gpu_temperature=progress_data.get('gpu_temperature'),
                    gpu_power_draw=progress_data.get('gpu_power_draw'),
                    # CPU 监控
                    cpu_utilization=progress_data.get('cpu_utilization'),
                    cpu_memory_used=progress_data.get('cpu_memory_used'),
                    cpu_memory_total=progress_data.get('cpu_memory_total'),
                    cpu_temperature=progress_data.get('cpu_temperature'),
                    # 训练性能
                    samples_per_second=progress_data.get('samples_per_second'),
                    tokens_per_second=progress_data.get('tokens_per_second'),
                    batch_size=progress_data.get('batch_size'),
                    gradient_norm=progress_data.get('gradient_norm'),
                    # IO 监控
                    disk_read_speed=progress_data.get('disk_read_speed'),
                    disk_write_speed=progress_data.get('disk_write_speed'),
                    disk_utilization=progress_data.get('disk_utilization'),
                    network_download_speed=progress_data.get('network_download_speed'),
                    network_upload_speed=progress_data.get('network_upload_speed'),
                    network_latency=progress_data.get('network_latency')
                )
                
                db.add(progress)
                db.commit()
                
                logger.debug(f"Progress record created: {progress_id}")
                return progress_id
                
        except Exception as e:
            logger.error(f"Failed to create progress record: {e}")
            return None
    
    def get_latest_progress(
        self,
        session_id: str
    ) -> Optional[Any]:
        """
        获取最新的进度记录
        
        Args:
            session_id: 会话ID
            
        Returns:
            最新的进度记录或None
        """
        if self._use_memory_storage:
            records = self._progress_records.get(session_id, [])
            if records:
                return max(records, key=lambda x: x.get('created_at', datetime.min))
            return None
        
        try:
            from backend.schemas.training_models import TrainingProgress
            
            with self._db_service.get_session() as db:
                return db.query(TrainingProgress).filter(
                    TrainingProgress.session_id == session_id
                ).order_by(TrainingProgress.created_at.desc()).first()
                
        except Exception as e:
            logger.error(f"Failed to get latest progress: {e}")
            return None
    
    def get_progress_history(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
        order_desc: bool = True
    ) -> Tuple[List[Any], int]:
        """
        获取进度历史记录
        
        Args:
            session_id: 会话ID
            limit: 返回数量
            offset: 偏移量
            order_desc: 是否降序排列
            
        Returns:
            (进度记录列表, 总数)
        """
        if self._use_memory_storage:
            records = self._progress_records.get(session_id, [])
            total = len(records)
            
            # 排序
            sorted_records = sorted(
                records,
                key=lambda x: x.get('created_at', datetime.min),
                reverse=order_desc
            )
            
            # 分页
            paginated = sorted_records[offset:offset + limit]
            return paginated, total
        
        try:
            from backend.schemas.training_models import TrainingProgress
            
            with self._db_service.get_session() as db:
                query = db.query(TrainingProgress).filter(
                    TrainingProgress.session_id == session_id
                )
                
                total = query.count()
                
                if order_desc:
                    query = query.order_by(TrainingProgress.created_at.desc())
                else:
                    query = query.order_by(TrainingProgress.created_at.asc())
                
                records = query.offset(offset).limit(limit).all()
                return records, total
                
        except Exception as e:
            logger.error(f"Failed to get progress history: {e}")
            return [], 0
    
    def get_all_progress(
        self,
        session_id: str,
        limit: int = None
    ) -> List[Any]:
        """
        获取所有进度记录（用于事件构建等）
        
        Args:
            session_id: 会话ID
            limit: 可选的数量限制
            
        Returns:
            进度记录列表
        """
        if self._use_memory_storage:
            records = self._progress_records.get(session_id, [])
            sorted_records = sorted(
                records,
                key=lambda x: x.get('created_at', datetime.min)
            )
            if limit:
                return sorted_records[:limit]
            return sorted_records
        
        try:
            from backend.schemas.training_models import TrainingProgress
            
            with self._db_service.get_session() as db:
                query = db.query(TrainingProgress).filter(
                    TrainingProgress.session_id == session_id
                ).order_by(TrainingProgress.created_at.asc())
                
                if limit:
                    query = query.limit(limit)
                
                return query.all()
                
        except Exception as e:
            logger.error(f"Failed to get all progress: {e}")
            return []
    
    # ==================== 指标统计 ====================
    
    def get_metrics_summary(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        获取指标摘要统计
        
        Args:
            session_id: 会话ID
            
        Returns:
            指标统计字典
        """
        if self._use_memory_storage:
            records = self._progress_records.get(session_id, [])
            if not records:
                return {}
            
            losses = [r.get('loss') for r in records if r.get('loss') is not None]
            accuracies = [r.get('accuracy') for r in records if r.get('accuracy') is not None]
            epochs = [r.get('epoch') for r in records if r.get('epoch') is not None]
            steps = [r.get('step') for r in records if r.get('step') is not None]
            gpu_utils = [r.get('gpu_utilization') for r in records if r.get('gpu_utilization') is not None]
            cpu_utils = [r.get('cpu_utilization') for r in records if r.get('cpu_utilization') is not None]
            
            return {
                'loss': {
                    'min': min(losses) if losses else 0.0,
                    'max': max(losses) if losses else 0.0,
                    'avg': sum(losses) / len(losses) if losses else 0.0
                },
                'accuracy': {
                    'min': min(accuracies) if accuracies else 0.0,
                    'max': max(accuracies) if accuracies else 0.0,
                    'avg': sum(accuracies) / len(accuracies) if accuracies else 0.0
                },
                'epochs_completed': max(epochs) if epochs else 0,
                'steps_completed': max(steps) if steps else 0,
                'avg_gpu_utilization': sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0,
                'avg_cpu_utilization': sum(cpu_utils) / len(cpu_utils) if cpu_utils else 0.0
            }
        
        try:
            from backend.schemas.training_models import TrainingProgress
            
            with self._db_service.get_session() as db:
                result = db.query(
                    func.min(TrainingProgress.loss).label('min_loss'),
                    func.max(TrainingProgress.loss).label('max_loss'),
                    func.avg(TrainingProgress.loss).label('avg_loss'),
                    func.min(TrainingProgress.accuracy).label('min_accuracy'),
                    func.max(TrainingProgress.accuracy).label('max_accuracy'),
                    func.avg(TrainingProgress.accuracy).label('avg_accuracy'),
                    func.max(TrainingProgress.epoch).label('max_epoch'),
                    func.max(TrainingProgress.step).label('max_step'),
                    func.avg(TrainingProgress.gpu_utilization).label('avg_gpu_utilization'),
                    func.avg(TrainingProgress.cpu_utilization).label('avg_cpu_utilization')
                ).filter(
                    TrainingProgress.session_id == session_id
                ).first()
                
                if not result:
                    return {}
                
                return {
                    'loss': {
                        'min': float(result.min_loss) if result.min_loss else 0.0,
                        'max': float(result.max_loss) if result.max_loss else 0.0,
                        'avg': float(result.avg_loss) if result.avg_loss else 0.0
                    },
                    'accuracy': {
                        'min': float(result.min_accuracy) if result.min_accuracy else 0.0,
                        'max': float(result.max_accuracy) if result.max_accuracy else 0.0,
                        'avg': float(result.avg_accuracy) if result.avg_accuracy else 0.0
                    },
                    'epochs_completed': int(result.max_epoch) if result.max_epoch else 0,
                    'steps_completed': int(result.max_step) if result.max_step else 0,
                    'avg_gpu_utilization': float(result.avg_gpu_utilization) if result.avg_gpu_utilization else 0.0,
                    'avg_cpu_utilization': float(result.avg_cpu_utilization) if result.avg_cpu_utilization else 0.0
                }
                
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {}
    
    def get_progress_by_epoch_range(
        self,
        session_id: str,
        start_epoch: int = None,
        end_epoch: int = None
    ) -> List[Any]:
        """
        按 epoch 范围获取进度记录
        
        Args:
            session_id: 会话ID
            start_epoch: 开始 epoch
            end_epoch: 结束 epoch
            
        Returns:
            进度记录列表
        """
        if self._use_memory_storage:
            records = self._progress_records.get(session_id, [])
            filtered = []
            for r in records:
                epoch = r.get('epoch')
                if epoch is None:
                    continue
                if start_epoch is not None and epoch < start_epoch:
                    continue
                if end_epoch is not None and epoch > end_epoch:
                    continue
                filtered.append(r)
            return filtered
        
        try:
            from backend.schemas.training_models import TrainingProgress
            
            with self._db_service.get_session() as db:
                query = db.query(TrainingProgress).filter(
                    TrainingProgress.session_id == session_id
                )
                
                if start_epoch is not None:
                    query = query.filter(TrainingProgress.epoch >= start_epoch)
                if end_epoch is not None:
                    query = query.filter(TrainingProgress.epoch <= end_epoch)
                
                return query.order_by(TrainingProgress.epoch.asc()).all()
                
        except Exception as e:
            logger.error(f"Failed to get progress by epoch range: {e}")
            return []
    
    # ==================== 检查点相关 ====================
    
    def get_checkpoints(
        self,
        session_id: str
    ) -> List[Dict[str, Any]]:
        """
        获取检查点列表
        
        从进度记录中提取检查点信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            检查点列表
        """
        checkpoints = []
        
        if self._use_memory_storage:
            records = self._progress_records.get(session_id, [])
            for r in records:
                metrics = r.get('metrics', {})
                if metrics and metrics.get('checkpoint_saved'):
                    checkpoints.append({
                        'epoch': r.get('epoch'),
                        'step': r.get('step'),
                        'path': metrics.get('checkpoint_path'),
                        'created_at': r.get('created_at'),
                        'metrics': {
                            'loss': r.get('loss'),
                            'accuracy': r.get('accuracy')
                        }
                    })
            return checkpoints
        
        try:
            from backend.schemas.training_models import TrainingProgress
            
            with self._db_service.get_session() as db:
                records = db.query(TrainingProgress).filter(
                    TrainingProgress.session_id == session_id
                ).all()
                
                for r in records:
                    if r.metrics and isinstance(r.metrics, dict):
                        if r.metrics.get('checkpoint_saved'):
                            checkpoints.append({
                                'epoch': r.epoch,
                                'step': r.step,
                                'path': r.metrics.get('checkpoint_path'),
                                'created_at': r.created_at.isoformat() if r.created_at else None,
                                'metrics': {
                                    'loss': r.loss,
                                    'accuracy': r.accuracy
                                }
                            })
                
                return checkpoints
                
        except Exception as e:
            logger.error(f"Failed to get checkpoints: {e}")
            return []
    
    # ==================== 日志相关 ====================
    
    def get_progress_with_logs(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[Any]:
        """
        获取包含日志信息的进度记录
        
        Args:
            session_id: 会话ID
            limit: 数量限制
            
        Returns:
            进度记录列表
        """
        if self._use_memory_storage:
            records = self._progress_records.get(session_id, [])
            sorted_records = sorted(
                records,
                key=lambda x: x.get('created_at', datetime.min),
                reverse=True
            )
            return sorted_records[:limit]
        
        try:
            from backend.schemas.training_models import TrainingProgress
            
            with self._db_service.get_session() as db:
                return db.query(TrainingProgress).filter(
                    TrainingProgress.session_id == session_id
                ).order_by(TrainingProgress.created_at.desc()).limit(limit).all()
                
        except Exception as e:
            logger.error(f"Failed to get progress with logs: {e}")
            return []
    
    # ==================== 批量操作 ====================
    
    def batch_create_progress(
        self,
        progress_list: List[Dict[str, Any]]
    ) -> int:
        """
        批量创建进度记录
        
        Args:
            progress_list: 进度数据列表
            
        Returns:
            成功创建的数量
        """
        if self._use_memory_storage:
            count = 0
            for progress_data in progress_list:
                if self.create_progress(progress_data):
                    count += 1
            return count
        
        try:
            from backend.schemas.training_models import TrainingProgress
            
            with self._db_service.get_session() as db:
                progress_objects = []
                for data in progress_list:
                    progress = TrainingProgress(
                        id=str(uuid.uuid4()),
                        session_id=data.get('session_id'),
                        stage=data.get('stage', 'training'),
                        epoch=data.get('epoch'),
                        step=data.get('step'),
                        total_steps=data.get('total_steps'),
                        loss=data.get('loss'),
                        accuracy=data.get('accuracy'),
                        learning_rate=data.get('learning_rate'),
                        metrics=data.get('metrics'),
                        gpu_utilization=data.get('gpu_utilization'),
                        gpu_memory_used=data.get('gpu_memory_used'),
                        gpu_memory_total=data.get('gpu_memory_total'),
                        gpu_temperature=data.get('gpu_temperature'),
                        gpu_power_draw=data.get('gpu_power_draw'),
                        cpu_utilization=data.get('cpu_utilization'),
                        cpu_memory_used=data.get('cpu_memory_used'),
                        cpu_memory_total=data.get('cpu_memory_total'),
                        cpu_temperature=data.get('cpu_temperature'),
                        samples_per_second=data.get('samples_per_second'),
                        tokens_per_second=data.get('tokens_per_second'),
                        batch_size=data.get('batch_size'),
                        gradient_norm=data.get('gradient_norm'),
                        disk_read_speed=data.get('disk_read_speed'),
                        disk_write_speed=data.get('disk_write_speed'),
                        disk_utilization=data.get('disk_utilization'),
                        network_download_speed=data.get('network_download_speed'),
                        network_upload_speed=data.get('network_upload_speed'),
                        network_latency=data.get('network_latency')
                    )
                    progress_objects.append(progress)
                
                db.bulk_save_objects(progress_objects)
                db.commit()
                
                return len(progress_objects)
                
        except Exception as e:
            logger.error(f"Failed to batch create progress: {e}")
            return 0
    
    def delete_progress_by_session(
        self,
        session_id: str
    ) -> int:
        """
        删除会话的所有进度记录
        
        Args:
            session_id: 会话ID
            
        Returns:
            删除的记录数
        """
        if self._use_memory_storage:
            records = self._progress_records.pop(session_id, [])
            return len(records)
        
        try:
            from backend.schemas.training_models import TrainingProgress
            
            with self._db_service.get_session() as db:
                deleted = db.query(TrainingProgress).filter(
                    TrainingProgress.session_id == session_id
                ).delete()
                
                db.commit()
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to delete progress by session: {e}")
            return 0
    
    # ==================== 统计查询 ====================
    
    def count_progress_records(
        self,
        session_id: str
    ) -> int:
        """
        统计进度记录数量
        
        Args:
            session_id: 会话ID
            
        Returns:
            记录数量
        """
        if self._use_memory_storage:
            return len(self._progress_records.get(session_id, []))
        
        try:
            from backend.schemas.training_models import TrainingProgress
            
            with self._db_service.get_session() as db:
                return db.query(TrainingProgress).filter(
                    TrainingProgress.session_id == session_id
                ).count()
                
        except Exception as e:
            logger.error(f"Failed to count progress records: {e}")
            return 0
    
    def get_session_log_path(
        self,
        session_id: str,
        user_id: str = None
    ) -> Optional[str]:
        """
        获取会话的日志文件路径
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            日志文件路径或None
        """
        if self._use_memory_storage:
            session = self._sessions.get(session_id)
            if session:
                config = session.get('config', {})
                result = session.get('result', {})
                return (
                    config.get('log_path') or
                    config.get('logPath') or
                    result.get('log_path') or
                    result.get('logPath')
                )
            return None
        
        try:
            from backend.schemas.training_models import TrainingSession
            
            with self._db_service.get_session() as db:
                query = db.query(TrainingSession).filter(
                    TrainingSession.session_id == session_id
                )
                
                if user_id:
                    query = query.filter(TrainingSession.user_id == user_id)
                
                session = query.first()
                
                if not session:
                    return None
                
                config = session.config or {}
                result = session.result or {}
                
                return (
                    config.get('log_path') or
                    config.get('logPath') or
                    result.get('log_path') or
                    result.get('logPath')
                )
                
        except Exception as e:
            logger.error(f"Failed to get session log path: {e}")
            return None


# ==================== 全局仓库实例 ====================

_global_training_progress_repository: Optional[TrainingProgressRepository] = None


def get_training_progress_repository(
    use_memory_storage: bool = False
) -> TrainingProgressRepository:
    """
    获取训练进度仓库实例
    
    Args:
        use_memory_storage: 是否使用内存存储
        
    Returns:
        TrainingProgressRepository 实例
    """
    global _global_training_progress_repository
    
    if _global_training_progress_repository is None:
        _global_training_progress_repository = TrainingProgressRepository(
            use_memory_storage=use_memory_storage
        )
    
    return _global_training_progress_repository

