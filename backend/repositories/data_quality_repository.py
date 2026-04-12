"""数据质量仓库层

实现数据质量评估记录、问题记录、清理记录等的持久化操作。
支持内存存储和数据库持久化两种模式。
"""

import sys
import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from contextlib import contextmanager

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.modules.dataset.dataset_exceptions import (
    QualityAssessmentNotFoundError,
    DataCleaningError,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 质量评估仓库
# ============================================================================

class QualityAssessmentRepository:
    """质量评估仓库
    
    管理质量评估记录的CRUD操作。
    支持内存存储和数据库持久化两种模式。
    """
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        """初始化仓库
        
        Args:
            db_service: 数据库服务实例
            use_memory_storage: 是否使用内存存储
        """
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._assessments: Dict[str, Dict[str, Any]] = {}
        
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
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    @contextmanager
    def _get_session(self):
        """获取数据库会话"""
        if self._db_service:
            with self._db_service.get_db_session() as session:
                yield session
        else:
            yield None
    
    def create(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建评估记录
        
        Args:
            assessment_data: 评估记录数据字典
            
        Returns:
            创建的评估记录数据
        """
        if not assessment_data.get('assessment_id'):
            assessment_data['assessment_id'] = str(uuid.uuid4())
        assessment_data['created_at'] = datetime.utcnow()
        assessment_data['assessed_at'] = datetime.utcnow()
        
        if self._use_memory_storage:
            self._assessments[assessment_data['assessment_id']] = assessment_data.copy()
            logger.info(f"Created quality assessment (memory): {assessment_data['assessment_id']}")
            return assessment_data
        
        try:
            from backend.schemas.data_quality_db_models import QualityAssessment
            
            with self._get_session() as session:
                if session:
                    db_assessment = QualityAssessment(
                        id=uuid.UUID(assessment_data['assessment_id']),
                        dataset_id=assessment_data.get('dataset_id', ''),
                        user_id=assessment_data.get('user_id', ''),
                        tenant_id=assessment_data.get('tenant_id'),
                        overall_score=assessment_data.get('overall_score', 0.0),
                        dimension_scores=assessment_data.get('dimension_scores', {}),
                        column_metrics=assessment_data.get('column_metrics', []),
                        total_records=assessment_data.get('total_records', 0),
                        total_columns=assessment_data.get('total_columns', 0),
                        missing_values_count=assessment_data.get('missing_values_count', 0),
                        missing_values_rate=assessment_data.get('missing_values_rate', 0.0),
                        duplicate_records_count=assessment_data.get('duplicate_records_count', 0),
                        duplicate_records_rate=assessment_data.get('duplicate_records_rate', 0.0),
                        outliers_count=assessment_data.get('outliers_count', 0),
                        outliers_rate=assessment_data.get('outliers_rate', 0.0),
                        status=assessment_data.get('status', 'completed'),
                        error_message=assessment_data.get('error_message'),
                        metadata_=assessment_data.get('metadata', {}),
                    )
                    session.add(db_assessment)
                    session.flush()
                    logger.info(f"Created quality assessment (db): {assessment_data['assessment_id']}")
                    return db_assessment.to_dict()
        except Exception as e:
            logger.error(f"Failed to create quality assessment in database: {e}")
            self._assessments[assessment_data['assessment_id']] = assessment_data.copy()
        
        return assessment_data
    
    def get_by_id(self, assessment_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取评估记录
        
        Args:
            assessment_id: 评估记录ID
            
        Returns:
            评估记录数据
        """
        if self._use_memory_storage:
            return self._assessments.get(assessment_id)
        
        try:
            from backend.schemas.data_quality_db_models import QualityAssessment
            
            with self._get_session() as session:
                if session:
                    db_assessment = session.query(QualityAssessment).filter(
                        QualityAssessment.id == uuid.UUID(assessment_id)
                    ).first()
                    if db_assessment:
                        return db_assessment.to_dict()
        except Exception as e:
            logger.error(f"Failed to get quality assessment from database: {e}")
            return self._assessments.get(assessment_id)
        
        return None
    
    def get_by_dataset(
        self,
        dataset_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取数据集的评估记录
        
        Args:
            dataset_id: 数据集ID
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            评估记录列表和总数
        """
        if self._use_memory_storage:
            filtered = [
                a for a in self._assessments.values()
                if a.get('dataset_id') == dataset_id
            ]
            filtered.sort(key=lambda x: x.get('created_at') or datetime.min, reverse=True)
            total = len(filtered)
            return filtered[offset:offset + limit], total
        
        try:
            from backend.schemas.data_quality_db_models import QualityAssessment
            from sqlalchemy import desc
            
            with self._get_session() as session:
                if session:
                    query = session.query(QualityAssessment).filter(
                        QualityAssessment.dataset_id == dataset_id
                    )
                    total = query.count()
                    assessments = query.order_by(desc(QualityAssessment.assessed_at)).offset(offset).limit(limit).all()
                    return [a.to_dict() for a in assessments], total
        except Exception as e:
            logger.error(f"Failed to get quality assessments from database: {e}")
        
        return [], 0
    
    def get_latest_by_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """获取数据集最新的评估记录
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            最新的评估记录数据
        """
        if self._use_memory_storage:
            assessments = [
                a for a in self._assessments.values()
                if a.get('dataset_id') == dataset_id
            ]
            if not assessments:
                return None
            return max(assessments, key=lambda x: x.get('created_at') or datetime.min)
        
        try:
            from backend.schemas.data_quality_db_models import QualityAssessment
            from sqlalchemy import desc
            
            with self._get_session() as session:
                if session:
                    db_assessment = session.query(QualityAssessment).filter(
                        QualityAssessment.dataset_id == dataset_id
                    ).order_by(desc(QualityAssessment.assessed_at)).first()
                    if db_assessment:
                        return db_assessment.to_dict()
        except Exception as e:
            logger.error(f"Failed to get latest quality assessment from database: {e}")
        
        return None
    
    def update(self, assessment_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新评估记录
        
        Args:
            assessment_id: 评估记录ID
            update_data: 更新数据
            
        Returns:
            更新后的评估记录数据
        """
        if self._use_memory_storage:
            if assessment_id not in self._assessments:
                raise QualityAssessmentNotFoundError(assessment_id)
            self._assessments[assessment_id].update(update_data)
            logger.info(f"Updated quality assessment (memory): {assessment_id}")
            return self._assessments[assessment_id]
        
        try:
            from backend.schemas.data_quality_db_models import QualityAssessment
            
            with self._get_session() as session:
                if session:
                    db_assessment = session.query(QualityAssessment).filter(
                        QualityAssessment.id == uuid.UUID(assessment_id)
                    ).first()
                    if not db_assessment:
                        raise QualityAssessmentNotFoundError(assessment_id)
                    
                    for key, value in update_data.items():
                        if key == 'metadata':
                            setattr(db_assessment, 'metadata_', value)
                        elif hasattr(db_assessment, key):
                            setattr(db_assessment, key, value)
                    
                    session.flush()
                    logger.info(f"Updated quality assessment (db): {assessment_id}")
                    return db_assessment.to_dict()
        except QualityAssessmentNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update quality assessment in database: {e}")
        
        return None
    
    def delete(self, assessment_id: str) -> bool:
        """删除评估记录
        
        Args:
            assessment_id: 评估记录ID
            
        Returns:
            是否删除成功
        """
        if self._use_memory_storage:
            if assessment_id in self._assessments:
                del self._assessments[assessment_id]
                logger.info(f"Deleted quality assessment (memory): {assessment_id}")
                return True
            return False
        
        try:
            from backend.schemas.data_quality_db_models import QualityAssessment
            
            with self._get_session() as session:
                if session:
                    db_assessment = session.query(QualityAssessment).filter(
                        QualityAssessment.id == uuid.UUID(assessment_id)
                    ).first()
                    if db_assessment:
                        session.delete(db_assessment)
                        logger.info(f"Deleted quality assessment (db): {assessment_id}")
                        return True
        except Exception as e:
            logger.error(f"Failed to delete quality assessment from database: {e}")
        
        return False


# ============================================================================
# 质量问题仓库
# ============================================================================

class QualityIssueRepository:
    """质量问题仓库
    
    管理质量问题记录的CRUD操作。
    """
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        """初始化仓库"""
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._issues: Dict[str, Dict[str, Any]] = {}
        
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
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    @contextmanager
    def _get_session(self):
        """获取数据库会话"""
        if self._db_service:
            with self._db_service.get_db_session() as session:
                yield session
        else:
            yield None
    
    def create(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建问题记录
        
        Args:
            issue_data: 问题记录数据字典
            
        Returns:
            创建的问题记录数据
        """
        if not issue_data.get('issue_id'):
            issue_data['issue_id'] = str(uuid.uuid4())
        issue_data['detected_at'] = datetime.utcnow()
        
        if self._use_memory_storage:
            self._issues[issue_data['issue_id']] = issue_data.copy()
            logger.info(f"Created quality issue (memory): {issue_data['issue_id']}")
            return issue_data
        
        try:
            from backend.schemas.data_quality_db_models import QualityIssue
            
            with self._get_session() as session:
                if session:
                    db_issue = QualityIssue(
                        id=uuid.UUID(issue_data['issue_id']),
                        dataset_id=issue_data.get('dataset_id', ''),
                        assessment_id=uuid.UUID(issue_data['assessment_id']) if issue_data.get('assessment_id') else None,
                        user_id=issue_data.get('user_id', ''),
                        tenant_id=issue_data.get('tenant_id'),
                        issue_type=issue_data.get('issue_type', ''),
                        severity=issue_data.get('severity', 'medium'),
                        column_name=issue_data.get('column_name'),
                        description=issue_data.get('description', ''),
                        affected_count=issue_data.get('affected_count', 0),
                        affected_rate=issue_data.get('affected_rate', 0.0),
                        sample_values=issue_data.get('sample_values', []),
                        recommendation=issue_data.get('recommendation', ''),
                        auto_fixable=issue_data.get('auto_fixable', False),
                        status=issue_data.get('status', 'open'),
                    )
                    session.add(db_issue)
                    session.flush()
                    logger.info(f"Created quality issue (db): {issue_data['issue_id']}")
                    return db_issue.to_dict()
        except Exception as e:
            logger.error(f"Failed to create quality issue in database: {e}")
            self._issues[issue_data['issue_id']] = issue_data.copy()
        
        return issue_data
    
    def create_batch(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量创建问题记录
        
        Args:
            issues: 问题记录数据列表
            
        Returns:
            创建的问题记录数据列表
        """
        created = []
        for issue in issues:
            created.append(self.create(issue))
        return created
    
    def get_by_id(self, issue_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取问题记录
        
        Args:
            issue_id: 问题记录ID
            
        Returns:
            问题记录数据
        """
        if self._use_memory_storage:
            return self._issues.get(issue_id)
        
        try:
            from backend.schemas.data_quality_db_models import QualityIssue
            
            with self._get_session() as session:
                if session:
                    db_issue = session.query(QualityIssue).filter(
                        QualityIssue.id == uuid.UUID(issue_id)
                    ).first()
                    if db_issue:
                        return db_issue.to_dict()
        except Exception as e:
            logger.error(f"Failed to get quality issue from database: {e}")
            return self._issues.get(issue_id)
        
        return None
    
    def get_by_dataset(
        self,
        dataset_id: str,
        status_filter: Optional[List[str]] = None,
        severity_filter: Optional[List[str]] = None,
        issue_type_filter: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取数据集的问题记录
        
        Args:
            dataset_id: 数据集ID
            status_filter: 状态过滤
            severity_filter: 严重程度过滤
            issue_type_filter: 问题类型过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            问题记录列表和总数
        """
        if self._use_memory_storage:
            filtered = []
            for issue in self._issues.values():
                if issue.get('dataset_id') != dataset_id:
                    continue
                if status_filter and issue.get('status') not in status_filter:
                    continue
                if severity_filter and issue.get('severity') not in severity_filter:
                    continue
                if issue_type_filter and issue.get('issue_type') not in issue_type_filter:
                    continue
                filtered.append(issue)
            
            # 按严重程度和时间排序
            severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'info': 4}
            filtered.sort(key=lambda x: (severity_order.get(x.get('severity', 'medium'), 5), -(x.get('detected_at') or datetime.min).timestamp()))
            
            total = len(filtered)
            return filtered[offset:offset + limit], total
        
        try:
            from backend.schemas.data_quality_db_models import QualityIssue
            from sqlalchemy import desc, case
            
            with self._get_session() as session:
                if session:
                    query = session.query(QualityIssue).filter(
                        QualityIssue.dataset_id == dataset_id
                    )
                    if status_filter:
                        query = query.filter(QualityIssue.status.in_(status_filter))
                    if severity_filter:
                        query = query.filter(QualityIssue.severity.in_(severity_filter))
                    if issue_type_filter:
                        query = query.filter(QualityIssue.issue_type.in_(issue_type_filter))
                    
                    total = query.count()
                    
                    # 按严重程度排序
                    severity_order = case(
                        (QualityIssue.severity == 'critical', 0),
                        (QualityIssue.severity == 'high', 1),
                        (QualityIssue.severity == 'medium', 2),
                        (QualityIssue.severity == 'low', 3),
                        (QualityIssue.severity == 'info', 4),
                        else_=5
                    )
                    
                    issues = query.order_by(severity_order, desc(QualityIssue.detected_at)).offset(offset).limit(limit).all()
                    return [i.to_dict() for i in issues], total
        except Exception as e:
            logger.error(f"Failed to get quality issues from database: {e}")
        
        return [], 0
    
    def get_by_assessment(self, assessment_id: str) -> List[Dict[str, Any]]:
        """获取评估相关的问题记录
        
        Args:
            assessment_id: 评估记录ID
            
        Returns:
            问题记录列表
        """
        if self._use_memory_storage:
            return [
                issue for issue in self._issues.values()
                if issue.get('assessment_id') == assessment_id
            ]
        
        try:
            from backend.schemas.data_quality_db_models import QualityIssue
            
            with self._get_session() as session:
                if session:
                    issues = session.query(QualityIssue).filter(
                        QualityIssue.assessment_id == uuid.UUID(assessment_id)
                    ).all()
                    return [i.to_dict() for i in issues]
        except Exception as e:
            logger.error(f"Failed to get quality issues from database: {e}")
        
        return []
    
    def update_status(
        self,
        issue_id: str,
        status: str,
        resolved_by: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """更新问题状态
        
        Args:
            issue_id: 问题记录ID
            status: 新状态
            resolved_by: 解决者ID
            
        Returns:
            更新后的问题记录数据
        """
        if self._use_memory_storage:
            issue = self._issues.get(issue_id)
            if not issue:
                return None
            
            issue['status'] = status
            if status == 'resolved':
                issue['resolved_at'] = datetime.utcnow()
                issue['resolved_by'] = resolved_by
            
            return issue
        
        try:
            from backend.schemas.data_quality_db_models import QualityIssue
            
            with self._get_session() as session:
                if session:
                    db_issue = session.query(QualityIssue).filter(
                        QualityIssue.id == uuid.UUID(issue_id)
                    ).first()
                    if not db_issue:
                        return None
                    
                    db_issue.status = status
                    if status == 'resolved':
                        db_issue.resolved_at = datetime.utcnow()
                        db_issue.resolved_by = resolved_by
                    
                    session.flush()
                    return db_issue.to_dict()
        except Exception as e:
            logger.error(f"Failed to update quality issue status in database: {e}")
        
        return None
    
    def delete(self, issue_id: str) -> bool:
        """删除问题记录
        
        Args:
            issue_id: 问题记录ID
            
        Returns:
            是否删除成功
        """
        if self._use_memory_storage:
            if issue_id in self._issues:
                del self._issues[issue_id]
                logger.info(f"Deleted quality issue (memory): {issue_id}")
                return True
            return False
        
        try:
            from backend.schemas.data_quality_db_models import QualityIssue
            
            with self._get_session() as session:
                if session:
                    db_issue = session.query(QualityIssue).filter(
                        QualityIssue.id == uuid.UUID(issue_id)
                    ).first()
                    if db_issue:
                        session.delete(db_issue)
                        logger.info(f"Deleted quality issue (db): {issue_id}")
                        return True
        except Exception as e:
            logger.error(f"Failed to delete quality issue from database: {e}")
        
        return False
    
    def delete_by_dataset(self, dataset_id: str) -> int:
        """删除数据集的所有问题记录
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            删除的记录数
        """
        if self._use_memory_storage:
            to_delete = [
                issue_id for issue_id, issue in self._issues.items()
                if issue.get('dataset_id') == dataset_id
            ]
            for issue_id in to_delete:
                del self._issues[issue_id]
            logger.info(f"Deleted {len(to_delete)} quality issues (memory) for dataset: {dataset_id}")
            return len(to_delete)
        
        try:
            from backend.schemas.data_quality_db_models import QualityIssue
            
            with self._get_session() as session:
                if session:
                    count = session.query(QualityIssue).filter(
                        QualityIssue.dataset_id == dataset_id
                    ).delete()
                    logger.info(f"Deleted {count} quality issues (db) for dataset: {dataset_id}")
                    return count
        except Exception as e:
            logger.error(f"Failed to delete quality issues from database: {e}")
        
        return 0


# ============================================================================
# 清理记录仓库
# ============================================================================

class CleaningRecordRepository:
    """清理记录仓库
    
    管理数据清理记录的CRUD操作。
    """
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        """初始化仓库"""
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._records: Dict[str, Dict[str, Any]] = {}
        
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
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    @contextmanager
    def _get_session(self):
        """获取数据库会话"""
        if self._db_service:
            with self._db_service.get_db_session() as session:
                yield session
        else:
            yield None
    
    def create(self, record_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建清理记录
        
        Args:
            record_data: 清理记录数据字典
            
        Returns:
            创建的清理记录数据
        """
        if not record_data.get('cleaning_id'):
            record_data['cleaning_id'] = str(uuid.uuid4())
        record_data['created_at'] = datetime.utcnow()
        
        if self._use_memory_storage:
            self._records[record_data['cleaning_id']] = record_data.copy()
            logger.info(f"Created cleaning record (memory): {record_data['cleaning_id']}")
            return record_data
        
        try:
            from backend.schemas.data_quality_db_models import CleaningRecord
            
            with self._get_session() as session:
                if session:
                    db_record = CleaningRecord(
                        id=uuid.UUID(record_data['cleaning_id']),
                        dataset_id=record_data.get('dataset_id', ''),
                        user_id=record_data.get('user_id', ''),
                        tenant_id=record_data.get('tenant_id'),
                        config=record_data.get('config', {}),
                        status=record_data.get('status', 'pending'),
                        original_dataset_id=record_data.get('original_dataset_id', ''),
                        cleaned_dataset_id=record_data.get('cleaned_dataset_id'),
                        original_record_count=record_data.get('original_record_count', 0),
                        cleaned_record_count=record_data.get('cleaned_record_count', 0),
                        total_records_affected=record_data.get('total_records_affected', 0),
                        operation_results=record_data.get('operation_results', []),
                        original_quality_score=record_data.get('original_quality_score', 0.0),
                        cleaned_quality_score=record_data.get('cleaned_quality_score', 0.0),
                        improvement=record_data.get('improvement', 0.0),
                        backup_path=record_data.get('backup_path'),
                        error_message=record_data.get('error_message'),
                        started_at=record_data.get('started_at'),
                        completed_at=record_data.get('completed_at'),
                        execution_time_ms=record_data.get('execution_time_ms', 0.0),
                    )
                    session.add(db_record)
                    session.flush()
                    logger.info(f"Created cleaning record (db): {record_data['cleaning_id']}")
                    return db_record.to_dict()
        except Exception as e:
            logger.error(f"Failed to create cleaning record in database: {e}")
            self._records[record_data['cleaning_id']] = record_data.copy()
        
        return record_data
    
    def get_by_id(self, cleaning_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取清理记录
        
        Args:
            cleaning_id: 清理记录ID
            
        Returns:
            清理记录数据
        """
        if self._use_memory_storage:
            return self._records.get(cleaning_id)
        
        try:
            from backend.schemas.data_quality_db_models import CleaningRecord
            
            with self._get_session() as session:
                if session:
                    db_record = session.query(CleaningRecord).filter(
                        CleaningRecord.id == uuid.UUID(cleaning_id)
                    ).first()
                    if db_record:
                        return db_record.to_dict()
        except Exception as e:
            logger.error(f"Failed to get cleaning record from database: {e}")
            return self._records.get(cleaning_id)
        
        return None
    
    def get_by_dataset(
        self,
        dataset_id: str,
        status_filter: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取数据集的清理记录
        
        Args:
            dataset_id: 数据集ID
            status_filter: 状态过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            清理记录列表和总数
        """
        if self._use_memory_storage:
            filtered = []
            for record in self._records.values():
                if record.get('dataset_id') != dataset_id:
                    continue
                if status_filter and record.get('status') not in status_filter:
                    continue
                filtered.append(record)
            
            filtered.sort(key=lambda x: x.get('created_at') or datetime.min, reverse=True)
            total = len(filtered)
            return filtered[offset:offset + limit], total
        
        try:
            from backend.schemas.data_quality_db_models import CleaningRecord
            from sqlalchemy import desc
            
            with self._get_session() as session:
                if session:
                    query = session.query(CleaningRecord).filter(
                        CleaningRecord.dataset_id == dataset_id
                    )
                    if status_filter:
                        query = query.filter(CleaningRecord.status.in_(status_filter))
                    
                    total = query.count()
                    records = query.order_by(desc(CleaningRecord.created_at)).offset(offset).limit(limit).all()
                    return [r.to_dict() for r in records], total
        except Exception as e:
            logger.error(f"Failed to get cleaning records from database: {e}")
        
        return [], 0
    
    def update(self, cleaning_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新清理记录
        
        Args:
            cleaning_id: 清理记录ID
            update_data: 更新数据
            
        Returns:
            更新后的清理记录数据
        """
        if self._use_memory_storage:
            if cleaning_id not in self._records:
                raise DataCleaningError('', 'update', '清理记录不存在')
            self._records[cleaning_id].update(update_data)
            logger.info(f"Updated cleaning record (memory): {cleaning_id}")
            return self._records[cleaning_id]
        
        try:
            from backend.schemas.data_quality_db_models import CleaningRecord
            
            with self._get_session() as session:
                if session:
                    db_record = session.query(CleaningRecord).filter(
                        CleaningRecord.id == uuid.UUID(cleaning_id)
                    ).first()
                    if not db_record:
                        raise DataCleaningError('', 'update', '清理记录不存在')
                    
                    for key, value in update_data.items():
                        if hasattr(db_record, key):
                            setattr(db_record, key, value)
                    
                    session.flush()
                    logger.info(f"Updated cleaning record (db): {cleaning_id}")
                    return db_record.to_dict()
        except DataCleaningError:
            raise
        except Exception as e:
            logger.error(f"Failed to update cleaning record in database: {e}")
        
        return None
    
    def update_status(
        self,
        cleaning_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """更新清理状态
        
        Args:
            cleaning_id: 清理记录ID
            status: 新状态
            error_message: 错误信息
            
        Returns:
            更新后的清理记录数据
        """
        update_data = {'status': status}
        if error_message:
            update_data['error_message'] = error_message
        
        if status == 'in_progress':
            update_data['started_at'] = datetime.utcnow()
        elif status in ('completed', 'failed', 'rolled_back'):
            update_data['completed_at'] = datetime.utcnow()
        
        return self.update(cleaning_id, update_data)
    
    def delete(self, cleaning_id: str) -> bool:
        """删除清理记录
        
        Args:
            cleaning_id: 清理记录ID
            
        Returns:
            是否删除成功
        """
        if self._use_memory_storage:
            if cleaning_id in self._records:
                del self._records[cleaning_id]
                logger.info(f"Deleted cleaning record (memory): {cleaning_id}")
                return True
            return False
        
        try:
            from backend.schemas.data_quality_db_models import CleaningRecord
            
            with self._get_session() as session:
                if session:
                    db_record = session.query(CleaningRecord).filter(
                        CleaningRecord.id == uuid.UUID(cleaning_id)
                    ).first()
                    if db_record:
                        session.delete(db_record)
                        logger.info(f"Deleted cleaning record (db): {cleaning_id}")
                        return True
        except Exception as e:
            logger.error(f"Failed to delete cleaning record from database: {e}")
        
        return False


# ============================================================================
# 质量规则仓库
# ============================================================================

class QualityRuleRepository:
    """质量规则仓库
    
    管理质量规则的CRUD操作。
    """
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        """初始化仓库"""
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._rules: Dict[str, Dict[str, Any]] = {}
        
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
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    @contextmanager
    def _get_session(self):
        """获取数据库会话"""
        if self._db_service:
            with self._db_service.get_db_session() as session:
                yield session
        else:
            yield None
    
    def create(self, rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建质量规则
        
        Args:
            rule_data: 质量规则数据字典
            
        Returns:
            创建的质量规则数据
        """
        if not rule_data.get('rule_id'):
            rule_data['rule_id'] = str(uuid.uuid4())
        rule_data['created_at'] = datetime.utcnow()
        
        if self._use_memory_storage:
            self._rules[rule_data['rule_id']] = rule_data.copy()
            logger.info(f"Created quality rule (memory): {rule_data['rule_id']}")
            return rule_data
        
        try:
            from backend.schemas.data_quality_db_models import QualityRule
            
            with self._get_session() as session:
                if session:
                    db_rule = QualityRule(
                        id=uuid.UUID(rule_data['rule_id']),
                        user_id=rule_data.get('user_id', ''),
                        tenant_id=rule_data.get('tenant_id'),
                        name=rule_data.get('name', ''),
                        description=rule_data.get('description', ''),
                        rule_type=rule_data.get('rule_type', ''),
                        target_column=rule_data.get('target_column'),
                        condition=rule_data.get('condition', ''),
                        parameters=rule_data.get('parameters', {}),
                        severity=rule_data.get('severity', 'medium'),
                        enabled=rule_data.get('enabled', True),
                        dataset_ids=rule_data.get('dataset_ids', []),
                    )
                    session.add(db_rule)
                    session.flush()
                    logger.info(f"Created quality rule (db): {rule_data['rule_id']}")
                    return db_rule.to_dict()
        except Exception as e:
            logger.error(f"Failed to create quality rule in database: {e}")
            self._rules[rule_data['rule_id']] = rule_data.copy()
        
        return rule_data
    
    def get_by_id(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取质量规则
        
        Args:
            rule_id: 质量规则ID
            
        Returns:
            质量规则数据
        """
        if self._use_memory_storage:
            return self._rules.get(rule_id)
        
        try:
            from backend.schemas.data_quality_db_models import QualityRule
            
            with self._get_session() as session:
                if session:
                    db_rule = session.query(QualityRule).filter(
                        QualityRule.id == uuid.UUID(rule_id)
                    ).first()
                    if db_rule:
                        return db_rule.to_dict()
        except Exception as e:
            logger.error(f"Failed to get quality rule from database: {e}")
            return self._rules.get(rule_id)
        
        return None
    
    def get_by_user(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        enabled_only: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取用户的质量规则
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            enabled_only: 是否只返回启用的规则
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            质量规则列表和总数
        """
        if self._use_memory_storage:
            filtered = []
            for rule in self._rules.values():
                if rule.get('user_id') != user_id:
                    continue
                if tenant_id is not None and rule.get('tenant_id') != tenant_id:
                    continue
                if enabled_only and not rule.get('enabled', True):
                    continue
                filtered.append(rule)
            
            filtered.sort(key=lambda x: x.get('created_at') or datetime.min, reverse=True)
            total = len(filtered)
            return filtered[offset:offset + limit], total
        
        try:
            from backend.schemas.data_quality_db_models import QualityRule
            from sqlalchemy import desc
            
            with self._get_session() as session:
                if session:
                    query = session.query(QualityRule).filter(
                        QualityRule.user_id == user_id
                    )
                    if tenant_id:
                        query = query.filter(QualityRule.tenant_id == tenant_id)
                    if enabled_only:
                        query = query.filter(QualityRule.enabled == True)
                    
                    total = query.count()
                    rules = query.order_by(desc(QualityRule.created_at)).offset(offset).limit(limit).all()
                    return [r.to_dict() for r in rules], total
        except Exception as e:
            logger.error(f"Failed to get quality rules from database: {e}")
        
        return [], 0
    
    def get_by_dataset(self, dataset_id: str) -> List[Dict[str, Any]]:
        """获取应用于数据集的质量规则
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            质量规则列表
        """
        if self._use_memory_storage:
            return [
                rule for rule in self._rules.values()
                if dataset_id in rule.get('dataset_ids', []) and rule.get('enabled', True)
            ]
        
        try:
            from backend.schemas.data_quality_db_models import QualityRule
            from sqlalchemy import func
            
            with self._get_session() as session:
                if session:
                    # 使用 JSON 包含查询
                    rules = session.query(QualityRule).filter(
                        QualityRule.enabled == True,
                        QualityRule.dataset_ids.contains([dataset_id])
                    ).all()
                    return [r.to_dict() for r in rules]
        except Exception as e:
            logger.error(f"Failed to get quality rules from database: {e}")
        
        return []
    
    def update(self, rule_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新质量规则
        
        Args:
            rule_id: 质量规则ID
            update_data: 更新数据
            
        Returns:
            更新后的质量规则数据
        """
        if self._use_memory_storage:
            if rule_id in self._rules:
                self._rules[rule_id].update(update_data)
                self._rules[rule_id]['updated_at'] = datetime.utcnow()
                logger.info(f"Updated quality rule (memory): {rule_id}")
                return self._rules[rule_id]
            return None
        
        try:
            from backend.schemas.data_quality_db_models import QualityRule
            
            with self._get_session() as session:
                if session:
                    db_rule = session.query(QualityRule).filter(
                        QualityRule.id == uuid.UUID(rule_id)
                    ).first()
                    if not db_rule:
                        return None
                    
                    for key, value in update_data.items():
                        if hasattr(db_rule, key):
                            setattr(db_rule, key, value)
                    
                    session.flush()
                    logger.info(f"Updated quality rule (db): {rule_id}")
                    return db_rule.to_dict()
        except Exception as e:
            logger.error(f"Failed to update quality rule in database: {e}")
        
        return None
    
    def delete(self, rule_id: str) -> bool:
        """删除质量规则
        
        Args:
            rule_id: 质量规则ID
            
        Returns:
            是否删除成功
        """
        if self._use_memory_storage:
            if rule_id in self._rules:
                del self._rules[rule_id]
                logger.info(f"Deleted quality rule (memory): {rule_id}")
                return True
            return False
        
        try:
            from backend.schemas.data_quality_db_models import QualityRule
            
            with self._get_session() as session:
                if session:
                    db_rule = session.query(QualityRule).filter(
                        QualityRule.id == uuid.UUID(rule_id)
                    ).first()
                    if db_rule:
                        session.delete(db_rule)
                        logger.info(f"Deleted quality rule (db): {rule_id}")
                        return True
        except Exception as e:
            logger.error(f"Failed to delete quality rule from database: {e}")
        
        return False


# ============================================================================
# 规则验证记录仓库
# ============================================================================

class RuleValidationRecordRepository:
    """规则验证记录仓库
    
    管理规则验证记录的CRUD操作。
    """
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        """初始化仓库"""
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._records: Dict[str, Dict[str, Any]] = {}
        
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
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    @contextmanager
    def _get_session(self):
        """获取数据库会话"""
        if self._db_service:
            with self._db_service.get_db_session() as session:
                yield session
        else:
            yield None
    
    def create(self, record_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建验证记录
        
        Args:
            record_data: 验证记录数据字典
            
        Returns:
            创建的验证记录数据
        """
        if not record_data.get('validation_id'):
            record_data['validation_id'] = str(uuid.uuid4())
        record_data['validated_at'] = datetime.utcnow()
        
        if self._use_memory_storage:
            self._records[record_data['validation_id']] = record_data.copy()
            logger.info(f"Created rule validation record (memory): {record_data['validation_id']}")
            return record_data
        
        try:
            from backend.schemas.data_quality_db_models import RuleValidationRecord
            
            with self._get_session() as session:
                if session:
                    db_record = RuleValidationRecord(
                        id=uuid.UUID(record_data['validation_id']),
                        dataset_id=record_data.get('dataset_id', ''),
                        user_id=record_data.get('user_id', ''),
                        tenant_id=record_data.get('tenant_id'),
                        total_rules=record_data.get('total_rules', 0),
                        passed_rules=record_data.get('passed_rules', 0),
                        failed_rules=record_data.get('failed_rules', 0),
                        pass_rate=record_data.get('pass_rate', 0.0),
                        results=record_data.get('results', []),
                    )
                    session.add(db_record)
                    session.flush()
                    logger.info(f"Created rule validation record (db): {record_data['validation_id']}")
                    return db_record.to_dict()
        except Exception as e:
            logger.error(f"Failed to create rule validation record in database: {e}")
            self._records[record_data['validation_id']] = record_data.copy()
        
        return record_data
    
    def get_by_id(self, validation_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取验证记录
        
        Args:
            validation_id: 验证记录ID
            
        Returns:
            验证记录数据
        """
        if self._use_memory_storage:
            return self._records.get(validation_id)
        
        try:
            from backend.schemas.data_quality_db_models import RuleValidationRecord
            
            with self._get_session() as session:
                if session:
                    db_record = session.query(RuleValidationRecord).filter(
                        RuleValidationRecord.id == uuid.UUID(validation_id)
                    ).first()
                    if db_record:
                        return db_record.to_dict()
        except Exception as e:
            logger.error(f"Failed to get rule validation record from database: {e}")
            return self._records.get(validation_id)
        
        return None
    
    def get_by_dataset(
        self,
        dataset_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取数据集的验证记录
        
        Args:
            dataset_id: 数据集ID
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            验证记录列表和总数
        """
        if self._use_memory_storage:
            filtered = [
                r for r in self._records.values()
                if r.get('dataset_id') == dataset_id
            ]
            filtered.sort(key=lambda x: x.get('validated_at') or datetime.min, reverse=True)
            total = len(filtered)
            return filtered[offset:offset + limit], total
        
        try:
            from backend.schemas.data_quality_db_models import RuleValidationRecord
            from sqlalchemy import desc
            
            with self._get_session() as session:
                if session:
                    query = session.query(RuleValidationRecord).filter(
                        RuleValidationRecord.dataset_id == dataset_id
                    )
                    total = query.count()
                    records = query.order_by(desc(RuleValidationRecord.validated_at)).offset(offset).limit(limit).all()
                    return [r.to_dict() for r in records], total
        except Exception as e:
            logger.error(f"Failed to get rule validation records from database: {e}")
        
        return [], 0
    
    def delete(self, validation_id: str) -> bool:
        """删除验证记录
        
        Args:
            validation_id: 验证记录ID
            
        Returns:
            是否删除成功
        """
        if self._use_memory_storage:
            if validation_id in self._records:
                del self._records[validation_id]
                logger.info(f"Deleted rule validation record (memory): {validation_id}")
                return True
            return False
        
        try:
            from backend.schemas.data_quality_db_models import RuleValidationRecord
            
            with self._get_session() as session:
                if session:
                    db_record = session.query(RuleValidationRecord).filter(
                        RuleValidationRecord.id == uuid.UUID(validation_id)
                    ).first()
                    if db_record:
                        session.delete(db_record)
                        logger.info(f"Deleted rule validation record (db): {validation_id}")
                        return True
        except Exception as e:
            logger.error(f"Failed to delete rule validation record from database: {e}")
        
        return False


# ============================================================================
# 质量报告仓库
# ============================================================================

class QualityReportRepository:
    """质量报告仓库
    
    管理质量报告的CRUD操作。
    """
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        """初始化仓库"""
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._reports: Dict[str, Dict[str, Any]] = {}
        
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
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    @contextmanager
    def _get_session(self):
        """获取数据库会话"""
        if self._db_service:
            with self._db_service.get_db_session() as session:
                yield session
        else:
            yield None
    
    def create(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建质量报告
        
        Args:
            report_data: 质量报告数据字典
            
        Returns:
            创建的质量报告数据
        """
        if not report_data.get('report_id'):
            report_data['report_id'] = str(uuid.uuid4())
        report_data['generated_at'] = datetime.utcnow()
        
        if self._use_memory_storage:
            self._reports[report_data['report_id']] = report_data.copy()
            logger.info(f"Created quality report (memory): {report_data['report_id']}")
            return report_data
        
        try:
            from backend.schemas.data_quality_db_models import QualityReport
            
            with self._get_session() as session:
                if session:
                    db_report = QualityReport(
                        id=uuid.UUID(report_data['report_id']),
                        dataset_id=report_data.get('dataset_id', ''),
                        user_id=report_data.get('user_id', ''),
                        tenant_id=report_data.get('tenant_id'),
                        dataset_name=report_data.get('dataset_name', ''),
                        report_content=report_data.get('report_content', {}),
                        summary=report_data.get('summary', ''),
                        recommendations=report_data.get('recommendations', []),
                        overall_score=report_data.get('overall_score', 0.0),
                    )
                    session.add(db_report)
                    session.flush()
                    logger.info(f"Created quality report (db): {report_data['report_id']}")
                    return db_report.to_dict()
        except Exception as e:
            logger.error(f"Failed to create quality report in database: {e}")
            self._reports[report_data['report_id']] = report_data.copy()
        
        return report_data
    
    def get_by_id(self, report_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取质量报告
        
        Args:
            report_id: 质量报告ID
            
        Returns:
            质量报告数据
        """
        if self._use_memory_storage:
            return self._reports.get(report_id)
        
        try:
            from backend.schemas.data_quality_db_models import QualityReport
            
            with self._get_session() as session:
                if session:
                    db_report = session.query(QualityReport).filter(
                        QualityReport.id == uuid.UUID(report_id)
                    ).first()
                    if db_report:
                        return db_report.to_dict()
        except Exception as e:
            logger.error(f"Failed to get quality report from database: {e}")
            return self._reports.get(report_id)
        
        return None
    
    def get_by_dataset(
        self,
        dataset_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取数据集的质量报告
        
        Args:
            dataset_id: 数据集ID
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            质量报告列表和总数
        """
        if self._use_memory_storage:
            filtered = [
                r for r in self._reports.values()
                if r.get('dataset_id') == dataset_id
            ]
            filtered.sort(key=lambda x: x.get('generated_at') or datetime.min, reverse=True)
            total = len(filtered)
            return filtered[offset:offset + limit], total
        
        try:
            from backend.schemas.data_quality_db_models import QualityReport
            from sqlalchemy import desc
            
            with self._get_session() as session:
                if session:
                    query = session.query(QualityReport).filter(
                        QualityReport.dataset_id == dataset_id
                    )
                    total = query.count()
                    reports = query.order_by(desc(QualityReport.generated_at)).offset(offset).limit(limit).all()
                    return [r.to_dict() for r in reports], total
        except Exception as e:
            logger.error(f"Failed to get quality reports from database: {e}")
        
        return [], 0
    
    def get_latest_by_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """获取数据集最新的质量报告
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            最新的质量报告数据
        """
        if self._use_memory_storage:
            reports = [
                r for r in self._reports.values()
                if r.get('dataset_id') == dataset_id
            ]
            if not reports:
                return None
            return max(reports, key=lambda x: x.get('generated_at') or datetime.min)
        
        try:
            from backend.schemas.data_quality_db_models import QualityReport
            from sqlalchemy import desc
            
            with self._get_session() as session:
                if session:
                    db_report = session.query(QualityReport).filter(
                        QualityReport.dataset_id == dataset_id
                    ).order_by(desc(QualityReport.generated_at)).first()
                    if db_report:
                        return db_report.to_dict()
        except Exception as e:
            logger.error(f"Failed to get latest quality report from database: {e}")
        
        return None
    
    def delete(self, report_id: str) -> bool:
        """删除质量报告
        
        Args:
            report_id: 质量报告ID
            
        Returns:
            是否删除成功
        """
        if self._use_memory_storage:
            if report_id in self._reports:
                del self._reports[report_id]
                logger.info(f"Deleted quality report (memory): {report_id}")
                return True
            return False
        
        try:
            from backend.schemas.data_quality_db_models import QualityReport
            
            with self._get_session() as session:
                if session:
                    db_report = session.query(QualityReport).filter(
                        QualityReport.id == uuid.UUID(report_id)
                    ).first()
                    if db_report:
                        session.delete(db_report)
                        logger.info(f"Deleted quality report (db): {report_id}")
                        return True
        except Exception as e:
            logger.error(f"Failed to delete quality report from database: {e}")
        
        return False


# ============================================================================
# 质量监控配置仓库
# ============================================================================

class QualityMonitoringConfigRepository:
    """质量监控配置仓库
    
    管理质量监控配置的CRUD操作。
    """
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        """初始化仓库"""
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._configs: Dict[str, Dict[str, Any]] = {}
        
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
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    @contextmanager
    def _get_session(self):
        """获取数据库会话"""
        if self._db_service:
            with self._db_service.get_db_session() as session:
                yield session
        else:
            yield None
    
    def create(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建监控配置
        
        Args:
            config_data: 监控配置数据字典
            
        Returns:
            创建的监控配置数据
        """
        if not config_data.get('config_id'):
            config_data['config_id'] = str(uuid.uuid4())
        config_data['created_at'] = datetime.utcnow()
        
        if self._use_memory_storage:
            self._configs[config_data['config_id']] = config_data.copy()
            logger.info(f"Created quality monitoring config (memory): {config_data['config_id']}")
            return config_data
        
        try:
            from backend.schemas.data_quality_db_models import QualityMonitoringConfig
            
            with self._get_session() as session:
                if session:
                    db_config = QualityMonitoringConfig(
                        id=uuid.UUID(config_data['config_id']),
                        dataset_id=config_data.get('dataset_id', ''),
                        user_id=config_data.get('user_id', ''),
                        tenant_id=config_data.get('tenant_id'),
                        enabled=config_data.get('enabled', True),
                        thresholds=config_data.get('thresholds', []),
                        check_interval_minutes=config_data.get('check_interval_minutes', 60),
                        alert_channels=config_data.get('alert_channels', []),
                        last_check_at=config_data.get('last_check_at'),
                        next_check_at=config_data.get('next_check_at'),
                    )
                    session.add(db_config)
                    session.flush()
                    logger.info(f"Created quality monitoring config (db): {config_data['config_id']}")
                    return db_config.to_dict()
        except Exception as e:
            logger.error(f"Failed to create quality monitoring config in database: {e}")
            self._configs[config_data['config_id']] = config_data.copy()
        
        return config_data
    
    def get_by_id(self, config_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取监控配置
        
        Args:
            config_id: 监控配置ID
            
        Returns:
            监控配置数据
        """
        if self._use_memory_storage:
            return self._configs.get(config_id)
        
        try:
            from backend.schemas.data_quality_db_models import QualityMonitoringConfig
            
            with self._get_session() as session:
                if session:
                    db_config = session.query(QualityMonitoringConfig).filter(
                        QualityMonitoringConfig.id == uuid.UUID(config_id)
                    ).first()
                    if db_config:
                        return db_config.to_dict()
        except Exception as e:
            logger.error(f"Failed to get quality monitoring config from database: {e}")
            return self._configs.get(config_id)
        
        return None
    
    def get_by_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """获取数据集的监控配置
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            监控配置数据
        """
        if self._use_memory_storage:
            for config in self._configs.values():
                if config.get('dataset_id') == dataset_id:
                    return config
            return None
        
        try:
            from backend.schemas.data_quality_db_models import QualityMonitoringConfig
            
            with self._get_session() as session:
                if session:
                    db_config = session.query(QualityMonitoringConfig).filter(
                        QualityMonitoringConfig.dataset_id == dataset_id
                    ).first()
                    if db_config:
                        return db_config.to_dict()
        except Exception as e:
            logger.error(f"Failed to get quality monitoring config from database: {e}")
        
        return None
    
    def get_active_configs(self) -> List[Dict[str, Any]]:
        """获取所有启用的监控配置
        
        Returns:
            启用的监控配置列表
        """
        if self._use_memory_storage:
            return [c for c in self._configs.values() if c.get('enabled', True)]
        
        try:
            from backend.schemas.data_quality_db_models import QualityMonitoringConfig
            
            with self._get_session() as session:
                if session:
                    configs = session.query(QualityMonitoringConfig).filter(
                        QualityMonitoringConfig.enabled == True
                    ).all()
                    return [c.to_dict() for c in configs]
        except Exception as e:
            logger.error(f"Failed to get active quality monitoring configs from database: {e}")
        
        return []
    
    def update(self, config_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新监控配置
        
        Args:
            config_id: 监控配置ID
            update_data: 更新数据
            
        Returns:
            更新后的监控配置数据
        """
        if self._use_memory_storage:
            if config_id in self._configs:
                self._configs[config_id].update(update_data)
                self._configs[config_id]['updated_at'] = datetime.utcnow()
                logger.info(f"Updated quality monitoring config (memory): {config_id}")
                return self._configs[config_id]
            return None
        
        try:
            from backend.schemas.data_quality_db_models import QualityMonitoringConfig
            
            with self._get_session() as session:
                if session:
                    db_config = session.query(QualityMonitoringConfig).filter(
                        QualityMonitoringConfig.id == uuid.UUID(config_id)
                    ).first()
                    if not db_config:
                        return None
                    
                    for key, value in update_data.items():
                        if hasattr(db_config, key):
                            setattr(db_config, key, value)
                    
                    session.flush()
                    logger.info(f"Updated quality monitoring config (db): {config_id}")
                    return db_config.to_dict()
        except Exception as e:
            logger.error(f"Failed to update quality monitoring config in database: {e}")
        
        return None
    
    def delete(self, config_id: str) -> bool:
        """删除监控配置
        
        Args:
            config_id: 监控配置ID
            
        Returns:
            是否删除成功
        """
        if self._use_memory_storage:
            if config_id in self._configs:
                del self._configs[config_id]
                logger.info(f"Deleted quality monitoring config (memory): {config_id}")
                return True
            return False
        
        try:
            from backend.schemas.data_quality_db_models import QualityMonitoringConfig
            
            with self._get_session() as session:
                if session:
                    db_config = session.query(QualityMonitoringConfig).filter(
                        QualityMonitoringConfig.id == uuid.UUID(config_id)
                    ).first()
                    if db_config:
                        session.delete(db_config)
                        logger.info(f"Deleted quality monitoring config (db): {config_id}")
                        return True
        except Exception as e:
            logger.error(f"Failed to delete quality monitoring config from database: {e}")
        
        return False


# ============================================================================
# 质量告警仓库
# ============================================================================

class QualityAlertRepository:
    """质量告警仓库
    
    管理质量告警的CRUD操作。
    """
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        """初始化仓库"""
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._alerts: Dict[str, Dict[str, Any]] = {}
        
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
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    @contextmanager
    def _get_session(self):
        """获取数据库会话"""
        if self._db_service:
            with self._db_service.get_db_session() as session:
                yield session
        else:
            yield None
    
    def create(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建质量告警
        
        Args:
            alert_data: 质量告警数据字典
            
        Returns:
            创建的质量告警数据
        """
        if not alert_data.get('alert_id'):
            alert_data['alert_id'] = str(uuid.uuid4())
        alert_data['triggered_at'] = datetime.utcnow()
        
        if self._use_memory_storage:
            self._alerts[alert_data['alert_id']] = alert_data.copy()
            logger.info(f"Created quality alert (memory): {alert_data['alert_id']}")
            return alert_data
        
        try:
            from backend.schemas.data_quality_db_models import QualityAlert
            
            with self._get_session() as session:
                if session:
                    db_alert = QualityAlert(
                        id=uuid.UUID(alert_data['alert_id']),
                        dataset_id=alert_data.get('dataset_id', ''),
                        config_id=uuid.UUID(alert_data['config_id']) if alert_data.get('config_id') else None,
                        user_id=alert_data.get('user_id', ''),
                        tenant_id=alert_data.get('tenant_id'),
                        dimension=alert_data.get('dimension', ''),
                        current_score=alert_data.get('current_score', 0.0),
                        threshold_score=alert_data.get('threshold_score', 0.0),
                        severity=alert_data.get('severity', 'medium'),
                        message=alert_data.get('message', ''),
                        acknowledged=alert_data.get('acknowledged', False),
                    )
                    session.add(db_alert)
                    session.flush()
                    logger.info(f"Created quality alert (db): {alert_data['alert_id']}")
                    return db_alert.to_dict()
        except Exception as e:
            logger.error(f"Failed to create quality alert in database: {e}")
            self._alerts[alert_data['alert_id']] = alert_data.copy()
        
        return alert_data
    
    def get_by_id(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取质量告警
        
        Args:
            alert_id: 质量告警ID
            
        Returns:
            质量告警数据
        """
        if self._use_memory_storage:
            return self._alerts.get(alert_id)
        
        try:
            from backend.schemas.data_quality_db_models import QualityAlert
            
            with self._get_session() as session:
                if session:
                    db_alert = session.query(QualityAlert).filter(
                        QualityAlert.id == uuid.UUID(alert_id)
                    ).first()
                    if db_alert:
                        return db_alert.to_dict()
        except Exception as e:
            logger.error(f"Failed to get quality alert from database: {e}")
            return self._alerts.get(alert_id)
        
        return None
    
    def get_by_dataset(
        self,
        dataset_id: str,
        acknowledged: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取数据集的质量告警
        
        Args:
            dataset_id: 数据集ID
            acknowledged: 是否已确认过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            质量告警列表和总数
        """
        if self._use_memory_storage:
            filtered = []
            for alert in self._alerts.values():
                if alert.get('dataset_id') != dataset_id:
                    continue
                if acknowledged is not None and alert.get('acknowledged') != acknowledged:
                    continue
                filtered.append(alert)
            
            filtered.sort(key=lambda x: x.get('triggered_at') or datetime.min, reverse=True)
            total = len(filtered)
            return filtered[offset:offset + limit], total
        
        try:
            from backend.schemas.data_quality_db_models import QualityAlert
            from sqlalchemy import desc
            
            with self._get_session() as session:
                if session:
                    query = session.query(QualityAlert).filter(
                        QualityAlert.dataset_id == dataset_id
                    )
                    if acknowledged is not None:
                        query = query.filter(QualityAlert.acknowledged == acknowledged)
                    
                    total = query.count()
                    alerts = query.order_by(desc(QualityAlert.triggered_at)).offset(offset).limit(limit).all()
                    return [a.to_dict() for a in alerts], total
        except Exception as e:
            logger.error(f"Failed to get quality alerts from database: {e}")
        
        return [], 0
    
    def get_active_alerts(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取未确认的告警
        
        Args:
            user_id: 用户ID过滤
            tenant_id: 租户ID过滤
            
        Returns:
            未确认的告警列表
        """
        if self._use_memory_storage:
            filtered = []
            for alert in self._alerts.values():
                if alert.get('acknowledged'):
                    continue
                if user_id is not None and alert.get('user_id') != user_id:
                    continue
                if tenant_id is not None and alert.get('tenant_id') != tenant_id:
                    continue
                filtered.append(alert)
            return filtered
        
        try:
            from backend.schemas.data_quality_db_models import QualityAlert
            
            with self._get_session() as session:
                if session:
                    query = session.query(QualityAlert).filter(
                        QualityAlert.acknowledged == False
                    )
                    if user_id:
                        query = query.filter(QualityAlert.user_id == user_id)
                    if tenant_id:
                        query = query.filter(QualityAlert.tenant_id == tenant_id)
                    
                    alerts = query.all()
                    return [a.to_dict() for a in alerts]
        except Exception as e:
            logger.error(f"Failed to get active quality alerts from database: {e}")
        
        return []
    
    def acknowledge(
        self,
        alert_id: str,
        acknowledged_by: str
    ) -> Optional[Dict[str, Any]]:
        """确认告警
        
        Args:
            alert_id: 告警ID
            acknowledged_by: 确认者ID
            
        Returns:
            更新后的告警数据
        """
        if self._use_memory_storage:
            alert = self._alerts.get(alert_id)
            if not alert:
                return None
            
            alert['acknowledged'] = True
            alert['acknowledged_by'] = acknowledged_by
            alert['acknowledged_at'] = datetime.utcnow()
            logger.info(f"Acknowledged quality alert (memory): {alert_id}")
            return alert
        
        try:
            from backend.schemas.data_quality_db_models import QualityAlert
            
            with self._get_session() as session:
                if session:
                    db_alert = session.query(QualityAlert).filter(
                        QualityAlert.id == uuid.UUID(alert_id)
                    ).first()
                    if not db_alert:
                        return None
                    
                    db_alert.acknowledged = True
                    db_alert.acknowledged_by = acknowledged_by
                    db_alert.acknowledged_at = datetime.utcnow()
                    
                    session.flush()
                    logger.info(f"Acknowledged quality alert (db): {alert_id}")
                    return db_alert.to_dict()
        except Exception as e:
            logger.error(f"Failed to acknowledge quality alert in database: {e}")
        
        return None
    
    def delete(self, alert_id: str) -> bool:
        """删除告警
        
        Args:
            alert_id: 告警ID
            
        Returns:
            是否删除成功
        """
        if self._use_memory_storage:
            if alert_id in self._alerts:
                del self._alerts[alert_id]
                logger.info(f"Deleted quality alert (memory): {alert_id}")
                return True
            return False
        
        try:
            from backend.schemas.data_quality_db_models import QualityAlert
            
            with self._get_session() as session:
                if session:
                    db_alert = session.query(QualityAlert).filter(
                        QualityAlert.id == uuid.UUID(alert_id)
                    ).first()
                    if db_alert:
                        session.delete(db_alert)
                        logger.info(f"Deleted quality alert (db): {alert_id}")
                        return True
        except Exception as e:
            logger.error(f"Failed to delete quality alert from database: {e}")
        
        return False


# ============================================================================
# 数据质量仓库管理器
# ============================================================================

class DataQualityRepositoryManager:
    """数据质量仓库管理器
    
    聚合管理所有数据质量相关的仓库。
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, use_memory_storage: bool = False):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, use_memory_storage: bool = False):
        if self._initialized:
            return
        
        self._use_memory_storage = use_memory_storage
        self.assessment_repo = QualityAssessmentRepository(use_memory_storage=use_memory_storage)
        self.issue_repo = QualityIssueRepository(use_memory_storage=use_memory_storage)
        self.cleaning_repo = CleaningRecordRepository(use_memory_storage=use_memory_storage)
        self.rule_repo = QualityRuleRepository(use_memory_storage=use_memory_storage)
        self.validation_repo = RuleValidationRecordRepository(use_memory_storage=use_memory_storage)
        self.report_repo = QualityReportRepository(use_memory_storage=use_memory_storage)
        self.monitoring_config_repo = QualityMonitoringConfigRepository(use_memory_storage=use_memory_storage)
        self.alert_repo = QualityAlertRepository(use_memory_storage=use_memory_storage)
        
        self._initialized = True
        storage_mode = "memory" if use_memory_storage else "database"
        logger.info(f"DataQualityRepositoryManager initialized with {storage_mode} storage")
    
    def reset(self):
        """重置所有仓库（用于测试）"""
        self.assessment_repo = QualityAssessmentRepository(use_memory_storage=self._use_memory_storage)
        self.issue_repo = QualityIssueRepository(use_memory_storage=self._use_memory_storage)
        self.cleaning_repo = CleaningRecordRepository(use_memory_storage=self._use_memory_storage)
        self.rule_repo = QualityRuleRepository(use_memory_storage=self._use_memory_storage)
        self.validation_repo = RuleValidationRecordRepository(use_memory_storage=self._use_memory_storage)
        self.report_repo = QualityReportRepository(use_memory_storage=self._use_memory_storage)
        self.monitoring_config_repo = QualityMonitoringConfigRepository(use_memory_storage=self._use_memory_storage)
        self.alert_repo = QualityAlertRepository(use_memory_storage=self._use_memory_storage)
        logger.info("DataQualityRepositoryManager reset")


# 全局仓库管理器实例
_repository_manager: Optional[DataQualityRepositoryManager] = None


def get_quality_repository_manager(use_memory_storage: bool = False) -> DataQualityRepositoryManager:
    """获取数据质量仓库管理器实例
    
    Args:
        use_memory_storage: 是否使用内存存储
    
    Returns:
        DataQualityRepositoryManager实例
    """
    global _repository_manager
    if _repository_manager is None:
        _repository_manager = DataQualityRepositoryManager(use_memory_storage=use_memory_storage)
    return _repository_manager


# ============================================================================
# 实体类别名导出（向后兼容）
# ============================================================================

# 从数据库模型导入并创建别名
try:
    from backend.schemas.data_quality_db_models import (
        QualityAssessment as QualityAssessmentEntity,
        QualityIssue as QualityIssueEntity,
        CleaningRecord as CleaningRecordEntity,
        QualityRule as QualityRuleEntity,
        RuleValidationRecord as RuleValidationRecordEntity,
        QualityReport as QualityReportEntity,
        QualityMonitoringConfig as QualityMonitoringConfigEntity,
        QualityAlert as QualityAlertEntity,
    )
except ImportError as e:
    # 如果数据库模型不可用，创建占位符类
    logger.warning(f"Failed to import data quality db models: {e}")
    
    class QualityAssessmentEntity:
        """占位符类"""
        pass
    
    class QualityIssueEntity:
        """占位符类"""
        pass
    
    class CleaningRecordEntity:
        """占位符类"""
        pass
    
    class QualityRuleEntity:
        """占位符类"""
        pass
    
    class RuleValidationRecordEntity:
        """占位符类"""
        pass
    
    class QualityReportEntity:
        """占位符类"""
        pass
    
    class QualityMonitoringConfigEntity:
        """占位符类"""
        pass
    
    class QualityAlertEntity:
        """占位符类"""
        pass


# 导出所有公开接口
__all__ = [
    # 仓库类
    'QualityAssessmentRepository',
    'QualityIssueRepository',
    'CleaningRecordRepository',
    'QualityRuleRepository',
    'RuleValidationRecordRepository',
    'QualityReportRepository',
    'QualityMonitoringConfigRepository',
    'QualityAlertRepository',
    'DataQualityRepositoryManager',
    # 工厂函数
    'get_quality_repository_manager',
    # 实体类别名
    'QualityAssessmentEntity',
    'QualityIssueEntity',
    'CleaningRecordEntity',
    'QualityRuleEntity',
    'RuleValidationRecordEntity',
    'QualityReportEntity',
    'QualityMonitoringConfigEntity',
    'QualityAlertEntity',
]
