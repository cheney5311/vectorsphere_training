"""数据质量服务

实现数据质量管理相关的业务逻辑。
"""

import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.repositories.dataset_repository import DatasetRepository
from backend.repositories.data_quality_repository import (
    QualityAssessmentRepository,
    QualityIssueRepository,
    CleaningRecordRepository,
    QualityRuleRepository,
    RuleValidationRecordRepository,
    QualityReportRepository,
    QualityMonitoringConfigRepository,
    QualityAlertRepository,
    QualityAssessmentEntity,
    QualityIssueEntity,
    CleaningRecordEntity,
    QualityRuleEntity,
    RuleValidationRecordEntity,
    QualityReportEntity,
    QualityMonitoringConfigEntity,
    get_quality_repository_manager,
)
from backend.modules.dataset.data_quality_module import DataQualityModule
from backend.modules.dataset.dataset_exceptions import (
    DatasetNotFoundError,
    QualityAssessmentFailedError,
    DataCleaningError,
    CleaningRollbackError,
    QualityRuleError,
    QualityReportGenerationError,
    QualityMonitoringError,
)
from backend.services.data_quality_service_interface import DataQualityServiceInterface

logger = logging.getLogger(__name__)


class DataQualityService(DataQualityServiceInterface):
    """数据质量服务
    
    实现数据质量评估、问题检测、数据清理等业务逻辑。
    """
    
    def __init__(
        self,
        dataset_repository: Optional[DatasetRepository] = None,
        assessment_repository: Optional[QualityAssessmentRepository] = None,
        issue_repository: Optional[QualityIssueRepository] = None,
        cleaning_repository: Optional[CleaningRecordRepository] = None,
        rule_repository: Optional[QualityRuleRepository] = None,
        validation_repository: Optional[RuleValidationRecordRepository] = None,
        report_repository: Optional[QualityReportRepository] = None,
        monitoring_config_repository: Optional[QualityMonitoringConfigRepository] = None,
        alert_repository: Optional[QualityAlertRepository] = None,
        quality_module: Optional[DataQualityModule] = None
    ):
        """初始化数据质量服务
        
        Args:
            dataset_repository: 数据集仓库
            assessment_repository: 评估记录仓库
            issue_repository: 问题记录仓库
            cleaning_repository: 清理记录仓库
            rule_repository: 规则仓库
            validation_repository: 验证记录仓库
            report_repository: 报告仓库
            monitoring_config_repository: 监控配置仓库
            alert_repository: 告警仓库
            quality_module: 数据质量模块
        """
        # 使用仓库管理器获取默认仓库
        repo_manager = get_quality_repository_manager()
        
        self.dataset_repository = dataset_repository or DatasetRepository()
        self.assessment_repository = assessment_repository or repo_manager.assessment_repo
        self.issue_repository = issue_repository or repo_manager.issue_repo
        self.cleaning_repository = cleaning_repository or repo_manager.cleaning_repo
        self.rule_repository = rule_repository or repo_manager.rule_repo
        self.validation_repository = validation_repository or repo_manager.validation_repo
        self.report_repository = report_repository or repo_manager.report_repo
        self.monitoring_config_repository = monitoring_config_repository or repo_manager.monitoring_config_repo
        self.alert_repository = alert_repository or repo_manager.alert_repo
        self.quality_module = quality_module or DataQualityModule()
        
        logger.info("DataQualityService initialized")
    
    # ========================================================================
    # 质量评估相关方法
    # ========================================================================
    
    def assess_data_quality(
        self,
        dataset_id: str,
        dimensions: Optional[List[str]] = None,
        include_column_metrics: bool = True,
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """评估数据质量
        
        Args:
            dataset_id: 数据集ID
            dimensions: 要评估的质量维度
            include_column_metrics: 是否包含列级指标
            sample_size: 采样大小
            
        Returns:
            Dict[str, Any]: 数据质量评估结果
            
        Raises:
            DatasetNotFoundError: 当数据集不存在时
            QualityAssessmentFailedError: 评估失败时
        """
        # 获取数据集
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        try:
            # 获取数据集数据（模拟，实际应该从数据存储中读取）
            data = self._load_dataset_data(dataset_id)
            
            # 执行质量评估
            metrics = self.quality_module.assess_quality(data, dataset_id, dimensions)
            
            # 如果不需要列级指标，移除
            if not include_column_metrics:
                metrics.pop('column_metrics', None)
            
            # 创建评估记录
            assessment = QualityAssessmentEntity(
                dataset_id=dataset_id,
                user_id=getattr(dataset, 'user_id', ''),
                overall_score=metrics.get('overall_score', 0.0),
                dimension_scores=metrics.get('dimension_scores', {}),
                column_metrics=metrics.get('column_metrics', []),
                total_records=metrics.get('total_records', 0),
                total_columns=metrics.get('total_columns', 0),
                missing_values_count=metrics.get('missing_values_count', 0),
                missing_values_rate=metrics.get('missing_values_rate', 0.0),
                duplicate_records_count=metrics.get('duplicate_records_count', 0),
                duplicate_records_rate=metrics.get('duplicate_records_rate', 0.0),
                outliers_count=metrics.get('outliers_count', 0),
                outliers_rate=metrics.get('outliers_rate', 0.0),
                status='completed'
            )
            
            saved_assessment = self.assessment_repository.create(assessment)
            
            # 更新数据集的质量信息
            if hasattr(dataset, 'config') and dataset.config:
                dataset.config['quality_metrics'] = metrics
                self.dataset_repository.update(dataset)
            
            logger.info(f"Quality assessment completed for dataset {dataset_id}, score: {metrics.get('overall_score', 0):.2%}")
            
            return {
                'success': True,
                'assessment_id': saved_assessment.assessment_id,
                'dataset_id': dataset_id,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Quality assessment failed for dataset {dataset_id}: {e}")
            raise QualityAssessmentFailedError(dataset_id, str(e))
    
    def get_assessment_history(
        self,
        dataset_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取评估历史
        
        Args:
            dataset_id: 数据集ID
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            Dict[str, Any]: 评估历史列表
        """
        assessments, total = self.assessment_repository.get_by_dataset(dataset_id, limit, offset)
        
        return {
            'success': True,
            'dataset_id': dataset_id,
            'assessments': [a.to_dict() for a in assessments],
            'total': total,
            'limit': limit,
            'offset': offset
        }
    
    def get_assessment_by_id(self, assessment_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取评估记录
        
        Args:
            assessment_id: 评估记录ID
            
        Returns:
            Optional[Dict[str, Any]]: 评估记录
        """
        assessment = self.assessment_repository.get_by_id(assessment_id)
        if not assessment:
            return None
        return assessment.to_dict()
    
    # ========================================================================
    # 问题检测相关方法
    # ========================================================================
    
    def detect_data_issues(
        self,
        dataset_id: str,
        issue_types: Optional[List[str]] = None,
        severity_threshold: str = "low",
        max_issues: int = 100,
        include_samples: bool = True,
        sample_count: int = 5
    ) -> Dict[str, Any]:
        """检测数据问题
        
        Args:
            dataset_id: 数据集ID
            issue_types: 要检测的问题类型
            severity_threshold: 严重程度阈值
            max_issues: 最大返回问题数
            include_samples: 是否包含示例值
            sample_count: 示例数量
            
        Returns:
            Dict[str, Any]: 检测到的数据问题列表
        """
        # 获取数据集
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        # 获取数据
        data = self._load_dataset_data(dataset_id)
        
        # 执行问题检测
        detection_result = self.quality_module.detect_issues(
            data, dataset_id, issue_types, severity_threshold, max_issues
        )
        
        # 保存检测到的问题
        for issue_data in detection_result.get('issues', []):
            issue_entity = QualityIssueEntity(
                dataset_id=dataset_id,
                user_id=getattr(dataset, 'user_id', ''),
                tenant_id=getattr(dataset, 'tenant_id', None),
                issue_type=issue_data.get('issue_type', ''),
                severity=issue_data.get('severity', 'medium'),
                column_name=issue_data.get('column_name'),
                description=issue_data.get('description', ''),
                affected_count=issue_data.get('affected_count', 0),
                affected_rate=issue_data.get('affected_rate', 0.0),
                sample_values=issue_data.get('sample_values', [])[:sample_count] if include_samples else [],
                recommendation=issue_data.get('recommendation', ''),
                auto_fixable=issue_data.get('auto_fixable', False)
            )
            self.issue_repository.create(issue_entity)
        
        logger.info(f"Detected {detection_result.get('total_issues', 0)} issues for dataset {dataset_id}")
        
        return {
            'success': True,
            'dataset_id': dataset_id,
            'result': detection_result
        }
    
    def get_issue_history(
        self,
        dataset_id: str,
        status_filter: Optional[List[str]] = None,
        severity_filter: Optional[List[str]] = None,
        issue_type_filter: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取问题历史
        
        Args:
            dataset_id: 数据集ID
            status_filter: 状态过滤
            severity_filter: 严重程度过滤
            issue_type_filter: 问题类型过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            Dict[str, Any]: 问题历史列表
        """
        issues, total = self.issue_repository.get_by_dataset(
            dataset_id, status_filter, severity_filter, issue_type_filter, limit, offset
        )
        
        return {
            'success': True,
            'dataset_id': dataset_id,
            'issues': [i.to_dict() for i in issues],
            'total': total,
            'limit': limit,
            'offset': offset
        }
    
    def resolve_issue(
        self,
        issue_id: str,
        user_id: str,
        resolution_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """解决问题
        
        Args:
            issue_id: 问题ID
            user_id: 解决者ID
            resolution_notes: 解决备注
            
        Returns:
            Dict[str, Any]: 更新后的问题记录
        """
        issue = self.issue_repository.update_status(issue_id, 'resolved', user_id)
        if not issue:
            return {'success': False, 'error': f'问题 {issue_id} 不存在'}
        
        logger.info(f"Issue {issue_id} resolved by user {user_id}")
        
        return {
            'success': True,
            'issue': issue.to_dict()
        }
    
    def ignore_issue(
        self,
        issue_id: str,
        user_id: str,
        ignore_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """忽略问题
        
        Args:
            issue_id: 问题ID
            user_id: 操作者ID
            ignore_reason: 忽略原因
            
        Returns:
            Dict[str, Any]: 更新后的问题记录
        """
        issue = self.issue_repository.update_status(issue_id, 'ignored', user_id)
        if not issue:
            return {'success': False, 'error': f'问题 {issue_id} 不存在'}
        
        logger.info(f"Issue {issue_id} ignored by user {user_id}")
        
        return {
            'success': True,
            'issue': issue.to_dict()
        }
    
    # ========================================================================
    # 数据清理相关方法
    # ========================================================================
    
    def clean_data(
        self,
        dataset_id: str,
        config: Dict[str, Any],
        user_id: str,
        create_new_dataset: bool = True,
        new_dataset_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """清理数据
        
        Args:
            dataset_id: 数据集ID
            config: 清理配置
            user_id: 用户ID
            create_new_dataset: 是否创建新数据集
            new_dataset_name: 新数据集名称
            
        Returns:
            Dict[str, Any]: 清理结果
        """
        # 获取数据集
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        # 创建清理记录
        cleaning_record = CleaningRecordEntity(
            dataset_id=dataset_id,
            user_id=user_id,
            tenant_id=getattr(dataset, 'tenant_id', None),
            config=config,
            original_dataset_id=dataset_id,
            status='in_progress'
        )
        cleaning_record.started_at = datetime.utcnow()
        saved_record = self.cleaning_repository.create(cleaning_record)
        
        try:
            # 获取原始数据
            data = self._load_dataset_data(dataset_id)
            original_count = len(data)
            
            # 获取清理前的质量评分
            original_metrics = self.quality_module.assess_quality(data, dataset_id)
            original_score = original_metrics.get('overall_score', 0)
            
            # 执行数据清理
            cleaning_result = self.quality_module.clean_data(data, config, dataset_id)
            
            cleaned_data = cleaning_result.get('cleaned_data', data)
            cleaned_count = len(cleaned_data)
            
            # 获取清理后的质量评分
            cleaned_metrics = self.quality_module.assess_quality(cleaned_data, dataset_id)
            cleaned_score = cleaned_metrics.get('overall_score', 0)
            
            # 创建新数据集（如果需要）
            cleaned_dataset_id = dataset_id
            if create_new_dataset:
                # 这里应该创建新数据集，简化实现
                cleaned_dataset_id = str(uuid.uuid4())
                new_name = new_dataset_name or f"{getattr(dataset, 'name', 'dataset')}_cleaned"
                logger.info(f"Created cleaned dataset: {cleaned_dataset_id} ({new_name})")
            
            # 更新清理记录
            saved_record.status = 'completed'
            saved_record.completed_at = datetime.utcnow()
            saved_record.cleaned_dataset_id = cleaned_dataset_id
            saved_record.original_record_count = original_count
            saved_record.cleaned_record_count = cleaned_count
            saved_record.total_records_affected = cleaning_result.get('total_records_affected', 0)
            saved_record.operation_results = cleaning_result.get('operation_results', [])
            saved_record.original_quality_score = original_score
            saved_record.cleaned_quality_score = cleaned_score
            saved_record.improvement = cleaned_score - original_score
            saved_record.execution_time_ms = cleaning_result.get('execution_time_ms', 0)
            
            self.cleaning_repository.update(saved_record)
            
            logger.info(f"Data cleaning completed for dataset {dataset_id}, "
                       f"quality improved from {original_score:.2%} to {cleaned_score:.2%}")
            
            return {
                'success': True,
                'dataset_id': dataset_id,
                'result': {
                    'cleaning_id': saved_record.cleaning_id,
                    'cleaned_dataset_id': cleaned_dataset_id,
                    'status': 'completed',
                    'original_record_count': original_count,
                    'cleaned_record_count': cleaned_count,
                    'total_records_affected': saved_record.total_records_affected,
                    'operation_results': saved_record.operation_results,
                    'original_quality_score': original_score,
                    'cleaned_quality_score': cleaned_score,
                    'improvement': saved_record.improvement,
                    'execution_time_ms': saved_record.execution_time_ms,
                    'cleaned_at': saved_record.completed_at.isoformat() if saved_record.completed_at else None
                }
            }
            
        except Exception as e:
            # 更新清理记录为失败状态
            self.cleaning_repository.update_status(
                saved_record.cleaning_id, 'failed', str(e)
            )
            logger.error(f"Data cleaning failed for dataset {dataset_id}: {e}")
            raise DataCleaningError(dataset_id, 'clean', str(e))
    
    def get_cleaning_history(
        self,
        dataset_id: str,
        status_filter: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取清理历史
        
        Args:
            dataset_id: 数据集ID
            status_filter: 状态过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            Dict[str, Any]: 清理历史列表
        """
        records, total = self.cleaning_repository.get_by_dataset(
            dataset_id, status_filter, limit, offset
        )
        
        return {
            'success': True,
            'dataset_id': dataset_id,
            'records': [r.to_dict() for r in records],
            'total': total,
            'limit': limit,
            'offset': offset
        }
    
    def get_cleaning_record(self, cleaning_id: str) -> Optional[Dict[str, Any]]:
        """获取清理记录
        
        Args:
            cleaning_id: 清理记录ID
            
        Returns:
            Optional[Dict[str, Any]]: 清理记录
        """
        record = self.cleaning_repository.get_by_id(cleaning_id)
        if not record:
            return None
        return record.to_dict()
    
    def rollback_cleaning(
        self,
        cleaning_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """回滚清理操作
        
        Args:
            cleaning_id: 清理记录ID
            user_id: 用户ID
            
        Returns:
            Dict[str, Any]: 回滚结果
        """
        record = self.cleaning_repository.get_by_id(cleaning_id)
        if not record:
            raise CleaningRollbackError(cleaning_id, '清理记录不存在')
        
        if record.status != 'completed':
            raise CleaningRollbackError(cleaning_id, '只能回滚已完成的清理操作')
        
        if not record.backup_path:
            raise CleaningRollbackError(cleaning_id, '没有备份数据，无法回滚')
        
        try:
            # 执行回滚逻辑（简化实现）
            self.cleaning_repository.update_status(cleaning_id, 'rolled_back')
            
            logger.info(f"Cleaning {cleaning_id} rolled back by user {user_id}")
            
            return {
                'success': True,
                'cleaning_id': cleaning_id,
                'status': 'rolled_back'
            }
            
        except Exception as e:
            logger.error(f"Rollback failed for cleaning {cleaning_id}: {e}")
            raise CleaningRollbackError(cleaning_id, str(e))
    
    # ========================================================================
    # 质量规则相关方法
    # ========================================================================
    
    def create_quality_rule(
        self,
        rule: Dict[str, Any],
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建质量规则
        
        Args:
            rule: 规则定义
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            Dict[str, Any]: 创建的规则
        """
        rule_entity = QualityRuleEntity(
            user_id=user_id,
            tenant_id=tenant_id,
            name=rule.get('name', ''),
            description=rule.get('description', ''),
            rule_type=rule.get('rule_type', ''),
            target_column=rule.get('target_column'),
            condition=rule.get('condition', ''),
            parameters=rule.get('parameters', {}),
            severity=rule.get('severity', 'medium'),
            enabled=rule.get('enabled', True),
            dataset_ids=rule.get('dataset_ids', [])
        )
        
        saved_rule = self.rule_repository.create(rule_entity)
        
        logger.info(f"Created quality rule: {saved_rule.rule_id}")
        
        return {
            'success': True,
            'rule': saved_rule.to_dict()
        }
    
    def get_quality_rules(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        enabled_only: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取质量规则列表
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            enabled_only: 是否只返回启用的规则
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            Dict[str, Any]: 规则列表
        """
        rules, total = self.rule_repository.get_by_user(
            user_id, tenant_id, enabled_only, limit, offset
        )
        
        return {
            'success': True,
            'rules': [r.to_dict() for r in rules],
            'total': total,
            'limit': limit,
            'offset': offset
        }
    
    def update_quality_rule(
        self,
        rule_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """更新质量规则
        
        Args:
            rule_id: 规则ID
            updates: 更新内容
            
        Returns:
            Dict[str, Any]: 更新后的规则
        """
        rule = self.rule_repository.get_by_id(rule_id)
        if not rule:
            raise QualityRuleError(rule_id, '规则不存在')
        
        # 更新字段
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        updated_rule = self.rule_repository.update(rule)
        
        logger.info(f"Updated quality rule: {rule_id}")
        
        return {
            'success': True,
            'rule': updated_rule.to_dict()
        }
    
    def delete_quality_rule(self, rule_id: str) -> bool:
        """删除质量规则
        
        Args:
            rule_id: 规则ID
            
        Returns:
            bool: 是否删除成功
        """
        result = self.rule_repository.delete(rule_id)
        if result:
            logger.info(f"Deleted quality rule: {rule_id}")
        return result
    
    def validate_rules(
        self,
        dataset_id: str,
        rules: List[Dict[str, Any]],
        user_id: str,
        stop_on_failure: bool = False
    ) -> Dict[str, Any]:
        """验证质量规则
        
        Args:
            dataset_id: 数据集ID
            rules: 规则列表
            user_id: 用户ID
            stop_on_failure: 遇到失败是否停止
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        # 获取数据集
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        # 获取数据
        data = self._load_dataset_data(dataset_id)
        
        # 执行规则验证
        validation_result = self.quality_module.validate_rules(
            data, rules, dataset_id, stop_on_failure
        )
        
        # 保存验证记录
        validation_record = RuleValidationRecordEntity(
            dataset_id=dataset_id,
            user_id=user_id,
            tenant_id=getattr(dataset, 'tenant_id', None),
            total_rules=validation_result.get('total_rules', 0),
            passed_rules=validation_result.get('passed_rules', 0),
            failed_rules=validation_result.get('failed_rules', 0),
            pass_rate=validation_result.get('pass_rate', 0.0),
            results=validation_result.get('results', [])
        )
        
        self.validation_repository.create(validation_record)
        
        logger.info(f"Rule validation completed for dataset {dataset_id}, "
                   f"pass rate: {validation_result.get('pass_rate', 0):.1%}")
        
        return {
            'success': True,
            'dataset_id': dataset_id,
            'result': validation_result
        }
    
    def get_validation_history(
        self,
        dataset_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取验证历史
        
        Args:
            dataset_id: 数据集ID
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            Dict[str, Any]: 验证历史列表
        """
        records, total = self.validation_repository.get_by_dataset(
            dataset_id, limit, offset
        )
        
        return {
            'success': True,
            'dataset_id': dataset_id,
            'records': [r.to_dict() for r in records],
            'total': total,
            'limit': limit,
            'offset': offset
        }
    
    # ========================================================================
    # 质量报告相关方法
    # ========================================================================
    
    def generate_quality_report(
        self,
        dataset_id: str,
        user_id: str,
        include_trends: bool = True,
        trend_period_days: int = 30,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """生成数据质量报告
        
        Args:
            dataset_id: 数据集ID
            user_id: 用户ID
            include_trends: 是否包含趋势分析
            trend_period_days: 趋势分析周期
            include_recommendations: 是否包含改进建议
            
        Returns:
            Dict[str, Any]: 数据质量报告
        """
        # 获取数据集
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        try:
            # 获取数据
            data = self._load_dataset_data(dataset_id)
            
            # 生成报告
            report_content = self.quality_module.generate_report(
                data, dataset_id, getattr(dataset, 'name', '')
            )
            
            # 添加趋势分析
            if include_trends:
                trends = self._get_quality_trends_data(dataset_id, trend_period_days)
                report_content['quality_trends'] = trends
            
            # 保存报告
            report_entity = QualityReportEntity(
                dataset_id=dataset_id,
                user_id=user_id,
                tenant_id=getattr(dataset, 'tenant_id', None),
                dataset_name=getattr(dataset, 'name', ''),
                report_content=report_content,
                summary=report_content.get('summary', ''),
                recommendations=report_content.get('recommendations', []),
                overall_score=report_content.get('overall_score', 0.0)
            )
            
            saved_report = self.report_repository.create(report_entity)
            
            logger.info(f"Generated quality report for dataset {dataset_id}")
            
            return {
                'success': True,
                'report': {
                    'report_id': saved_report.report_id,
                    **report_content
                }
            }
            
        except Exception as e:
            logger.error(f"Report generation failed for dataset {dataset_id}: {e}")
            raise QualityReportGenerationError(dataset_id, str(e))
    
    def get_report_history(
        self,
        dataset_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取报告历史
        
        Args:
            dataset_id: 数据集ID
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            Dict[str, Any]: 报告历史列表
        """
        reports, total = self.report_repository.get_by_dataset(
            dataset_id, limit, offset
        )
        
        return {
            'success': True,
            'dataset_id': dataset_id,
            'reports': [r.to_dict() for r in reports],
            'total': total,
            'limit': limit,
            'offset': offset
        }
    
    def get_report_by_id(self, report_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取报告
        
        Args:
            report_id: 报告ID
            
        Returns:
            Optional[Dict[str, Any]]: 报告内容
        """
        report = self.report_repository.get_by_id(report_id)
        if not report:
            return None
        return report.to_dict()
    
    # ========================================================================
    # 质量监控相关方法
    # ========================================================================
    
    def setup_quality_monitoring(
        self,
        dataset_id: str,
        config: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """设置质量监控
        
        Args:
            dataset_id: 数据集ID
            config: 监控配置
            user_id: 用户ID
            
        Returns:
            Dict[str, Any]: 监控配置
        """
        # 获取数据集
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(f"数据集 {dataset_id} 不存在")
        
        # 检查是否已存在配置
        existing_config = self.monitoring_config_repository.get_by_dataset(dataset_id)
        
        check_interval = config.get('check_interval_minutes', 60)
        
        if existing_config:
            # 更新现有配置
            existing_config.enabled = config.get('enabled', True)
            existing_config.thresholds = config.get('thresholds', [])
            existing_config.check_interval_minutes = check_interval
            existing_config.alert_channels = config.get('alert_channels', [])
            existing_config.next_check_at = datetime.utcnow() + timedelta(minutes=check_interval)
            
            saved_config = self.monitoring_config_repository.update(existing_config)
        else:
            # 创建新配置
            monitoring_config = QualityMonitoringConfigEntity(
                dataset_id=dataset_id,
                user_id=user_id,
                tenant_id=getattr(dataset, 'tenant_id', None),
                enabled=config.get('enabled', True),
                thresholds=config.get('thresholds', []),
                check_interval_minutes=check_interval,
                alert_channels=config.get('alert_channels', []),
                next_check_at=datetime.utcnow() + timedelta(minutes=check_interval)
            )
            saved_config = self.monitoring_config_repository.create(monitoring_config)
        
        logger.info(f"Setup quality monitoring for dataset {dataset_id}")
        
        return {
            'success': True,
            'config': saved_config.to_dict()
        }
    
    def get_monitoring_status(self, dataset_id: str) -> Dict[str, Any]:
        """获取监控状态
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            Dict[str, Any]: 监控状态
        """
        config = self.monitoring_config_repository.get_by_dataset(dataset_id)
        
        if not config:
            return {
                'success': True,
                'dataset_id': dataset_id,
                'monitoring_enabled': False,
                'config': None,
                'active_alerts': []
            }
        
        # 获取活跃告警
        alerts, _ = self.alert_repository.get_by_dataset(dataset_id, acknowledged=False)
        
        # 获取最新质量评分
        latest_assessment = self.assessment_repository.get_latest_by_dataset(dataset_id)
        current_score = latest_assessment.overall_score if latest_assessment else 0.0
        
        return {
            'success': True,
            'dataset_id': dataset_id,
            'monitoring_enabled': config.enabled,
            'last_check_at': config.last_check_at.isoformat() if config.last_check_at else None,
            'next_check_at': config.next_check_at.isoformat() if config.next_check_at else None,
            'current_score': current_score,
            'active_alerts': [a.to_dict() for a in alerts],
            'config': config.to_dict()
        }
    
    def update_monitoring_config(
        self,
        dataset_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """更新监控配置
        
        Args:
            dataset_id: 数据集ID
            updates: 更新内容
            
        Returns:
            Dict[str, Any]: 更新后的配置
        """
        config = self.monitoring_config_repository.get_by_dataset(dataset_id)
        if not config:
            raise QualityMonitoringError(dataset_id, '监控配置不存在')
        
        # 更新字段
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        updated_config = self.monitoring_config_repository.update(config)
        
        logger.info(f"Updated monitoring config for dataset {dataset_id}")
        
        return {
            'success': True,
            'config': updated_config.to_dict()
        }
    
    def disable_monitoring(self, dataset_id: str) -> bool:
        """禁用监控
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            bool: 是否成功
        """
        config = self.monitoring_config_repository.get_by_dataset(dataset_id)
        if not config:
            return False
        
        config.enabled = False
        self.monitoring_config_repository.update(config)
        
        logger.info(f"Disabled monitoring for dataset {dataset_id}")
        return True
    
    def get_quality_alerts(
        self,
        dataset_id: str,
        acknowledged: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取质量告警
        
        Args:
            dataset_id: 数据集ID
            acknowledged: 是否已确认过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            Dict[str, Any]: 告警列表
        """
        alerts, total = self.alert_repository.get_by_dataset(
            dataset_id, acknowledged, limit, offset
        )
        
        return {
            'success': True,
            'dataset_id': dataset_id,
            'alerts': [a.to_dict() for a in alerts],
            'total': total,
            'limit': limit,
            'offset': offset
        }
    
    def acknowledge_alert(
        self,
        alert_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """确认告警
        
        Args:
            alert_id: 告警ID
            user_id: 确认者ID
            
        Returns:
            Dict[str, Any]: 更新后的告警
        """
        alert = self.alert_repository.acknowledge(alert_id, user_id)
        if not alert:
            return {'success': False, 'error': f'告警 {alert_id} 不存在'}
        
        logger.info(f"Alert {alert_id} acknowledged by user {user_id}")
        
        return {
            'success': True,
            'alert': alert.to_dict()
        }
    
    # ========================================================================
    # 统计相关方法
    # ========================================================================
    
    def get_quality_stats(
        self,
        dataset_id: str,
        period_days: int = 30
    ) -> Dict[str, Any]:
        """获取质量统计
        
        Args:
            dataset_id: 数据集ID
            period_days: 统计周期
            
        Returns:
            Dict[str, Any]: 质量统计信息
        """
        # 获取最新评估
        latest_assessment = self.assessment_repository.get_latest_by_dataset(dataset_id)
        
        # 获取历史评估
        assessments, total_assessments = self.assessment_repository.get_by_dataset(
            dataset_id, limit=period_days
        )
        
        # 获取问题统计
        issues, total_issues = self.issue_repository.get_by_dataset(
            dataset_id, status_filter=['open'], limit=1000
        )
        
        # 计算统计信息
        if assessments:
            scores = [a.overall_score for a in assessments]
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            
            # 计算趋势
            if len(scores) >= 2:
                trend = scores[0] - scores[-1]  # 最新 - 最旧
            else:
                trend = 0.0
        else:
            avg_score = 0.0
            min_score = 0.0
            max_score = 0.0
            trend = 0.0
        
        # 按严重程度统计问题
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for issue in issues:
            if issue.severity in severity_counts:
                severity_counts[issue.severity] += 1
        
        return {
            'success': True,
            'dataset_id': dataset_id,
            'period_days': period_days,
            'stats': {
                'current_score': latest_assessment.overall_score if latest_assessment else 0.0,
                'average_score': round(avg_score, 4),
                'min_score': round(min_score, 4),
                'max_score': round(max_score, 4),
                'score_trend': round(trend, 4),
                'total_assessments': total_assessments,
                'open_issues': total_issues,
                'issues_by_severity': severity_counts,
                'last_assessed_at': (
                    latest_assessment.assessed_at.isoformat()
                    if latest_assessment and latest_assessment.assessed_at
                    else None
                )
            }
        }
    
    def get_quality_trends(
        self,
        dataset_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取质量趋势
        
        Args:
            dataset_id: 数据集ID
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, Any]: 质量趋势数据
        """
        trends = self._get_quality_trends_data(dataset_id, 30)
        
        return {
            'success': True,
            'dataset_id': dataset_id,
            'trends': trends
        }
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def _load_dataset_data(self, dataset_id: str) -> List[Dict[str, Any]]:
        """加载数据集数据
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            List[Dict[str, Any]]: 数据列表
            
        Note:
            这是一个模拟实现，实际应该从数据存储中读取数据
        """
        # 模拟数据
        # 在实际实现中，应该从文件系统、数据库或对象存储中读取数据
        import random
        
        sample_data = []
        for i in range(100):
            sample_data.append({
                'id': i + 1,
                'name': f'Item_{i + 1}' if random.random() > 0.05 else None,
                'value': random.randint(1, 1000) if random.random() > 0.03 else None,
                'category': random.choice(['A', 'B', 'C', 'D', None]),
                'score': round(random.uniform(0, 100), 2) if random.random() > 0.08 else None,
                'created_at': f'2024-01-{random.randint(1, 28):02d}',
            })
        
        return sample_data
    
    def _get_quality_trends_data(
        self,
        dataset_id: str,
        days: int
    ) -> List[Dict[str, Any]]:
        """获取质量趋势数据
        
        Args:
            dataset_id: 数据集ID
            days: 天数
            
        Returns:
            List[Dict[str, Any]]: 趋势数据
        """
        assessments, _ = self.assessment_repository.get_by_dataset(
            dataset_id, limit=days
        )
        
        trends = []
        for assessment in assessments:
            trends.append({
                'timestamp': assessment.assessed_at.isoformat() if assessment.assessed_at else None,
                'overall_score': assessment.overall_score,
                'dimension_scores': {
                    k: v.get('score', 0) if isinstance(v, dict) else 0
                    for k, v in assessment.dimension_scores.items()
                }
            })
        
        return trends
