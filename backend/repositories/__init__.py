"""数据访问层

提供统一的数据访问接口。
"""

import logging

logger = logging.getLogger(__name__)

# 延迟导入以避免循环依赖和模型缺失问题
def _lazy_import_user_repository():
    try:
        from .user_repository import UserRepository, get_user_repository
        return UserRepository, get_user_repository
    except ImportError as e:
        logger.warning(f"Failed to import user_repository: {e}")
        return None, None

def _lazy_import_training_repositories():
    try:
        from .training_session_repository import TrainingSessionRepository
        from .training_history_repository import TrainingHistoryRepository
        return TrainingSessionRepository, TrainingHistoryRepository
    except ImportError as e:
        logger.warning(f"Failed to import training repositories: {e}")
        return None, None

def _lazy_import_other_repositories():
    try:
        from .model_repository import ModelRepository
        from .agent_repository import AgentRepository
        from .dataset_repository import DatasetRepository
        return ModelRepository, AgentRepository, DatasetRepository
    except ImportError as e:
        logger.warning(f"Failed to import other repositories: {e}")
        return None, None, None

# 租户相关仓库（核心功能，必须成功导入）
from .tenant_repository import (
    TenantRepository,
    TenantUserRepository,
    TenantQuotaRepository,
    get_tenant_repository,
    get_tenant_user_repository,
    get_tenant_quota_repository
)
from .tenant_api_key_repository import (
    TenantApiKeyRepository,
    get_tenant_api_key_repository
)
from .tenant_invite_repository import (
    TenantInviteRepository,
    get_tenant_invite_repository
)
from .tenant_audit_log_repository import (
    TenantAuditLogRepository,
    get_tenant_audit_log_repository
)

# 计费相关仓库
def _lazy_import_billing_repositories():
    try:
        from .billing_repository import (
            BillingRuleRepository,
            UsageRecordRepository,
            InvoiceRepository,
            BillingItemRepository,
            PaymentRepository,
            WalletRepository,
            WalletTransactionRepository
        )
        return (BillingRuleRepository, UsageRecordRepository, InvoiceRepository,
                BillingItemRepository, PaymentRepository, WalletRepository, WalletTransactionRepository)
    except ImportError as e:
        logger.warning(f"Failed to import billing repositories: {e}")
        return (None, None, None, None, None, None, None)


# 工作流相关仓库
def _lazy_import_workflow_repositories():
    try:
        from .workflow_repository import (
            WorkflowRepository,
            WorkflowExecutionRepository,
            WorkflowStepRepository,
            WorkflowTemplateRepository,
            WorkflowLogRepository
        )
        return (WorkflowRepository, WorkflowExecutionRepository, WorkflowStepRepository,
                WorkflowTemplateRepository, WorkflowLogRepository)
    except ImportError as e:
        logger.warning(f"Failed to import workflow repositories: {e}")
        return (None, None, None, None, None)


# 超参数优化相关仓库
def _lazy_import_hyperparameter_repositories():
    try:
        from .hyperparameter_optimization_repository import (
            HyperparameterOptimizationRepository,
            HyperparameterTrialRepository,
            HyperparameterSearchSpaceRepository,
            get_hyperparameter_optimization_repository,
            get_hyperparameter_trial_repository,
            get_hyperparameter_search_space_repository
        )
        return (HyperparameterOptimizationRepository, HyperparameterTrialRepository, 
                HyperparameterSearchSpaceRepository, get_hyperparameter_optimization_repository,
                get_hyperparameter_trial_repository, get_hyperparameter_search_space_repository)
    except ImportError as e:
        logger.warning(f"Failed to import hyperparameter repositories: {e}")
        return (None, None, None, None, None, None)


# 智能决策相关仓库
def _lazy_import_intelligent_decision_repositories():
    try:
        from .intelligent_decision_repository import (
            IntelligentDecisionRepository,
            AdaptiveOptimizationRepository,
            KnowledgeBaseRepository,
            ExperienceRecordRepository,
            get_intelligent_decision_repository,
            get_adaptive_optimization_repository,
            get_knowledge_base_repository,
            get_experience_record_repository
        )
        return (IntelligentDecisionRepository, AdaptiveOptimizationRepository,
                KnowledgeBaseRepository, ExperienceRecordRepository,
                get_intelligent_decision_repository, get_adaptive_optimization_repository,
                get_knowledge_base_repository, get_experience_record_repository)
    except ImportError as e:
        logger.warning(f"Failed to import intelligent decision repositories: {e}")
        return (None, None, None, None, None, None, None, None)


# 模型部署相关仓库
def _lazy_import_model_deployment_repositories():
    try:
        from .model_deployment_repository import (
            ModelDeploymentRepository,
            ModelDeploymentLogRepository,
            ModelServiceRepository,
            DeploymentModeConfigRepository,
            get_model_deployment_repository,
            get_model_deployment_log_repository,
            get_model_service_repository,
            get_deployment_mode_config_repository
        )
        return (ModelDeploymentRepository, ModelDeploymentLogRepository, ModelServiceRepository,
                DeploymentModeConfigRepository, get_model_deployment_repository, 
                get_model_deployment_log_repository, get_model_service_repository,
                get_deployment_mode_config_repository)
    except ImportError as e:
        logger.warning(f"Failed to import model deployment repositories: {e}")
        return (None, None, None, None, None, None, None, None)


# 模型评估相关仓库
def _lazy_import_model_evaluation_repositories():
    try:
        from .model_evaluation_repository import (
            ModelEvaluationRepository,
            ModelComparisonRepository,
            get_model_evaluation_repository,
            get_model_comparison_repository
        )
        return (ModelEvaluationRepository, ModelComparisonRepository,
                get_model_evaluation_repository, get_model_comparison_repository)
    except ImportError as e:
        logger.warning(f"Failed to import model evaluation repositories: {e}")
        return (None, None, None, None)


# 模型优化相关仓库
def _lazy_import_model_optimization_repositories():
    try:
        from .model_optimization_repository import (
            ModelOptimizationRepository,
            get_model_optimization_repository
        )
        return (ModelOptimizationRepository, get_model_optimization_repository)
    except ImportError as e:
        logger.warning(f"Failed to import model optimization repositories: {e}")
        return (None, None)


# 模型选择相关仓库
def _lazy_import_model_selection_repositories():
    try:
        from .model_selection_repository import (
            ModelRecommendationRepository,
            ModelConfigurationRepository,
            ModelCatalogRepository,
            get_model_recommendation_repository,
            get_model_configuration_repository,
            get_model_catalog_repository
        )
        return (ModelRecommendationRepository, ModelConfigurationRepository, ModelCatalogRepository,
                get_model_recommendation_repository, get_model_configuration_repository, 
                get_model_catalog_repository)
    except ImportError as e:
        logger.warning(f"Failed to import model selection repositories: {e}")
        return (None, None, None, None, None, None)


# 监控运维相关仓库
def _lazy_import_monitoring_operations_repositories():
    try:
        from .monitoring_operations_repository import (
            PerformanceMetricRepository,
            AlertRuleRepository,
            AlertHistoryRepository,
            AutomationTaskRepository,
            MonitoringReportRepository,
            get_performance_metric_repository,
            get_alert_rule_repository,
            get_alert_history_repository,
            get_automation_task_repository,
            get_monitoring_report_repository
        )
        return (PerformanceMetricRepository, AlertRuleRepository, AlertHistoryRepository,
                AutomationTaskRepository, MonitoringReportRepository,
                get_performance_metric_repository, get_alert_rule_repository,
                get_alert_history_repository, get_automation_task_repository,
                get_monitoring_report_repository)
    except ImportError as e:
        logger.warning(f"Failed to import monitoring operations repositories: {e}")
        return (None, None, None, None, None, None, None, None, None, None)


# 训练流水线相关仓库
def _lazy_import_pipeline_repositories():
    try:
        from .pipeline_repository import (
            TrainingPipelineRepository,
            PipelineExecutionRepository,
            PipelineStepExecutionRepository,
            PipelineTemplateRepository,
            get_pipeline_repository,
            get_execution_repository,
            get_step_execution_repository,
            get_template_repository
        )
        return (TrainingPipelineRepository, PipelineExecutionRepository,
                PipelineStepExecutionRepository, PipelineTemplateRepository,
                get_pipeline_repository, get_execution_repository,
                get_step_execution_repository, get_template_repository)
    except ImportError as e:
        logger.warning(f"Failed to import pipeline repositories: {e}")
        return (None, None, None, None, None, None, None, None)


# 三阶段训练相关仓库
def _lazy_import_three_stage_repositories():
    try:
        from .three_stage_training_repository import (
            ThreeStageSessionRepository,
            ThreeStageProgressRepository,
            get_session_repository,
            get_progress_repository
        )
        return (ThreeStageSessionRepository, ThreeStageProgressRepository,
                get_session_repository, get_progress_repository)
    except ImportError as e:
        logger.warning(f"Failed to import three stage training repositories: {e}")
        return (None, None, None, None)


# 训练执行相关仓库
def _lazy_import_training_execution_repositories():
    try:
        from .training_execution_repository import (
            TrainingExecutionRepository,
            get_training_execution_repository
        )
        return (TrainingExecutionRepository, get_training_execution_repository)
    except ImportError as e:
        logger.warning(f"Failed to import training execution repositories: {e}")
        return (None, None)


# 训练进度相关仓库
def _lazy_import_training_progress_repositories():
    try:
        from .training_progress_repository import (
            TrainingProgressRepository,
            get_training_progress_repository
        )
        return (TrainingProgressRepository, get_training_progress_repository)
    except ImportError as e:
        logger.warning(f"Failed to import training progress repositories: {e}")
        return (None, None)


# 训练任务相关仓库
def _lazy_import_training_job_repositories():
    try:
        from .training_job_repository import (
            TrainingJobRepository,
            TrainingJobLogRepository,
            get_training_job_repository,
            get_training_job_log_repository
        )
        return (TrainingJobRepository, TrainingJobLogRepository, 
                get_training_job_repository, get_training_job_log_repository)
    except ImportError as e:
        logger.warning(f"Failed to import training job repositories: {e}")
        return (None, None, None, None)


# 训练统计相关仓库
def _lazy_import_training_statistics_repositories():
    try:
        from .training_statistics_repository import (
            TrainingStatisticsRepository,
            get_training_statistics_repository,
            reset_training_statistics_repository
        )
        return (TrainingStatisticsRepository, get_training_statistics_repository,
                reset_training_statistics_repository)
    except ImportError as e:
        logger.warning(f"Failed to import training statistics repositories: {e}")
        return (None, None, None)


__all__ = [
    # 租户相关 Repository（核心）
    'TenantRepository',
    'TenantUserRepository',
    'TenantQuotaRepository',
    'get_tenant_repository',
    'get_tenant_user_repository',
    'get_tenant_quota_repository',
    
    # API密钥
    'TenantApiKeyRepository',
    'get_tenant_api_key_repository',
    
    # 邀请
    'TenantInviteRepository',
    'get_tenant_invite_repository',
    
    # 审计日志
    'TenantAuditLogRepository',
    'get_tenant_audit_log_repository',
    
    # 延迟导入辅助函数
    '_lazy_import_user_repository',
    '_lazy_import_training_repositories',
    '_lazy_import_other_repositories',
    '_lazy_import_billing_repositories',
    '_lazy_import_workflow_repositories',
    '_lazy_import_hyperparameter_repositories',
    '_lazy_import_intelligent_decision_repositories',
    '_lazy_import_model_deployment_repositories',
    '_lazy_import_model_evaluation_repositories',
    '_lazy_import_model_optimization_repositories',
    '_lazy_import_model_selection_repositories',
    '_lazy_import_monitoring_operations_repositories',
    '_lazy_import_pipeline_repositories',
    '_lazy_import_three_stage_repositories',
    '_lazy_import_training_execution_repositories',
    '_lazy_import_training_progress_repositories',
    '_lazy_import_training_job_repositories',
    '_lazy_import_training_statistics_repositories',
    '_lazy_import_security_repositories',
    '_lazy_import_artifact_repositories',
]


def _lazy_import_security_repositories():
    """延迟导入安全仓库"""
    try:
        from .security_repository import (
            UserSessionRepository,
            SecurityAuditLogRepository,
            UserRoleRepository,
            AccessPolicyRepository,
            EncryptionKeyRepository,
            DataProcessingRecordRepository,
            ComplianceReportRepository,
            get_session_repository,
            get_audit_log_repository,
            get_user_role_repository,
            get_access_policy_repository,
            get_encryption_key_repository,
            get_data_processing_repository,
            get_compliance_report_repository
        )
        return (
            UserSessionRepository, SecurityAuditLogRepository, UserRoleRepository,
            AccessPolicyRepository, EncryptionKeyRepository, DataProcessingRecordRepository,
            ComplianceReportRepository, get_session_repository, get_audit_log_repository,
            get_user_role_repository, get_access_policy_repository, get_encryption_key_repository,
            get_data_processing_repository, get_compliance_report_repository
        )
    except ImportError as e:
        logger.warning(f"Failed to import security repositories: {e}")
        return tuple([None] * 14)


def _lazy_import_artifact_repositories():
    """延迟导入工件仓库"""
    try:
        from .artifact_repository import (
            SecurityPolicyRepository,
            ArtifactRepository,
            ArtifactVersionRepository,
            FileMetadataRepository,
            ArtifactDependencyRepository,
            ArtifactAccessLogRepository,
            get_security_policy_repository,
            get_artifact_repository,
            get_artifact_version_repository,
            get_file_metadata_repository,
            get_artifact_dependency_repository,
            get_artifact_access_log_repository,
            reset_artifact_repositories
        )
        return (
            SecurityPolicyRepository, ArtifactRepository, ArtifactVersionRepository,
            FileMetadataRepository, ArtifactDependencyRepository, ArtifactAccessLogRepository,
            get_security_policy_repository, get_artifact_repository, get_artifact_version_repository,
            get_file_metadata_repository, get_artifact_dependency_repository,
            get_artifact_access_log_repository, reset_artifact_repositories
        )
    except ImportError as e:
        logger.warning(f"Failed to import artifact repositories: {e}")
        return tuple([None] * 13)


def _lazy_import_scheduler_repositories():
    """延迟导入调度器仓库"""
    try:
        from .scheduler_repository import (
            ScheduledTaskRepository,
            TaskExecutionLogRepository,
            TaskTemplateRepository,
            SchedulerMetricsRepository,
            get_scheduled_task_repository,
            get_execution_log_repository,
            get_task_template_repository,
            get_scheduler_metrics_repository,
            reset_scheduler_repositories
        )
        return (
            ScheduledTaskRepository, TaskExecutionLogRepository, TaskTemplateRepository,
            SchedulerMetricsRepository, get_scheduled_task_repository, get_execution_log_repository,
            get_task_template_repository, get_scheduler_metrics_repository, reset_scheduler_repositories
        )
    except ImportError as e:
        logger.warning(f"Failed to import scheduler repositories: {e}")
        return tuple([None] * 9)


def _lazy_import_gpu_resource_repositories():
    """延迟导入 GPU 资源仓库"""
    try:
        from .gpu_resource_repository import (
            GPUNodeRepository,
            GPUDeviceRepository,
            GPUAllocationRepository,
            GPUUsageHistoryRepository,
            get_gpu_node_repository,
            get_gpu_device_repository,
            get_gpu_allocation_repository,
            get_gpu_usage_repository,
            reset_gpu_repositories
        )
        return (
            GPUNodeRepository, GPUDeviceRepository, GPUAllocationRepository,
            GPUUsageHistoryRepository, get_gpu_node_repository, get_gpu_device_repository,
            get_gpu_allocation_repository, get_gpu_usage_repository, reset_gpu_repositories
        )
    except ImportError as e:
        logger.warning(f"Failed to import GPU resource repositories: {e}")
        return tuple([None] * 9)


def _lazy_import_optimization_repositories():
    """延迟导入优化仓库"""
    try:
        from .optimization_repository import (
            OptimizationSessionRepository,
            OptimizationRecommendationRepository,
            ResourceMetricSnapshotRepository,
            PerformanceAnalysisReportRepository,
            ResourceAlertRepository,
            get_optimization_session_repository,
            get_optimization_recommendation_repository,
            get_resource_metric_snapshot_repository,
            get_performance_analysis_report_repository,
            get_resource_alert_repository
        )
        return (
            OptimizationSessionRepository, OptimizationRecommendationRepository,
            ResourceMetricSnapshotRepository, PerformanceAnalysisReportRepository,
            ResourceAlertRepository, get_optimization_session_repository,
            get_optimization_recommendation_repository, get_resource_metric_snapshot_repository,
            get_performance_analysis_report_repository, get_resource_alert_repository
        )
    except ImportError as e:
        logger.warning(f"Failed to import optimization repositories: {e}")
        return tuple([None] * 10)
