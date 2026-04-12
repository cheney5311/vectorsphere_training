"""数据集模块异常定义

定义数据集模块相关的自定义异常。
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.exceptions import ValidationError, ResourceNotFoundError, BusinessLogicError


class DatasetError(BusinessLogicError):
    """数据集模块基础异常"""
    pass


class DatasetNotFoundError(DatasetError, ResourceNotFoundError):
    """数据集不存在异常"""
    pass


class DatasetValidationError(DatasetError, ValidationError):
    """数据集验证异常"""
    pass


class DatasetBusinessLogicError(DatasetError, BusinessLogicError):
    """数据集业务逻辑异常"""
    pass


# ============================================================================
# 数据发现相关异常
# ============================================================================

class DataDiscoveryError(DatasetError):
    """数据发现基础异常"""
    pass


class DataSourceConnectionError(DataDiscoveryError):
    """数据源连接异常"""
    def __init__(self, source_type: str, location: str, message: str = None):
        self.source_type = source_type
        self.location = location
        super().__init__(message or f"无法连接到数据源: {source_type}://{location}")


class DataSourceAuthenticationError(DataDiscoveryError):
    """数据源认证异常"""
    def __init__(self, source_type: str, message: str = None):
        self.source_type = source_type
        super().__init__(message or f"数据源认证失败: {source_type}")


class DataSourceScanError(DataDiscoveryError):
    """数据源扫描异常"""
    def __init__(self, source_id: str, reason: str):
        self.source_id = source_id
        self.reason = reason
        super().__init__(f"扫描数据源 {source_id} 失败: {reason}")


class DataSourceNotFoundError(DataDiscoveryError, ResourceNotFoundError):
    """数据源不存在异常"""
    def __init__(self, source_id: str):
        self.source_id = source_id
        super().__init__(f"数据源 {source_id} 不存在")


class DiscoveryNotFoundError(DataDiscoveryError, ResourceNotFoundError):
    """发现记录不存在异常"""
    def __init__(self, discovery_id: str):
        self.discovery_id = discovery_id
        super().__init__(f"发现记录 {discovery_id} 不存在")


class DataIngestionError(DataDiscoveryError):
    """数据接入异常"""
    def __init__(self, dataset_name: str, reason: str):
        self.dataset_name = dataset_name
        self.reason = reason
        super().__init__(f"接入数据集 {dataset_name} 失败: {reason}")


class SchemaInferenceError(DataDiscoveryError):
    """模式推断异常"""
    def __init__(self, dataset_id: str, reason: str):
        self.dataset_id = dataset_id
        self.reason = reason
        super().__init__(f"推断数据集 {dataset_id} 模式失败: {reason}")


class DataTransformationError(DataDiscoveryError):
    """数据转换异常"""
    def __init__(self, dataset_id: str, operation: str, reason: str):
        self.dataset_id = dataset_id
        self.operation = operation
        self.reason = reason
        super().__init__(f"转换数据集 {dataset_id} 失败 (操作: {operation}): {reason}")


class SyncConfigurationError(DataDiscoveryError):
    """同步配置异常"""
    def __init__(self, dataset_id: str, reason: str):
        self.dataset_id = dataset_id
        self.reason = reason
        super().__init__(f"配置数据集 {dataset_id} 同步失败: {reason}")


class SyncExecutionError(DataDiscoveryError):
    """同步执行异常"""
    def __init__(self, dataset_id: str, reason: str):
        self.dataset_id = dataset_id
        self.reason = reason
        super().__init__(f"同步数据集 {dataset_id} 失败: {reason}")


class UnsupportedDataFormatError(DataDiscoveryError):
    """不支持的数据格式异常"""
    def __init__(self, format_name: str, supported_formats: list = None):
        self.format_name = format_name
        self.supported_formats = supported_formats or []
        msg = f"不支持的数据格式: {format_name}"
        if supported_formats:
            msg += f"，支持的格式: {', '.join(supported_formats)}"
        super().__init__(msg)


class UnsupportedDataSourceError(DataDiscoveryError):
    """不支持的数据源类型异常"""
    def __init__(self, source_type: str, supported_types: list = None):
        self.source_type = source_type
        self.supported_types = supported_types or []
        msg = f"不支持的数据源类型: {source_type}"
        if supported_types:
            msg += f"，支持的类型: {', '.join(supported_types)}"
        super().__init__(msg)


class DataQualityError(DataDiscoveryError):
    """数据质量异常"""
    def __init__(self, dataset_id: str, issues: list):
        self.dataset_id = dataset_id
        self.issues = issues
        super().__init__(f"数据集 {dataset_id} 存在质量问题: {', '.join(issues)}")


# ============================================================================
# 数据预处理相关异常
# ============================================================================

class DataPreprocessingError(DatasetError):
    """数据预处理基础异常"""
    pass


class PreprocessingTaskNotFoundError(DataPreprocessingError, ResourceNotFoundError):
    """预处理任务不存在异常"""
    def __init__(self, task_id: str):
        self.task_id = task_id
        super().__init__(f"预处理任务 {task_id} 不存在")


class PreprocessingOperationError(DataPreprocessingError):
    """预处理操作异常"""
    def __init__(self, operation: str, dataset_id: str, reason: str):
        self.operation = operation
        self.dataset_id = dataset_id
        self.reason = reason
        super().__init__(f"预处理操作 {operation} 在数据集 {dataset_id} 上执行失败: {reason}")


class PreprocessingConfigError(DataPreprocessingError, ValidationError):
    """预处理配置异常"""
    def __init__(self, config_field: str, reason: str):
        self.config_field = config_field
        self.reason = reason
        super().__init__(f"预处理配置错误 ({config_field}): {reason}")


class PreprocessingPipelineError(DataPreprocessingError):
    """预处理流水线异常"""
    def __init__(self, pipeline_id: str, step: int, reason: str):
        self.pipeline_id = pipeline_id
        self.step = step
        self.reason = reason
        super().__init__(f"预处理流水线 {pipeline_id} 在第 {step} 步失败: {reason}")


class FeatureEngineeringError(DataPreprocessingError):
    """特征工程异常"""
    def __init__(self, dataset_id: str, operation: str, reason: str):
        self.dataset_id = dataset_id
        self.operation = operation
        self.reason = reason
        super().__init__(f"特征工程操作 {operation} 在数据集 {dataset_id} 上失败: {reason}")


class FeatureNotFoundError(DataPreprocessingError, ResourceNotFoundError):
    """特征不存在异常"""
    def __init__(self, feature_name: str, dataset_id: str):
        self.feature_name = feature_name
        self.dataset_id = dataset_id
        super().__init__(f"数据集 {dataset_id} 中不存在特征 {feature_name}")


class FeatureCreationError(DataPreprocessingError):
    """特征创建异常"""
    def __init__(self, feature_name: str, reason: str):
        self.feature_name = feature_name
        self.reason = reason
        super().__init__(f"创建特征 {feature_name} 失败: {reason}")


class FeatureSelectionError(DataPreprocessingError):
    """特征选择异常"""
    def __init__(self, method: str, reason: str):
        self.method = method
        self.reason = reason
        super().__init__(f"特征选择方法 {method} 执行失败: {reason}")


class DataAugmentationError(DataPreprocessingError):
    """数据增强异常"""
    def __init__(self, dataset_id: str, method: str, reason: str):
        self.dataset_id = dataset_id
        self.method = method
        self.reason = reason
        super().__init__(f"数据增强方法 {method} 在数据集 {dataset_id} 上失败: {reason}")


class DataSplitError(DataPreprocessingError):
    """数据分割异常"""
    def __init__(self, dataset_id: str, reason: str):
        self.dataset_id = dataset_id
        self.reason = reason
        super().__init__(f"数据集 {dataset_id} 分割失败: {reason}")


class DataSplitRatioError(DataPreprocessingError, ValidationError):
    """数据分割比例异常"""
    def __init__(self, train_ratio: float, val_ratio: float, test_ratio: float):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        total = train_ratio + val_ratio + test_ratio
        super().__init__(
            f"分割比例之和必须为1，当前为 {total} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )


class InsufficientDataError(DataPreprocessingError):
    """数据不足异常"""
    def __init__(self, dataset_id: str, required: int, actual: int):
        self.dataset_id = dataset_id
        self.required = required
        self.actual = actual
        super().__init__(
            f"数据集 {dataset_id} 数据量不足: 需要 {required} 条，实际只有 {actual} 条"
        )


class MissingValueHandlingError(DataPreprocessingError):
    """缺失值处理异常"""
    def __init__(self, column: str, strategy: str, reason: str):
        self.column = column
        self.strategy = strategy
        self.reason = reason
        super().__init__(f"使用策略 {strategy} 处理列 {column} 的缺失值失败: {reason}")


class OutlierHandlingError(DataPreprocessingError):
    """异常值处理异常"""
    def __init__(self, column: str, method: str, reason: str):
        self.column = column
        self.method = method
        self.reason = reason
        super().__init__(f"使用方法 {method} 处理列 {column} 的异常值失败: {reason}")


class NormalizationError(DataPreprocessingError):
    """标准化异常"""
    def __init__(self, column: str, method: str, reason: str):
        self.column = column
        self.method = method
        self.reason = reason
        super().__init__(f"使用方法 {method} 标准化列 {column} 失败: {reason}")


class EncodingError(DataPreprocessingError):
    """编码异常"""
    def __init__(self, column: str, method: str, reason: str):
        self.column = column
        self.method = method
        self.reason = reason
        super().__init__(f"使用方法 {method} 编码列 {column} 失败: {reason}")


class TokenizationError(DataPreprocessingError):
    """分词异常"""
    def __init__(self, column: str, reason: str):
        self.column = column
        self.reason = reason
        super().__init__(f"分词列 {column} 失败: {reason}")


class VectorizationError(DataPreprocessingError):
    """向量化异常"""
    def __init__(self, column: str, method: str, reason: str):
        self.column = column
        self.method = method
        self.reason = reason
        super().__init__(f"使用方法 {method} 向量化列 {column} 失败: {reason}")


class DimensionReductionError(DataPreprocessingError):
    """降维异常"""
    def __init__(self, method: str, reason: str):
        self.method = method
        self.reason = reason
        super().__init__(f"降维方法 {method} 执行失败: {reason}")


class SamplingError(DataPreprocessingError):
    """采样异常"""
    def __init__(self, method: str, reason: str):
        self.method = method
        self.reason = reason
        super().__init__(f"采样方法 {method} 执行失败: {reason}")


class PreprocessingRollbackError(DataPreprocessingError):
    """预处理回滚异常"""
    def __init__(self, task_id: str, reason: str):
        self.task_id = task_id
        self.reason = reason
        super().__init__(f"回滚预处理任务 {task_id} 失败: {reason}")


# ============================================================================
# 数据质量相关异常
# ============================================================================

class DataQualityAssessmentError(DatasetError):
    """数据质量评估基础异常"""
    pass


class QualityAssessmentNotFoundError(DataQualityAssessmentError, ResourceNotFoundError):
    """质量评估记录不存在异常"""
    def __init__(self, assessment_id: str):
        self.assessment_id = assessment_id
        super().__init__(f"质量评估记录 {assessment_id} 不存在")


class QualityAssessmentFailedError(DataQualityAssessmentError):
    """质量评估失败异常"""
    def __init__(self, dataset_id: str, reason: str):
        self.dataset_id = dataset_id
        self.reason = reason
        super().__init__(f"数据集 {dataset_id} 质量评估失败: {reason}")


class QualityMetricsCalculationError(DataQualityAssessmentError):
    """质量指标计算异常"""
    def __init__(self, metric_name: str, reason: str):
        self.metric_name = metric_name
        self.reason = reason
        super().__init__(f"计算质量指标 {metric_name} 失败: {reason}")


class IssueDetectionError(DataQualityAssessmentError):
    """问题检测异常"""
    def __init__(self, dataset_id: str, issue_type: str, reason: str):
        self.dataset_id = dataset_id
        self.issue_type = issue_type
        self.reason = reason
        super().__init__(f"检测数据集 {dataset_id} 的 {issue_type} 问题失败: {reason}")


class DataCleaningError(DataQualityAssessmentError):
    """数据清理异常"""
    def __init__(self, dataset_id: str, operation: str, reason: str):
        self.dataset_id = dataset_id
        self.operation = operation
        self.reason = reason
        super().__init__(f"清理数据集 {dataset_id} 失败 (操作: {operation}): {reason}")


class CleaningOperationError(DataQualityAssessmentError):
    """清理操作执行异常"""
    def __init__(self, operation_id: str, strategy: str, reason: str):
        self.operation_id = operation_id
        self.strategy = strategy
        self.reason = reason
        super().__init__(f"执行清理操作 {operation_id} ({strategy}) 失败: {reason}")


class CleaningConfigError(DataQualityAssessmentError, ValidationError):
    """清理配置异常"""
    def __init__(self, config_field: str, reason: str):
        self.config_field = config_field
        self.reason = reason
        super().__init__(f"清理配置错误 ({config_field}): {reason}")


class CleaningRollbackError(DataQualityAssessmentError):
    """清理回滚异常"""
    def __init__(self, cleaning_id: str, reason: str):
        self.cleaning_id = cleaning_id
        self.reason = reason
        super().__init__(f"回滚清理操作 {cleaning_id} 失败: {reason}")


class QualityRuleError(DataQualityAssessmentError):
    """质量规则异常"""
    def __init__(self, rule_id: str, reason: str):
        self.rule_id = rule_id
        self.reason = reason
        super().__init__(f"质量规则 {rule_id} 错误: {reason}")


class RuleValidationError(DataQualityAssessmentError):
    """规则验证异常"""
    def __init__(self, rule_id: str, dataset_id: str, reason: str):
        self.rule_id = rule_id
        self.dataset_id = dataset_id
        self.reason = reason
        super().__init__(f"在数据集 {dataset_id} 上验证规则 {rule_id} 失败: {reason}")


class QualityReportGenerationError(DataQualityAssessmentError):
    """质量报告生成异常"""
    def __init__(self, dataset_id: str, reason: str):
        self.dataset_id = dataset_id
        self.reason = reason
        super().__init__(f"生成数据集 {dataset_id} 的质量报告失败: {reason}")


class QualityThresholdViolationError(DataQualityAssessmentError):
    """质量阈值违反异常"""
    def __init__(self, dataset_id: str, dimension: str, current_score: float, threshold: float):
        self.dataset_id = dataset_id
        self.dimension = dimension
        self.current_score = current_score
        self.threshold = threshold
        super().__init__(
            f"数据集 {dataset_id} 的 {dimension} 质量维度低于阈值 "
            f"(当前: {current_score}, 阈值: {threshold})"
        )


class QualityMonitoringError(DataQualityAssessmentError):
    """质量监控异常"""
    def __init__(self, dataset_id: str, reason: str):
        self.dataset_id = dataset_id
        self.reason = reason
        super().__init__(f"数据集 {dataset_id} 质量监控错误: {reason}")


class QualityAlertError(DataQualityAssessmentError):
    """质量告警异常"""
    def __init__(self, alert_id: str, reason: str):
        self.alert_id = alert_id
        self.reason = reason
        super().__init__(f"处理质量告警 {alert_id} 失败: {reason}")


class StatisticalAnalysisError(DataQualityAssessmentError):
    """统计分析异常"""
    def __init__(self, column: str, analysis_type: str, reason: str):
        self.column = column
        self.analysis_type = analysis_type
        self.reason = reason
        super().__init__(f"对列 {column} 执行 {analysis_type} 分析失败: {reason}")


class OutlierDetectionError(DataQualityAssessmentError):
    """异常值检测异常"""
    def __init__(self, column: str, method: str, reason: str):
        self.column = column
        self.method = method
        self.reason = reason
        super().__init__(f"使用 {method} 方法检测列 {column} 的异常值失败: {reason}")


class DuplicateDetectionError(DataQualityAssessmentError):
    """重复检测异常"""
    def __init__(self, dataset_id: str, reason: str):
        self.dataset_id = dataset_id
        self.reason = reason
        super().__init__(f"检测数据集 {dataset_id} 的重复记录失败: {reason}")


class DataProfilingError(DataQualityAssessmentError):
    """数据剖析异常"""
    def __init__(self, dataset_id: str, reason: str):
        self.dataset_id = dataset_id
        self.reason = reason
        super().__init__(f"数据集 {dataset_id} 剖析失败: {reason}")