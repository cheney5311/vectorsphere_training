"""数据质量模型定义

定义数据质量管理相关的Pydantic模型和枚举类型。
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


# ============================================================================
# 枚举类型定义
# ============================================================================

class QualityDimension(str, Enum):
    """数据质量维度"""
    COMPLETENESS = "completeness"           # 完整性
    CONSISTENCY = "consistency"             # 一致性
    ACCURACY = "accuracy"                   # 准确性
    TIMELINESS = "timeliness"              # 时效性
    UNIQUENESS = "uniqueness"               # 唯一性
    VALIDITY = "validity"                   # 有效性
    INTEGRITY = "integrity"                 # 完整性约束
    CONFORMITY = "conformity"               # 符合性


class IssueSeverity(str, Enum):
    """问题严重程度"""
    CRITICAL = "critical"                   # 严重
    HIGH = "high"                           # 高
    MEDIUM = "medium"                       # 中等
    LOW = "low"                             # 低
    INFO = "info"                           # 信息


class IssueType(str, Enum):
    """数据问题类型"""
    MISSING_VALUES = "missing_values"       # 缺失值
    DUPLICATE_RECORDS = "duplicate_records" # 重复记录
    OUTLIERS = "outliers"                   # 异常值
    INCONSISTENT_FORMAT = "inconsistent_format"  # 格式不一致
    INVALID_VALUES = "invalid_values"       # 无效值
    DATA_TYPE_MISMATCH = "data_type_mismatch"    # 数据类型不匹配
    REFERENTIAL_INTEGRITY = "referential_integrity"  # 引用完整性
    BUSINESS_RULE_VIOLATION = "business_rule_violation"  # 业务规则违反
    ENCODING_ERROR = "encoding_error"       # 编码错误
    RANGE_VIOLATION = "range_violation"     # 范围违规
    PATTERN_MISMATCH = "pattern_mismatch"   # 模式不匹配
    STATISTICAL_ANOMALY = "statistical_anomaly"  # 统计异常


class CleaningStrategy(str, Enum):
    """清理策略"""
    # 缺失值处理策略
    DROP_ROWS = "drop_rows"                 # 删除包含缺失值的行
    DROP_COLUMNS = "drop_columns"           # 删除缺失值过多的列
    FILL_MEAN = "fill_mean"                 # 均值填充
    FILL_MEDIAN = "fill_median"             # 中位数填充
    FILL_MODE = "fill_mode"                 # 众数填充
    FILL_CONSTANT = "fill_constant"         # 常量填充
    FILL_FORWARD = "fill_forward"           # 前向填充
    FILL_BACKWARD = "fill_backward"         # 后向填充
    FILL_INTERPOLATE = "fill_interpolate"   # 插值填充
    
    # 重复值处理策略
    DROP_DUPLICATES = "drop_duplicates"     # 删除重复记录
    KEEP_FIRST = "keep_first"               # 保留第一条
    KEEP_LAST = "keep_last"                 # 保留最后一条
    
    # 异常值处理策略
    CLIP_VALUES = "clip_values"             # 截断值
    REMOVE_OUTLIERS = "remove_outliers"     # 删除异常值
    REPLACE_WITH_BOUNDARY = "replace_with_boundary"  # 替换为边界值
    WINSORIZE = "winsorize"                 # Winsorize处理
    
    # 格式处理策略
    STANDARDIZE_FORMAT = "standardize_format"  # 标准化格式
    TYPE_CONVERSION = "type_conversion"     # 类型转换


class QualityRuleType(str, Enum):
    """质量规则类型"""
    COMPLETENESS_RULE = "completeness_rule"   # 完整性规则
    RANGE_RULE = "range_rule"                 # 范围规则
    PATTERN_RULE = "pattern_rule"             # 模式规则
    UNIQUENESS_RULE = "uniqueness_rule"       # 唯一性规则
    REFERENTIAL_RULE = "referential_rule"     # 引用规则
    CUSTOM_RULE = "custom_rule"               # 自定义规则
    STATISTICAL_RULE = "statistical_rule"     # 统计规则
    BUSINESS_RULE = "business_rule"           # 业务规则


class QualityAssessmentStatus(str, Enum):
    """质量评估状态"""
    PENDING = "pending"                       # 待评估
    IN_PROGRESS = "in_progress"               # 评估中
    COMPLETED = "completed"                   # 已完成
    FAILED = "failed"                         # 评估失败


class CleaningStatus(str, Enum):
    """清理状态"""
    PENDING = "pending"                       # 待清理
    IN_PROGRESS = "in_progress"               # 清理中
    COMPLETED = "completed"                   # 已完成
    FAILED = "failed"                         # 清理失败
    ROLLED_BACK = "rolled_back"               # 已回滚


# ============================================================================
# 质量评估相关模型
# ============================================================================

class ColumnQualityMetrics(BaseModel):
    """列级质量指标"""
    column_name: str = Field(..., description="列名")
    data_type: str = Field(..., description="数据类型")
    total_count: int = Field(default=0, description="总记录数")
    null_count: int = Field(default=0, description="空值数量")
    null_rate: float = Field(default=0.0, description="空值率")
    distinct_count: int = Field(default=0, description="唯一值数量")
    distinct_rate: float = Field(default=0.0, description="唯一值率")
    min_value: Optional[Any] = Field(default=None, description="最小值")
    max_value: Optional[Any] = Field(default=None, description="最大值")
    mean_value: Optional[float] = Field(default=None, description="平均值")
    median_value: Optional[float] = Field(default=None, description="中位数")
    std_value: Optional[float] = Field(default=None, description="标准差")
    outlier_count: int = Field(default=0, description="异常值数量")
    outlier_rate: float = Field(default=0.0, description="异常值率")
    pattern_consistency: Optional[float] = Field(default=None, description="模式一致性")
    quality_score: float = Field(default=0.0, description="列质量评分")


class DimensionScore(BaseModel):
    """维度评分"""
    dimension: QualityDimension = Field(..., description="质量维度")
    score: float = Field(..., ge=0.0, le=1.0, description="评分 (0-1)")
    weight: float = Field(default=1.0, ge=0.0, description="权重")
    details: Dict[str, Any] = Field(default_factory=dict, description="详细信息")


class QualityMetrics(BaseModel):
    """数据质量指标"""
    dataset_id: str = Field(..., description="数据集ID")
    total_records: int = Field(default=0, description="总记录数")
    total_columns: int = Field(default=0, description="总列数")
    missing_values_count: int = Field(default=0, description="缺失值数量")
    missing_values_rate: float = Field(default=0.0, description="缺失值率")
    duplicate_records_count: int = Field(default=0, description="重复记录数量")
    duplicate_records_rate: float = Field(default=0.0, description="重复记录率")
    outliers_count: int = Field(default=0, description="异常值数量")
    outliers_rate: float = Field(default=0.0, description="异常值率")
    invalid_values_count: int = Field(default=0, description="无效值数量")
    invalid_values_rate: float = Field(default=0.0, description="无效值率")
    dimension_scores: List[DimensionScore] = Field(default_factory=list, description="维度评分")
    column_metrics: List[ColumnQualityMetrics] = Field(default_factory=list, description="列级指标")
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0, description="总体质量评分")
    assessed_at: datetime = Field(default_factory=datetime.utcnow, description="评估时间")


# ============================================================================
# 数据问题相关模型
# ============================================================================

class DataIssue(BaseModel):
    """数据问题"""
    issue_id: str = Field(..., description="问题ID")
    issue_type: IssueType = Field(..., description="问题类型")
    severity: IssueSeverity = Field(..., description="严重程度")
    column_name: Optional[str] = Field(default=None, description="相关列名")
    description: str = Field(..., description="问题描述")
    affected_count: int = Field(default=0, description="影响记录数")
    affected_rate: float = Field(default=0.0, description="影响比例")
    sample_values: List[Any] = Field(default_factory=list, description="示例值")
    recommendation: str = Field(default="", description="建议处理方式")
    auto_fixable: bool = Field(default=False, description="是否可自动修复")
    detected_at: datetime = Field(default_factory=datetime.utcnow, description="检测时间")


class IssueDetectionResult(BaseModel):
    """问题检测结果"""
    dataset_id: str = Field(..., description="数据集ID")
    total_issues: int = Field(default=0, description="总问题数")
    critical_count: int = Field(default=0, description="严重问题数")
    high_count: int = Field(default=0, description="高级问题数")
    medium_count: int = Field(default=0, description="中级问题数")
    low_count: int = Field(default=0, description="低级问题数")
    issues: List[DataIssue] = Field(default_factory=list, description="问题列表")
    summary_by_type: Dict[str, int] = Field(default_factory=dict, description="按类型统计")
    detected_at: datetime = Field(default_factory=datetime.utcnow, description="检测时间")


# ============================================================================
# 数据清理相关模型
# ============================================================================

class CleaningOperation(BaseModel):
    """清理操作"""
    operation_id: str = Field(..., description="操作ID")
    strategy: CleaningStrategy = Field(..., description="清理策略")
    target_column: Optional[str] = Field(default=None, description="目标列")
    target_issue_type: Optional[IssueType] = Field(default=None, description="目标问题类型")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="操作参数")
    priority: int = Field(default=0, description="优先级")
    enabled: bool = Field(default=True, description="是否启用")


class CleaningConfig(BaseModel):
    """清理配置"""
    operations: List[CleaningOperation] = Field(default_factory=list, description="清理操作列表")
    
    # 全局配置
    remove_duplicates: bool = Field(default=True, description="是否删除重复记录")
    handle_missing_values: bool = Field(default=True, description="是否处理缺失值")
    handle_outliers: bool = Field(default=False, description="是否处理异常值")
    
    # 缺失值处理配置
    missing_value_strategy: CleaningStrategy = Field(
        default=CleaningStrategy.FILL_MEDIAN, 
        description="缺失值默认处理策略"
    )
    missing_threshold: float = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0, 
        description="缺失率阈值，超过则删除列"
    )
    
    # 异常值处理配置
    outlier_strategy: CleaningStrategy = Field(
        default=CleaningStrategy.CLIP_VALUES,
        description="异常值默认处理策略"
    )
    outlier_std_threshold: float = Field(
        default=3.0,
        gt=0.0,
        description="异常值标准差阈值"
    )
    
    # 其他配置
    preserve_original: bool = Field(default=True, description="是否保留原始数据备份")
    dry_run: bool = Field(default=False, description="是否为预演模式")


class CleaningOperationResult(BaseModel):
    """清理操作结果"""
    operation_id: str = Field(..., description="操作ID")
    strategy: CleaningStrategy = Field(..., description="清理策略")
    target_column: Optional[str] = Field(default=None, description="目标列")
    success: bool = Field(default=False, description="是否成功")
    records_affected: int = Field(default=0, description="影响记录数")
    execution_time_ms: float = Field(default=0.0, description="执行时间(毫秒)")
    error_message: Optional[str] = Field(default=None, description="错误信息")


class CleaningResult(BaseModel):
    """清理结果"""
    dataset_id: str = Field(..., description="数据集ID")
    cleaned_dataset_id: str = Field(..., description="清理后数据集ID")
    status: CleaningStatus = Field(default=CleaningStatus.COMPLETED, description="清理状态")
    original_record_count: int = Field(default=0, description="原始记录数")
    cleaned_record_count: int = Field(default=0, description="清理后记录数")
    total_records_affected: int = Field(default=0, description="总影响记录数")
    operation_results: List[CleaningOperationResult] = Field(
        default_factory=list, 
        description="操作结果列表"
    )
    original_quality_score: float = Field(default=0.0, description="原始质量评分")
    cleaned_quality_score: float = Field(default=0.0, description="清理后质量评分")
    improvement: float = Field(default=0.0, description="质量提升")
    backup_path: Optional[str] = Field(default=None, description="原始数据备份路径")
    cleaned_at: datetime = Field(default_factory=datetime.utcnow, description="清理时间")
    execution_time_ms: float = Field(default=0.0, description="总执行时间(毫秒)")


# ============================================================================
# 质量规则相关模型
# ============================================================================

class QualityRule(BaseModel):
    """质量规则"""
    rule_id: str = Field(..., description="规则ID")
    name: str = Field(..., description="规则名称")
    description: str = Field(default="", description="规则描述")
    rule_type: QualityRuleType = Field(..., description="规则类型")
    target_column: Optional[str] = Field(default=None, description="目标列")
    condition: str = Field(..., description="规则条件表达式")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="规则参数")
    severity: IssueSeverity = Field(default=IssueSeverity.MEDIUM, description="违反规则的严重程度")
    enabled: bool = Field(default=True, description="是否启用")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")


class RuleValidationResult(BaseModel):
    """规则验证结果"""
    rule_id: str = Field(..., description="规则ID")
    rule_name: str = Field(..., description="规则名称")
    passed: bool = Field(default=False, description="是否通过")
    violations_count: int = Field(default=0, description="违反数量")
    violation_rate: float = Field(default=0.0, description="违反率")
    sample_violations: List[Any] = Field(default_factory=list, description="示例违规记录")
    execution_time_ms: float = Field(default=0.0, description="执行时间(毫秒)")


class RuleSetValidationResult(BaseModel):
    """规则集验证结果"""
    dataset_id: str = Field(..., description="数据集ID")
    total_rules: int = Field(default=0, description="总规则数")
    passed_rules: int = Field(default=0, description="通过规则数")
    failed_rules: int = Field(default=0, description="失败规则数")
    pass_rate: float = Field(default=0.0, description="通过率")
    results: List[RuleValidationResult] = Field(default_factory=list, description="验证结果列表")
    validated_at: datetime = Field(default_factory=datetime.utcnow, description="验证时间")


# ============================================================================
# 质量报告相关模型
# ============================================================================

class QualityTrend(BaseModel):
    """质量趋势"""
    timestamp: datetime = Field(..., description="时间戳")
    overall_score: float = Field(..., description="总体评分")
    dimension_scores: Dict[str, float] = Field(default_factory=dict, description="维度评分")


class QualityReport(BaseModel):
    """质量报告"""
    report_id: str = Field(..., description="报告ID")
    dataset_id: str = Field(..., description="数据集ID")
    dataset_name: str = Field(default="", description="数据集名称")
    
    # 基本信息
    total_records: int = Field(default=0, description="总记录数")
    total_columns: int = Field(default=0, description="总列数")
    
    # 质量评估
    quality_metrics: QualityMetrics = Field(..., description="质量指标")
    overall_score: float = Field(default=0.0, description="总体评分")
    
    # 问题检测
    issue_detection: IssueDetectionResult = Field(..., description="问题检测结果")
    
    # 规则验证
    rule_validation: Optional[RuleSetValidationResult] = Field(
        default=None, 
        description="规则验证结果"
    )
    
    # 清理建议
    cleaning_recommendations: List[CleaningOperation] = Field(
        default_factory=list, 
        description="清理建议"
    )
    
    # 历史趋势
    quality_trends: List[QualityTrend] = Field(default_factory=list, description="质量趋势")
    
    # 总结
    summary: str = Field(default="", description="报告总结")
    recommendations: List[str] = Field(default_factory=list, description="改进建议")
    
    # 元数据
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="生成时间")
    generated_by: Optional[str] = Field(default=None, description="生成者")


# ============================================================================
# API请求/响应模型
# ============================================================================

class AssessQualityRequest(BaseModel):
    """评估质量请求"""
    dimensions: Optional[List[QualityDimension]] = Field(
        default=None, 
        description="要评估的质量维度"
    )
    include_column_metrics: bool = Field(default=True, description="是否包含列级指标")
    sample_size: Optional[int] = Field(default=None, description="采样大小")


class AssessQualityResponse(BaseModel):
    """评估质量响应"""
    success: bool = Field(default=True, description="是否成功")
    dataset_id: str = Field(..., description="数据集ID")
    metrics: QualityMetrics = Field(..., description="质量指标")


class DetectIssuesRequest(BaseModel):
    """检测问题请求"""
    issue_types: Optional[List[IssueType]] = Field(default=None, description="要检测的问题类型")
    severity_threshold: IssueSeverity = Field(
        default=IssueSeverity.LOW, 
        description="严重程度阈值"
    )
    max_issues: int = Field(default=100, description="最大返回问题数")
    include_samples: bool = Field(default=True, description="是否包含示例值")
    sample_count: int = Field(default=5, description="示例数量")


class DetectIssuesResponse(BaseModel):
    """检测问题响应"""
    success: bool = Field(default=True, description="是否成功")
    dataset_id: str = Field(..., description="数据集ID")
    result: IssueDetectionResult = Field(..., description="检测结果")


class CleanDataRequest(BaseModel):
    """清理数据请求"""
    config: CleaningConfig = Field(default_factory=CleaningConfig, description="清理配置")
    create_new_dataset: bool = Field(default=True, description="是否创建新数据集")
    new_dataset_name: Optional[str] = Field(default=None, description="新数据集名称")


class CleanDataResponse(BaseModel):
    """清理数据响应"""
    success: bool = Field(default=True, description="是否成功")
    dataset_id: str = Field(..., description="原数据集ID")
    result: CleaningResult = Field(..., description="清理结果")


class ValidateRulesRequest(BaseModel):
    """验证规则请求"""
    rules: List[QualityRule] = Field(..., description="要验证的规则列表")
    stop_on_failure: bool = Field(default=False, description="遇到失败是否停止")


class ValidateRulesResponse(BaseModel):
    """验证规则响应"""
    success: bool = Field(default=True, description="是否成功")
    dataset_id: str = Field(..., description="数据集ID")
    result: RuleSetValidationResult = Field(..., description="验证结果")


class GenerateReportRequest(BaseModel):
    """生成报告请求"""
    include_trends: bool = Field(default=True, description="是否包含趋势分析")
    trend_period_days: int = Field(default=30, description="趋势分析周期(天)")
    include_recommendations: bool = Field(default=True, description="是否包含改进建议")


class GenerateReportResponse(BaseModel):
    """生成报告响应"""
    success: bool = Field(default=True, description="是否成功")
    report: QualityReport = Field(..., description="质量报告")


# ============================================================================
# 质量监控相关模型
# ============================================================================

class QualityThreshold(BaseModel):
    """质量阈值"""
    dimension: QualityDimension = Field(..., description="质量维度")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="最低分数")
    warning_score: float = Field(default=0.7, ge=0.0, le=1.0, description="警告分数")


class QualityAlert(BaseModel):
    """质量告警"""
    alert_id: str = Field(..., description="告警ID")
    dataset_id: str = Field(..., description="数据集ID")
    dimension: QualityDimension = Field(..., description="质量维度")
    current_score: float = Field(..., description="当前分数")
    threshold_score: float = Field(..., description="阈值分数")
    severity: IssueSeverity = Field(..., description="告警级别")
    message: str = Field(..., description="告警消息")
    triggered_at: datetime = Field(default_factory=datetime.utcnow, description="触发时间")
    acknowledged: bool = Field(default=False, description="是否已确认")


class QualityMonitoringConfig(BaseModel):
    """质量监控配置"""
    dataset_id: str = Field(..., description="数据集ID")
    enabled: bool = Field(default=True, description="是否启用监控")
    thresholds: List[QualityThreshold] = Field(default_factory=list, description="质量阈值")
    check_interval_minutes: int = Field(default=60, description="检查间隔(分钟)")
    alert_channels: List[str] = Field(default_factory=list, description="告警渠道")


class QualityMonitoringStatus(BaseModel):
    """质量监控状态"""
    dataset_id: str = Field(..., description="数据集ID")
    monitoring_enabled: bool = Field(default=False, description="监控是否启用")
    last_check_at: Optional[datetime] = Field(default=None, description="最后检查时间")
    next_check_at: Optional[datetime] = Field(default=None, description="下次检查时间")
    current_score: float = Field(default=0.0, description="当前质量分数")
    active_alerts: List[QualityAlert] = Field(default_factory=list, description="活跃告警")
