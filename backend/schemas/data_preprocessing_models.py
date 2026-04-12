"""数据预处理模块数据模型

定义数据预处理相关的请求和响应模型。
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


# ============================================================================
# 预处理相关枚举
# ============================================================================

class PreprocessingStatus(str, Enum):
    """预处理状态枚举"""
    PENDING = "pending"                   # 等待处理
    PROCESSING = "processing"             # 处理中
    COMPLETED = "completed"               # 已完成
    FAILED = "failed"                     # 失败
    CANCELLED = "cancelled"               # 已取消


class PreprocessingOperationType(str, Enum):
    """预处理操作类型枚举"""
    # 数据清洗
    REMOVE_DUPLICATES = "remove_duplicates"           # 去重
    HANDLE_MISSING = "handle_missing"                 # 处理缺失值
    REMOVE_OUTLIERS = "remove_outliers"               # 去除异常值
    FILTER_ROWS = "filter_rows"                       # 过滤行
    FILTER_COLUMNS = "filter_columns"                 # 过滤列
    
    # 数据转换
    NORMALIZE = "normalize"                           # 标准化
    STANDARDIZE = "standardize"                       # 归一化
    ENCODE_CATEGORICAL = "encode_categorical"         # 类别编码
    TOKENIZE = "tokenize"                             # 分词
    VECTORIZE = "vectorize"                           # 向量化
    
    # 数据增强
    OVERSAMPLE = "oversample"                         # 过采样
    UNDERSAMPLE = "undersample"                       # 欠采样
    SMOTE = "smote"                                   # SMOTE采样
    AUGMENT_TEXT = "augment_text"                     # 文本增强
    AUGMENT_IMAGE = "augment_image"                   # 图像增强
    
    # 特征工程
    CREATE_FEATURE = "create_feature"                 # 创建特征
    SELECT_FEATURE = "select_feature"                 # 特征选择
    TRANSFORM_FEATURE = "transform_feature"           # 特征转换
    REDUCE_DIMENSION = "reduce_dimension"             # 降维
    
    # 数据分割
    SPLIT_TRAIN_TEST = "split_train_test"             # 训练测试分割
    SPLIT_STRATIFIED = "split_stratified"             # 分层分割
    CROSS_VALIDATION = "cross_validation"             # 交叉验证


class MissingValueStrategy(str, Enum):
    """缺失值处理策略"""
    DROP = "drop"                         # 删除
    FILL_MEAN = "fill_mean"               # 均值填充
    FILL_MEDIAN = "fill_median"           # 中位数填充
    FILL_MODE = "fill_mode"               # 众数填充
    FILL_CONSTANT = "fill_constant"       # 常量填充
    FILL_FORWARD = "fill_forward"         # 前向填充
    FILL_BACKWARD = "fill_backward"       # 后向填充
    INTERPOLATE = "interpolate"           # 插值填充
    KNN_IMPUTE = "knn_impute"             # KNN填充


class OutlierDetectionMethod(str, Enum):
    """异常值检测方法"""
    IQR = "iqr"                           # 四分位距
    ZSCORE = "zscore"                     # Z分数
    ISOLATION_FOREST = "isolation_forest" # 孤立森林
    LOF = "lof"                           # 局部异常因子
    DBSCAN = "dbscan"                     # DBSCAN聚类
    PERCENTILE = "percentile"             # 百分位数


class NormalizationMethod(str, Enum):
    """标准化方法"""
    MIN_MAX = "min_max"                   # 最小最大标准化
    Z_SCORE = "z_score"                   # Z分数标准化
    MAX_ABS = "max_abs"                   # 最大绝对值标准化
    ROBUST = "robust"                     # 鲁棒标准化
    L1 = "l1"                             # L1范数
    L2 = "l2"                             # L2范数


class EncodingMethod(str, Enum):
    """编码方法"""
    ONE_HOT = "one_hot"                   # 独热编码
    LABEL = "label"                       # 标签编码
    ORDINAL = "ordinal"                   # 有序编码
    BINARY = "binary"                     # 二进制编码
    TARGET = "target"                     # 目标编码
    FREQUENCY = "frequency"               # 频率编码
    HASH = "hash"                         # 哈希编码


class FeatureSelectionMethod(str, Enum):
    """特征选择方法"""
    VARIANCE = "variance"                 # 方差阈值
    CORRELATION = "correlation"           # 相关性
    MUTUAL_INFO = "mutual_info"           # 互信息
    CHI2 = "chi2"                         # 卡方检验
    ANOVA = "anova"                       # 方差分析
    RFE = "rfe"                           # 递归特征消除
    LASSO = "lasso"                       # L1正则化
    TREE_IMPORTANCE = "tree_importance"   # 树模型重要性
    BORUTA = "boruta"                     # Boruta算法


class DimensionReductionMethod(str, Enum):
    """降维方法"""
    PCA = "pca"                           # 主成分分析
    LDA = "lda"                           # 线性判别分析
    TSNE = "tsne"                         # t-SNE
    UMAP = "umap"                         # UMAP
    SVD = "svd"                           # 奇异值分解
    NMF = "nmf"                           # 非负矩阵分解
    AUTOENCODER = "autoencoder"           # 自编码器


class TextAugmentationMethod(str, Enum):
    """文本增强方法"""
    SYNONYM_REPLACEMENT = "synonym_replacement"   # 同义词替换
    RANDOM_INSERTION = "random_insertion"         # 随机插入
    RANDOM_SWAP = "random_swap"                   # 随机交换
    RANDOM_DELETION = "random_deletion"           # 随机删除
    BACK_TRANSLATION = "back_translation"         # 回译
    CONTEXTUAL_EMBEDDING = "contextual_embedding" # 上下文嵌入


class ImageAugmentationMethod(str, Enum):
    """图像增强方法"""
    FLIP = "flip"                         # 翻转
    ROTATE = "rotate"                     # 旋转
    SCALE = "scale"                       # 缩放
    CROP = "crop"                         # 裁剪
    BRIGHTNESS = "brightness"             # 亮度调整
    CONTRAST = "contrast"                 # 对比度调整
    NOISE = "noise"                       # 添加噪声
    BLUR = "blur"                         # 模糊
    MIXUP = "mixup"                       # Mixup
    CUTOUT = "cutout"                     # Cutout


# ============================================================================
# 预处理操作配置模型
# ============================================================================

class MissingValueConfig(BaseModel):
    """缺失值处理配置"""
    strategy: MissingValueStrategy = Field(MissingValueStrategy.DROP, description="处理策略")
    columns: Optional[List[str]] = Field(None, description="目标列（不填则处理所有列）")
    fill_value: Optional[Any] = Field(None, description="填充值（常量填充时使用）")
    threshold: float = Field(0.5, ge=0, le=1, description="缺失比例阈值（超过则删除列）")
    k_neighbors: int = Field(5, ge=1, description="KNN填充的邻居数")


class OutlierConfig(BaseModel):
    """异常值处理配置"""
    method: OutlierDetectionMethod = Field(OutlierDetectionMethod.IQR, description="检测方法")
    columns: Optional[List[str]] = Field(None, description="目标列")
    threshold: float = Field(1.5, ge=0, description="阈值（IQR倍数/Z分数）")
    action: str = Field("remove", description="处理方式：remove/cap/transform")
    contamination: float = Field(0.1, ge=0, le=0.5, description="污染比例")


class NormalizationConfig(BaseModel):
    """标准化配置"""
    method: NormalizationMethod = Field(NormalizationMethod.MIN_MAX, description="标准化方法")
    columns: Optional[List[str]] = Field(None, description="目标列")
    feature_range: tuple = Field((0, 1), description="特征范围（min_max时使用）")


class EncodingConfig(BaseModel):
    """编码配置"""
    method: EncodingMethod = Field(EncodingMethod.ONE_HOT, description="编码方法")
    columns: List[str] = Field(..., description="目标列")
    handle_unknown: str = Field("ignore", description="处理未知值：ignore/error")
    max_categories: Optional[int] = Field(None, description="最大类别数")


class TokenizationConfig(BaseModel):
    """分词配置"""
    columns: List[str] = Field(..., description="目标文本列")
    language: str = Field("zh", description="语言：zh/en")
    remove_stopwords: bool = Field(True, description="是否去除停用词")
    lowercase: bool = Field(True, description="是否转小写")
    min_token_length: int = Field(1, ge=1, description="最小词长度")
    max_tokens: Optional[int] = Field(None, description="最大词数")
    custom_stopwords: Optional[List[str]] = Field(None, description="自定义停用词")


class VectorizationConfig(BaseModel):
    """向量化配置"""
    columns: List[str] = Field(..., description="目标文本列")
    method: str = Field("tfidf", description="向量化方法：tfidf/count/word2vec/bert")
    max_features: Optional[int] = Field(None, description="最大特征数")
    ngram_range: tuple = Field((1, 1), description="N-gram范围")
    min_df: float = Field(1, description="最小文档频率")
    max_df: float = Field(1.0, description="最大文档频率")


class FilterConfig(BaseModel):
    """数据过滤配置"""
    conditions: List[Dict[str, Any]] = Field(..., description="过滤条件列表")
    logic: str = Field("and", description="条件逻辑：and/or")
    keep_filtered: bool = Field(False, description="是否保留被过滤的数据")


# ============================================================================
# 特征工程配置模型
# ============================================================================

class FeatureCreationConfig(BaseModel):
    """特征创建配置"""
    name: str = Field(..., description="新特征名称")
    expression: str = Field(..., description="计算表达式")
    description: Optional[str] = Field(None, description="特征描述")


class FeatureSelectionConfig(BaseModel):
    """特征选择配置"""
    method: FeatureSelectionMethod = Field(
        FeatureSelectionMethod.VARIANCE, 
        description="选择方法"
    )
    target_column: Optional[str] = Field(None, description="目标列（监督方法需要）")
    n_features: Optional[int] = Field(None, description="选择的特征数")
    threshold: Optional[float] = Field(None, description="阈值")
    exclude_columns: Optional[List[str]] = Field(None, description="排除的列")


class DimensionReductionConfig(BaseModel):
    """降维配置"""
    method: DimensionReductionMethod = Field(DimensionReductionMethod.PCA, description="降维方法")
    n_components: int = Field(2, ge=1, description="目标维度数")
    columns: Optional[List[str]] = Field(None, description="目标列")
    random_state: int = Field(42, description="随机种子")


class FeatureTransformConfig(BaseModel):
    """特征转换配置"""
    columns: List[str] = Field(..., description="目标列")
    transform_type: str = Field("log", description="转换类型：log/sqrt/square/exp/box-cox")
    inverse: bool = Field(False, description="是否逆转换")


# ============================================================================
# 数据增强配置模型
# ============================================================================

class TextAugmentationConfig(BaseModel):
    """文本增强配置"""
    methods: List[TextAugmentationMethod] = Field(
        [TextAugmentationMethod.SYNONYM_REPLACEMENT],
        description="增强方法列表"
    )
    columns: List[str] = Field(..., description="目标文本列")
    augment_ratio: float = Field(0.3, ge=0, le=1, description="增强比例")
    num_augment: int = Field(1, ge=1, description="每条数据增强次数")
    language: str = Field("zh", description="语言")


class ImageAugmentationConfig(BaseModel):
    """图像增强配置"""
    methods: List[ImageAugmentationMethod] = Field(
        [ImageAugmentationMethod.FLIP, ImageAugmentationMethod.ROTATE],
        description="增强方法列表"
    )
    columns: List[str] = Field(..., description="目标图像列/路径")
    augment_ratio: float = Field(0.3, ge=0, le=1, description="增强比例")
    num_augment: int = Field(1, ge=1, description="每张图像增强次数")
    # 各方法参数
    rotation_range: int = Field(30, ge=0, le=180, description="旋转角度范围")
    brightness_range: tuple = Field((0.8, 1.2), description="亮度范围")
    scale_range: tuple = Field((0.8, 1.2), description="缩放范围")


class SamplingConfig(BaseModel):
    """采样配置"""
    method: str = Field("oversample", description="采样方法：oversample/undersample/smote")
    target_column: str = Field(..., description="目标列（类别列）")
    sampling_strategy: Union[str, Dict[str, int]] = Field(
        "auto", 
        description="采样策略：auto/minority/majority/not majority/all 或具体数量字典"
    )
    random_state: int = Field(42, description="随机种子")
    k_neighbors: int = Field(5, ge=1, description="SMOTE的近邻数")


# ============================================================================
# 数据分割配置模型
# ============================================================================

class DataSplitConfig(BaseModel):
    """数据分割配置"""
    train_ratio: float = Field(0.8, ge=0.1, le=0.95, description="训练集比例")
    val_ratio: float = Field(0.1, ge=0, le=0.4, description="验证集比例")
    test_ratio: float = Field(0.1, ge=0, le=0.4, description="测试集比例")
    stratify_column: Optional[str] = Field(None, description="分层采样目标列")
    shuffle: bool = Field(True, description="是否打乱数据")
    random_state: int = Field(42, description="随机种子")
    
    @validator('test_ratio')
    def validate_ratios(cls, v, values):
        train = values.get('train_ratio', 0)
        val = values.get('val_ratio', 0)
        if abs(train + val + v - 1.0) > 0.001:
            raise ValueError("训练、验证、测试比例之和必须为1")
        return v


class CrossValidationConfig(BaseModel):
    """交叉验证配置"""
    n_folds: int = Field(5, ge=2, le=20, description="折数")
    stratify_column: Optional[str] = Field(None, description="分层目标列")
    shuffle: bool = Field(True, description="是否打乱")
    random_state: int = Field(42, description="随机种子")
    group_column: Optional[str] = Field(None, description="分组列（Group K-Fold）")


# ============================================================================
# 预处理请求和响应模型
# ============================================================================

class PreprocessingOperation(BaseModel):
    """预处理操作定义"""
    operation_type: PreprocessingOperationType = Field(..., description="操作类型")
    config: Dict[str, Any] = Field(default_factory=dict, description="操作配置")
    order: int = Field(0, ge=0, description="执行顺序")
    enabled: bool = Field(True, description="是否启用")


class PreprocessingPipelineConfig(BaseModel):
    """预处理流水线配置"""
    operations: List[PreprocessingOperation] = Field(..., description="操作列表")
    save_intermediate: bool = Field(False, description="是否保存中间结果")
    parallel_execution: bool = Field(False, description="是否并行执行")
    fail_on_error: bool = Field(True, description="出错时是否停止")


class PreprocessDatasetRequest(BaseModel):
    """预处理数据集请求"""
    pipeline_config: Optional[PreprocessingPipelineConfig] = Field(
        None, 
        description="流水线配置（高级模式）"
    )
    # 简单模式配置
    normalize: bool = Field(False, description="是否标准化")
    tokenize: bool = Field(False, description="是否分词")
    filter_invalid: bool = Field(False, description="是否过滤无效数据")
    remove_duplicates: bool = Field(False, description="是否去重")
    handle_missing: Optional[MissingValueConfig] = Field(None, description="缺失值处理")
    handle_outliers: Optional[OutlierConfig] = Field(None, description="异常值处理")


class FeatureEngineeringRequest(BaseModel):
    """特征工程请求"""
    # 特征创建
    create_features: Optional[List[FeatureCreationConfig]] = Field(
        None, 
        description="要创建的特征列表"
    )
    # 特征选择
    feature_selection: Optional[FeatureSelectionConfig] = Field(
        None, 
        description="特征选择配置"
    )
    # 特征转换
    feature_transform: Optional[List[FeatureTransformConfig]] = Field(
        None, 
        description="特征转换配置列表"
    )
    # 降维
    dimension_reduction: Optional[DimensionReductionConfig] = Field(
        None, 
        description="降维配置"
    )
    # 编码
    encoding: Optional[List[EncodingConfig]] = Field(
        None, 
        description="编码配置列表"
    )


class DataAugmentationRequest(BaseModel):
    """数据增强请求"""
    augmentation_type: str = Field("text", description="增强类型：text/image/tabular")
    text_config: Optional[TextAugmentationConfig] = Field(None, description="文本增强配置")
    image_config: Optional[ImageAugmentationConfig] = Field(None, description="图像增强配置")
    sampling_config: Optional[SamplingConfig] = Field(None, description="采样配置")
    target_size: Optional[int] = Field(None, description="目标数据量")
    keep_original: bool = Field(True, description="是否保留原始数据")


class DataSplitRequest(BaseModel):
    """数据分割请求"""
    split_config: DataSplitConfig = Field(
        default_factory=DataSplitConfig,
        description="分割配置"
    )
    cross_validation: Optional[CrossValidationConfig] = Field(
        None, 
        description="交叉验证配置（与split_config二选一）"
    )
    output_format: str = Field("separate", description="输出格式：separate/combined")
    create_new_datasets: bool = Field(True, description="是否创建新数据集")


# ============================================================================
# 预处理结果模型
# ============================================================================

class OperationResult(BaseModel):
    """单个操作的执行结果"""
    operation_type: str = Field(..., description="操作类型")
    status: PreprocessingStatus = Field(..., description="执行状态")
    started_at: datetime = Field(..., description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    duration_ms: Optional[int] = Field(None, description="执行耗时（毫秒）")
    rows_affected: int = Field(0, description="影响的行数")
    columns_affected: List[str] = Field(default_factory=list, description="影响的列")
    error_message: Optional[str] = Field(None, description="错误信息")
    details: Dict[str, Any] = Field(default_factory=dict, description="详细信息")


class PreprocessingResult(BaseModel):
    """预处理结果"""
    dataset_id: str = Field(..., description="数据集ID")
    task_id: str = Field(..., description="预处理任务ID")
    status: PreprocessingStatus = Field(..., description="总体状态")
    started_at: datetime = Field(..., description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    total_duration_ms: Optional[int] = Field(None, description="总耗时")
    operations_results: List[OperationResult] = Field(
        default_factory=list, 
        description="各操作结果"
    )
    
    # 数据统计
    original_rows: int = Field(0, description="原始行数")
    final_rows: int = Field(0, description="最终行数")
    original_columns: int = Field(0, description="原始列数")
    final_columns: int = Field(0, description="最终列数")
    
    # 变更摘要
    rows_removed: int = Field(0, description="删除的行数")
    rows_added: int = Field(0, description="添加的行数")
    columns_removed: List[str] = Field(default_factory=list, description="删除的列")
    columns_added: List[str] = Field(default_factory=list, description="添加的列")
    columns_modified: List[str] = Field(default_factory=list, description="修改的列")


class FeatureEngineeringResult(BaseModel):
    """特征工程结果"""
    dataset_id: str = Field(..., description="数据集ID")
    task_id: str = Field(..., description="任务ID")
    status: PreprocessingStatus = Field(..., description="状态")
    
    # 特征统计
    features_created: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="创建的特征信息"
    )
    features_selected: List[str] = Field(default_factory=list, description="选中的特征")
    features_removed: List[str] = Field(default_factory=list, description="移除的特征")
    features_transformed: List[str] = Field(default_factory=list, description="转换的特征")
    
    # 特征重要性（如果有）
    feature_importance: Optional[Dict[str, float]] = Field(
        None, 
        description="特征重要性得分"
    )
    
    engineered_at: datetime = Field(..., description="执行时间")


class AugmentationResult(BaseModel):
    """数据增强结果"""
    dataset_id: str = Field(..., description="数据集ID")
    task_id: str = Field(..., description="任务ID")
    status: PreprocessingStatus = Field(..., description="状态")
    
    # 增强统计
    original_samples: int = Field(0, description="原始样本数")
    generated_samples: int = Field(0, description="生成的样本数")
    final_samples: int = Field(0, description="最终样本数")
    
    # 各方法生成数量
    method_stats: Dict[str, int] = Field(default_factory=dict, description="各方法生成数量")
    
    # 类别分布（如果是平衡采样）
    original_distribution: Optional[Dict[str, int]] = Field(
        None, 
        description="原始类别分布"
    )
    final_distribution: Optional[Dict[str, int]] = Field(
        None, 
        description="最终类别分布"
    )
    
    augmented_at: datetime = Field(..., description="执行时间")


class SplitResult(BaseModel):
    """数据分割结果"""
    dataset_id: str = Field(..., description="原数据集ID")
    task_id: str = Field(..., description="任务ID")
    status: PreprocessingStatus = Field(..., description="状态")
    
    # 分割统计
    total_samples: int = Field(0, description="总样本数")
    train_samples: int = Field(0, description="训练集样本数")
    val_samples: int = Field(0, description="验证集样本数")
    test_samples: int = Field(0, description="测试集样本数")
    
    # 生成的数据集ID
    train_dataset_id: Optional[str] = Field(None, description="训练集数据集ID")
    val_dataset_id: Optional[str] = Field(None, description="验证集数据集ID")
    test_dataset_id: Optional[str] = Field(None, description="测试集数据集ID")
    
    # 交叉验证折（如果使用）
    cv_folds: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="交叉验证各折信息"
    )
    
    # 类别分布（分层采样时）
    train_distribution: Optional[Dict[str, int]] = Field(
        None, 
        description="训练集类别分布"
    )
    val_distribution: Optional[Dict[str, int]] = Field(
        None, 
        description="验证集类别分布"
    )
    test_distribution: Optional[Dict[str, int]] = Field(
        None, 
        description="测试集类别分布"
    )
    
    split_at: datetime = Field(..., description="分割时间")


# ============================================================================
# 预处理任务和历史模型
# ============================================================================

class PreprocessingTask(BaseModel):
    """预处理任务"""
    task_id: str = Field(..., description="任务ID")
    dataset_id: str = Field(..., description="数据集ID")
    user_id: str = Field(..., description="用户ID")
    tenant_id: Optional[str] = Field(None, description="租户ID")
    
    task_type: str = Field(..., description="任务类型：preprocessing/feature_engineering/augmentation/split")
    status: PreprocessingStatus = Field(..., description="任务状态")
    config: Dict[str, Any] = Field(default_factory=dict, description="任务配置")
    result: Optional[Dict[str, Any]] = Field(None, description="任务结果")
    
    created_at: datetime = Field(..., description="创建时间")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    error_message: Optional[str] = Field(None, description="错误信息")


class PreprocessingHistory(BaseModel):
    """预处理历史记录"""
    history_id: str = Field(..., description="历史记录ID")
    dataset_id: str = Field(..., description="数据集ID")
    task_id: str = Field(..., description="关联任务ID")
    
    operation_type: str = Field(..., description="操作类型")
    operation_config: Dict[str, Any] = Field(default_factory=dict, description="操作配置")
    operation_result: Dict[str, Any] = Field(default_factory=dict, description="操作结果")
    
    # 数据快照信息（用于回滚）
    snapshot_path: Optional[str] = Field(None, description="数据快照路径")
    can_rollback: bool = Field(True, description="是否可以回滚")
    
    executed_at: datetime = Field(..., description="执行时间")
    executed_by: str = Field(..., description="执行用户ID")


class ListPreprocessingTasksResponse(BaseModel):
    """预处理任务列表响应"""
    tasks: List[PreprocessingTask] = Field(..., description="任务列表")
    total: int = Field(..., description="总数")
    page: int = Field(..., description="当前页")
    page_size: int = Field(..., description="每页大小")
