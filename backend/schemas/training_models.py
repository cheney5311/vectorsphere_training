"""训练相关数据模型

定义训练会话、训练进度等训练相关数据模型。
"""

from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, Float, ForeignKey, Index, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from backend.schemas.base_models import Base, UUIDMixin, TimestampMixin, TenantMixin, GUID
from .enums import TrainingStatus, TrainingStage, TrainingScenario, TrainingMethod


class TrainingSession(Base, TimestampMixin, TenantMixin):
    """训练会话模型"""
    __tablename__ = 'training_sessions'
    
    session_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment="会话ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    model_id = Column(String(100), comment="模型ID")
    dataset_id = Column(String(100), comment="数据集ID")
    training_type = Column(String(50), nullable=False, index=True, comment="训练类型")
    status = Column(String(20), default='pending', index=True, comment="状态")
    config = Column(JSON, comment="训练配置")
    result = Column(JSON, comment="训练结果")
    error_message = Column(Text, comment="错误信息")
    progress = Column(Float, default=0.0, comment="进度")
    started_at = Column(DateTime, comment="开始时间")
    completed_at = Column(DateTime, comment="完成时间")
    
    # 关系
    progress_records = relationship("TrainingProgress", back_populates="session", cascade="all, delete-orphan")
    
    @property
    def id(self):
        """id 属性作为 session_id 的别名，用于兼容性"""
        return self.session_id
    
    @id.setter
    def id(self, value):
        """设置 id 属性时实际设置 session_id"""
        self.session_id = value
    
    def __repr__(self):
        return f"<TrainingSession(session_id='{self.session_id}', user_id='{self.user_id}', status='{self.status}')>"
    
    def to_dict(self):
        """将训练会话转换为字典"""
        result = {
            'id': self.session_id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'model_id': self.model_id,
            'dataset_id': self.dataset_id,
            'training_type': self.training_type,
            'status': self.status,
            'config': self.config,
            'result': self.result,
            'error_message': self.error_message,
            'progress': self.progress,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'created_at': getattr(self, 'created_at', None).isoformat() if getattr(self, 'created_at', None) else None,
            'updated_at': getattr(self, 'updated_at', None).isoformat() if getattr(self, 'updated_at', None) else None
        }
        
        # 添加名称和描述（如果存在）
        if self.config:
            if 'session_name' in self.config:
                result['name'] = self.config['session_name']
            if 'session_description' in self.config:
                result['description'] = self.config['session_description']
        
        return result


class TrainingProgress(Base, UUIDMixin, TimestampMixin):
    """训练进度模型"""
    __tablename__ = 'training_progress'
    
    session_id = Column(String(36), ForeignKey('training_sessions.session_id'), nullable=False, index=True, comment="会话ID")
    stage = Column(String(20), nullable=False, index=True, comment="训练阶段")
    epoch = Column(Integer, comment="轮次")
    step = Column(Integer, comment="步骤")
    total_steps = Column(Integer, comment="总步骤数")
    loss = Column(Float, comment="损失值")
    accuracy = Column(Float, comment="准确率")
    learning_rate = Column(Float, comment="学习率")
    metrics = Column(JSON, comment="其他指标")
    
    # 添加性能监控相关字段
    gpu_utilization = Column(Float, comment="GPU使用率")
    gpu_memory_used = Column(Float, comment="GPU显存使用量(GB)")
    gpu_memory_total = Column(Float, comment="GPU显存总量(GB)")
    gpu_temperature = Column(Float, comment="GPU温度(°C)")
    gpu_power_draw = Column(Float, comment="GPU功耗(W)")
    
    cpu_utilization = Column(Float, comment="CPU使用率")
    cpu_memory_used = Column(Float, comment="CPU内存使用量(GB)")
    cpu_memory_total = Column(Float, comment="CPU内存总量(GB)")
    cpu_temperature = Column(Float, comment="CPU温度(°C)")
    
    samples_per_second = Column(Float, comment="样本处理速度(样本/秒)")
    tokens_per_second = Column(Float, comment="Token处理速度(tokens/秒)")
    batch_size = Column(Integer, comment="批次大小")
    gradient_norm = Column(Float, comment="梯度范数")
    
    disk_read_speed = Column(Float, comment="磁盘读取速度(MB/s)")
    disk_write_speed = Column(Float, comment="磁盘写入速度(MB/s)")
    disk_utilization = Column(Float, comment="磁盘使用率")
    
    network_download_speed = Column(Float, comment="网络下载速度(MB/s)")
    network_upload_speed = Column(Float, comment="网络上传速度(MB/s)")
    network_latency = Column(Float, comment="网络延迟(ms)")
    
    # 关系
    session = relationship("TrainingSession", back_populates="progress_records")
    
    def __repr__(self):
        return f"<TrainingProgress(id='{self.id}', session_id='{self.session_id}', stage='{self.stage}')>"


class TrainingExecution(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """训练执行记录模型
    
    记录训练任务的完整执行过程，包括配置、状态、资源和结果。
    """
    __tablename__ = 'training_executions'
    
    # 执行标识
    execution_id = Column(String(64), unique=True, nullable=False, index=True, comment="执行唯一标识")
    session_id = Column(String(36), ForeignKey('training_sessions.session_id'), index=True, comment="关联会话ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    
    # 执行基本信息
    name = Column(String(200), comment="执行名称")
    description = Column(Text, comment="执行描述")
    
    # 场景和模式
    scenario_type = Column(String(50), nullable=False, index=True, comment="训练场景类型")
    training_mode = Column(String(50), default='standard', comment="训练模式")
    trainer_type = Column(String(100), comment="训练器类型")
    
    # 配置
    training_config = Column(JSON, comment="训练配置")
    resource_config = Column(JSON, comment="资源配置")
    
    # 状态和进度
    status = Column(String(20), default=TrainingStatus.PENDING.value, index=True, comment="执行状态")
    progress = Column(Float, default=0.0, comment="执行进度(0-100)")
    current_epoch = Column(Integer, default=0, comment="当前轮次")
    total_epochs = Column(Integer, comment="总轮次")
    current_step = Column(Integer, default=0, comment="当前步骤")
    total_steps = Column(Integer, comment="总步骤数")
    
    # 性能指标
    metrics = Column(JSON, comment="当前指标")
    best_metrics = Column(JSON, comment="最佳指标")
    metrics_history = Column(JSON, comment="指标历史")
    
    # 资源使用
    allocated_resources = Column(JSON, comment="已分配资源")
    resource_usage = Column(JSON, comment="资源使用情况")
    allocation_id = Column(String(100), comment="资源分配ID")
    
    # 检查点
    checkpoint_path = Column(String(500), comment="最新检查点路径")
    checkpoint_epoch = Column(Integer, comment="检查点轮次")
    checkpoints = Column(JSON, comment="检查点列表")
    
    # 时间记录
    started_at = Column(DateTime, comment="开始时间")
    paused_at = Column(DateTime, comment="暂停时间")
    resumed_at = Column(DateTime, comment="恢复时间")
    completed_at = Column(DateTime, comment="完成时间")
    estimated_end_time = Column(DateTime, comment="预计结束时间")
    
    # 错误处理
    error_message = Column(Text, comment="错误信息")
    error_details = Column(JSON, comment="错误详情")
    retry_count = Column(Integer, default=0, comment="重试次数")
    
    # 结果
    result = Column(JSON, comment="执行结果")
    output_path = Column(String(500), comment="输出路径")
    
    # 租约管理
    lease_id = Column(String(100), comment="租约ID")
    lease_expires_at = Column(DateTime, comment="租约过期时间")
    
    # 标签和元数据
    tags = Column(JSON, comment="标签")
    extra_data = Column(JSON, comment="额外数据")
    
    def __repr__(self):
        return f"<TrainingExecution(id='{self.id}', execution_id='{self.execution_id}', status='{self.status}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'execution_id': self.execution_id,
            'session_id': self.session_id,
            'tenant_id': str(self.tenant_id) if self.tenant_id else None,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description,
            'scenario_type': self.scenario_type,
            'training_mode': self.training_mode,
            'trainer_type': self.trainer_type,
            'training_config': self.training_config,
            'resource_config': self.resource_config,
            'status': self.status,
            'progress': self.progress,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'metrics': self.metrics,
            'best_metrics': self.best_metrics,
            'allocated_resources': self.allocated_resources,
            'resource_usage': self.resource_usage,
            'checkpoint_path': self.checkpoint_path,
            'checkpoint_epoch': self.checkpoint_epoch,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'paused_at': self.paused_at.isoformat() if self.paused_at else None,
            'resumed_at': self.resumed_at.isoformat() if self.resumed_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'estimated_end_time': self.estimated_end_time.isoformat() if self.estimated_end_time else None,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'result': self.result,
            'output_path': self.output_path,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class TrainingExecutionLog(Base, UUIDMixin, TimestampMixin):
    """训练执行日志模型
    
    记录训练执行过程中的详细事件和状态变更。
    """
    __tablename__ = 'training_execution_logs'
    
    execution_id = Column(String(64), ForeignKey('training_executions.execution_id'), nullable=False, index=True, comment="执行ID")
    
    # 日志类型
    log_type = Column(String(30), nullable=False, index=True, comment="日志类型")
    log_level = Column(String(10), default='info', comment="日志级别")
    
    # 内容
    message = Column(Text, nullable=False, comment="日志消息")
    details = Column(JSON, comment="详细信息")
    
    # 状态变更
    from_status = Column(String(20), comment="变更前状态")
    to_status = Column(String(20), comment="变更后状态")
    
    # 进度信息
    epoch = Column(Integer, comment="轮次")
    step = Column(Integer, comment="步骤")
    progress = Column(Float, comment="进度")
    
    # 指标快照
    metrics = Column(JSON, comment="指标快照")
    
    # 资源信息
    resource_snapshot = Column(JSON, comment="资源快照")
    
    # 来源
    source = Column(String(50), comment="日志来源")
    operator_id = Column(String(36), comment="操作者ID")
    
    def __repr__(self):
        return f"<TrainingExecutionLog(id='{self.id}', execution_id='{self.execution_id}', log_type='{self.log_type}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'execution_id': self.execution_id,
            'log_type': self.log_type,
            'log_level': self.log_level,
            'message': self.message,
            'details': self.details,
            'from_status': self.from_status,
            'to_status': self.to_status,
            'epoch': self.epoch,
            'step': self.step,
            'progress': self.progress,
            'metrics': self.metrics,
            'resource_snapshot': self.resource_snapshot,
            'source': self.source,
            'operator_id': self.operator_id,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


# 训练执行相关索引
Index('idx_training_execution_tenant', TrainingExecution.tenant_id)
Index('idx_training_execution_user', TrainingExecution.user_id)
Index('idx_training_execution_session', TrainingExecution.session_id)
Index('idx_training_execution_status', TrainingExecution.status)
Index('idx_training_execution_scenario', TrainingExecution.scenario_type)
Index('idx_execution_log_execution', TrainingExecutionLog.execution_id)
Index('idx_execution_log_type', TrainingExecutionLog.log_type)


class ThreeStageSession(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """三阶段训练会话模型"""
    __tablename__ = 'three_stage_sessions'
    
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    name = Column(String(200), nullable=False, comment="会话名称")
    description = Column(Text, comment="会话描述")
    model_name = Column(String(100), nullable=False, comment="模型名称")
    status = Column(String(20), default=TrainingStatus.PENDING.value, index=True, comment="状态")
    config = Column(JSON, comment="训练配置")
    result = Column(JSON, comment="训练结果")
    error_message = Column(Text, comment="错误信息")
    progress = Column(Float, default=0.0, comment="总体进度")
    current_stage = Column(String(20), comment="当前阶段")
    pretrain_progress = Column(Float, default=0.0, comment="预训练进度")
    finetune_progress = Column(Float, default=0.0, comment="微调进度")
    preference_progress = Column(Float, default=0.0, comment="偏好优化进度")
    started_at = Column(DateTime, comment="开始时间")
    completed_at = Column(DateTime, comment="完成时间")
    
    def __repr__(self):
        return f"<ThreeStageSession(id='{self.id}', name='{self.name}', status='{self.status}')>"


class ThreeStageProgress(Base, UUIDMixin, TimestampMixin):
    """三阶段训练进度模型"""
    __tablename__ = 'three_stage_progress'
    
    session_id = Column(GUID(), ForeignKey('three_stage_sessions.id'), nullable=False, index=True, comment="会话ID")
    stage = Column(String(20), nullable=False, index=True, comment="训练阶段")
    epoch = Column(Integer, comment="轮次")
    step = Column(Integer, comment="步骤")
    total_steps = Column(Integer, comment="总步骤数")
    loss = Column(Float, comment="损失值")
    accuracy = Column(Float, comment="准确率")
    learning_rate = Column(Float, comment="学习率")
    val_loss = Column(Float, comment="验证损失")
    val_accuracy = Column(Float, comment="验证准确率")
    metrics = Column(JSON, comment="其他指标")
    
    def __repr__(self):
        return f"<ThreeStageProgress(id='{self.id}', session_id='{self.session_id}', stage='{self.stage}')>"


# ==============================================================================
# 超参数优化相关模型
# ==============================================================================

class HyperparameterOptimization(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """超参数优化任务模型"""
    __tablename__ = 'hyperparameter_optimizations'
    
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    name = Column(String(200), comment="优化任务名称")
    description = Column(Text, comment="任务描述")
    scenario_type = Column(String(50), nullable=False, index=True, comment="训练场景类型")
    optimization_method = Column(String(50), default='random', comment="优化方法 (random, grid, bayesian)")
    status = Column(String(20), default='pending', index=True, comment="状态 (pending, running, completed, failed, cancelled)")
    
    # 搜索空间配置
    search_space = Column(JSON, nullable=False, comment="超参数搜索空间定义")
    training_config = Column(JSON, comment="训练配置")
    
    # 优化参数
    max_trials = Column(Integer, default=10, comment="最大试验次数")
    current_trial = Column(Integer, default=0, comment="当前试验次数")
    timeout_seconds = Column(Integer, comment="超时时间(秒)")
    early_stopping_patience = Column(Integer, comment="早停耐心值")
    
    # 最佳结果
    best_params = Column(JSON, comment="最佳超参数组合")
    best_score = Column(Float, comment="最佳得分")
    best_trial_id = Column(String(36), comment="最佳试验ID")
    
    # 统计信息
    total_trials = Column(Integer, default=0, comment="总试验次数")
    successful_trials = Column(Integer, default=0, comment="成功试验次数")
    failed_trials = Column(Integer, default=0, comment="失败试验次数")
    avg_trial_duration = Column(Float, comment="平均试验时长(秒)")
    
    # 时间
    started_at = Column(DateTime, comment="开始时间")
    completed_at = Column(DateTime, comment="完成时间")
    
    # 关联
    model_id = Column(String(100), comment="关联模型ID")
    dataset_id = Column(String(100), comment="关联数据集ID")
    
    # 元数据
    tags = Column(JSON, comment="标签")
    extra_data = Column(JSON, comment="额外数据")
    
    # 关系
    trials = relationship("HyperparameterTrial", back_populates="optimization", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<HyperparameterOptimization(id='{self.id}', user_id='{self.user_id}', status='{self.status}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'name': self.name,
            'description': self.description,
            'scenario_type': self.scenario_type,
            'optimization_method': self.optimization_method,
            'status': self.status,
            'search_space': self.search_space,
            'training_config': self.training_config,
            'max_trials': self.max_trials,
            'current_trial': self.current_trial,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_trial_id': self.best_trial_id,
            'total_trials': self.total_trials,
            'successful_trials': self.successful_trials,
            'failed_trials': self.failed_trials,
            'avg_trial_duration': self.avg_trial_duration,
            'model_id': self.model_id,
            'dataset_id': self.dataset_id,
            'tags': self.tags,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class HyperparameterTrial(Base, UUIDMixin, TimestampMixin):
    """超参数试验记录模型"""
    __tablename__ = 'hyperparameter_trials'
    
    optimization_id = Column(GUID(), ForeignKey('hyperparameter_optimizations.id'), 
                           nullable=False, index=True, comment="优化任务ID")
    trial_number = Column(Integer, nullable=False, comment="试验编号")
    status = Column(String(20), default='pending', index=True, comment="状态")
    
    # 超参数
    params = Column(JSON, nullable=False, comment="试验超参数")
    
    # 评估结果
    score = Column(Float, comment="评估得分")
    metrics = Column(JSON, comment="评估指标")
    
    # 训练详情
    training_session_id = Column(String(36), comment="关联的训练会话ID")
    training_loss = Column(Float, comment="训练损失")
    validation_loss = Column(Float, comment="验证损失")
    training_accuracy = Column(Float, comment="训练准确率")
    validation_accuracy = Column(Float, comment="验证准确率")
    
    # 时间
    started_at = Column(DateTime, comment="开始时间")
    completed_at = Column(DateTime, comment="完成时间")
    duration_seconds = Column(Float, comment="持续时间(秒)")
    
    # 错误信息
    error_message = Column(Text, comment="错误信息")
    error_details = Column(JSON, comment="错误详情")
    
    # 额外数据
    extra_data = Column(JSON, comment="额外数据")
    
    # 关系
    optimization = relationship("HyperparameterOptimization", back_populates="trials")
    
    def __repr__(self):
        return f"<HyperparameterTrial(id='{self.id}', trial_number={self.trial_number}, score={self.score})>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'optimization_id': str(self.optimization_id) if self.optimization_id else None,
            'trial_number': self.trial_number,
            'status': self.status,
            'params': self.params,
            'score': self.score,
            'metrics': self.metrics,
            'training_session_id': self.training_session_id,
            'training_loss': self.training_loss,
            'validation_loss': self.validation_loss,
            'training_accuracy': self.training_accuracy,
            'validation_accuracy': self.validation_accuracy,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.duration_seconds,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class HyperparameterSearchSpace(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """超参数搜索空间模板模型"""
    __tablename__ = 'hyperparameter_search_spaces'
    
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    name = Column(String(200), nullable=False, comment="模板名称")
    description = Column(Text, comment="模板描述")
    scenario_type = Column(String(50), index=True, comment="适用场景类型")
    is_default = Column(Boolean, default=False, comment="是否为默认模板")
    is_public = Column(Boolean, default=False, comment="是否公开")
    
    # 搜索空间定义
    parameters = Column(JSON, nullable=False, comment="参数定义列表")
    
    # 推荐配置
    recommended_method = Column(String(50), comment="推荐优化方法")
    recommended_trials = Column(Integer, comment="推荐试验次数")
    
    # 使用统计
    usage_count = Column(Integer, default=0, comment="使用次数")
    
    # 元数据
    tags = Column(JSON, comment="标签")
    extra_data = Column(JSON, comment="额外数据")
    
    def __repr__(self):
        return f"<HyperparameterSearchSpace(id='{self.id}', name='{self.name}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'name': self.name,
            'description': self.description,
            'scenario_type': self.scenario_type,
            'is_default': self.is_default,
            'is_public': self.is_public,
            'parameters': self.parameters,
            'recommended_method': self.recommended_method,
            'recommended_trials': self.recommended_trials,
            'usage_count': self.usage_count,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


# ==============================================================================
# 智能决策相关模型
# ==============================================================================

class IntelligentDecision(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """智能决策记录模型"""
    __tablename__ = 'intelligent_decisions'
    
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    decision_id = Column(String(100), unique=True, nullable=False, index=True, comment="决策ID")
    scenario = Column(String(50), nullable=False, index=True, comment="决策场景")
    algorithm = Column(String(50), comment="使用的算法")
    status = Column(String(20), default='completed', index=True, comment="状态")
    
    # 输入数据
    inputs = Column(JSON, comment="输入数据")
    constraints = Column(JSON, comment="约束条件")
    history_context = Column(JSON, comment="历史上下文")
    
    # 决策结果
    recommended_action = Column(JSON, nullable=False, comment="推荐动作")
    confidence = Column(Float, nullable=False, comment="置信度")
    reasoning = Column(Text, comment="推理过程")
    alternatives = Column(JSON, comment="备选方案")
    execution_plan = Column(JSON, comment="执行计划")
    
    # 执行结果
    executed = Column(Boolean, default=False, comment="是否已执行")
    execution_result = Column(JSON, comment="执行结果")
    feedback_score = Column(Float, comment="反馈评分")
    feedback_comment = Column(Text, comment="反馈说明")
    
    # 元数据
    tags = Column(JSON, comment="标签")
    extra_data = Column(JSON, comment="额外数据")
    
    def __repr__(self):
        return f"<IntelligentDecision(id='{self.id}', decision_id='{self.decision_id}', scenario='{self.scenario}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'decision_id': self.decision_id,
            'scenario': self.scenario,
            'algorithm': self.algorithm,
            'status': self.status,
            'inputs': self.inputs,
            'constraints': self.constraints,
            'recommended_action': self.recommended_action,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'alternatives': self.alternatives,
            'execution_plan': self.execution_plan,
            'executed': self.executed,
            'execution_result': self.execution_result,
            'feedback_score': self.feedback_score,
            'feedback_comment': self.feedback_comment,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class AdaptiveOptimization(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """自适应优化记录模型"""
    __tablename__ = 'adaptive_optimizations'
    
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    optimization_id = Column(String(100), unique=True, nullable=False, index=True, comment="优化ID")
    parameter_name = Column(String(100), nullable=False, index=True, comment="参数名称")
    
    # 优化配置
    adjustment_strategy = Column(String(50), nullable=False, comment="调整策略")
    adjustment_range = Column(JSON, comment="调整范围")
    monitoring_metrics = Column(JSON, comment="监控指标")
    
    # 优化结果
    original_value = Column(JSON, nullable=False, comment="原始值")
    optimized_value = Column(JSON, nullable=False, comment="优化后的值")
    improvement_metric = Column(String(100), comment="改进指标")
    improvement_value = Column(Float, comment="改进值")
    adjustment_reason = Column(Text, comment="调整原因")
    
    # 应用结果
    applied = Column(Boolean, default=False, comment="是否已应用")
    applied_at = Column(DateTime, comment="应用时间")
    rollback = Column(Boolean, default=False, comment="是否已回滚")
    rollback_reason = Column(Text, comment="回滚原因")
    
    # 元数据
    tags = Column(JSON, comment="标签")
    extra_data = Column(JSON, comment="额外数据")
    
    def __repr__(self):
        return f"<AdaptiveOptimization(id='{self.id}', parameter_name='{self.parameter_name}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'optimization_id': self.optimization_id,
            'parameter_name': self.parameter_name,
            'adjustment_strategy': self.adjustment_strategy,
            'adjustment_range': self.adjustment_range,
            'monitoring_metrics': self.monitoring_metrics,
            'original_value': self.original_value,
            'optimized_value': self.optimized_value,
            'improvement_metric': self.improvement_metric,
            'improvement_value': self.improvement_value,
            'adjustment_reason': self.adjustment_reason,
            'applied': self.applied,
            'applied_at': self.applied_at.isoformat() if self.applied_at else None,
            'rollback': self.rollback,
            'rollback_reason': self.rollback_reason,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class KnowledgeBase(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """知识库模型"""
    __tablename__ = 'knowledge_base'
    
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    knowledge_type = Column(String(50), nullable=False, index=True, comment="知识类型")
    category = Column(String(100), index=True, comment="类别")
    title = Column(String(255), nullable=False, comment="标题")
    content = Column(JSON, nullable=False, comment="知识内容")
    
    # 关联
    related_entities = Column(JSON, comment="相关实体")
    relationships = Column(JSON, comment="关系")
    
    # 来源和可信度
    source = Column(String(255), comment="来源")
    confidence = Column(Float, default=1.0, comment="可信度")
    verified = Column(Boolean, default=False, comment="是否已验证")
    verified_by = Column(String(36), comment="验证者ID")
    verified_at = Column(DateTime, comment="验证时间")
    
    # 使用统计
    usage_count = Column(Integer, default=0, comment="使用次数")
    last_used_at = Column(DateTime, comment="最后使用时间")
    
    # 状态
    is_active = Column(Boolean, default=True, comment="是否激活")
    is_public = Column(Boolean, default=False, comment="是否公开")
    
    # 元数据
    tags = Column(JSON, comment="标签")
    extra_data = Column(JSON, comment="额外数据")
    
    def __repr__(self):
        return f"<KnowledgeBase(id='{self.id}', title='{self.title}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'knowledge_type': self.knowledge_type,
            'category': self.category,
            'title': self.title,
            'content': self.content,
            'related_entities': self.related_entities,
            'relationships': self.relationships,
            'source': self.source,
            'confidence': self.confidence,
            'verified': self.verified,
            'usage_count': self.usage_count,
            'is_active': self.is_active,
            'is_public': self.is_public,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ExperienceRecord(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """经验记录模型"""
    __tablename__ = 'experience_records'
    
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    experience_type = Column(String(50), nullable=False, index=True, comment="经验类型")
    scenario = Column(String(100), index=True, comment="场景")
    
    # 经验数据
    context = Column(JSON, comment="上下文")
    action = Column(JSON, nullable=False, comment="采取的行动")
    result = Column(JSON, nullable=False, comment="结果")
    reward = Column(Float, comment="奖励值")
    
    # 关联决策
    decision_id = Column(String(100), index=True, comment="关联的决策ID")
    
    # 评估
    effectiveness = Column(Float, comment="有效性评分")
    lessons_learned = Column(Text, comment="经验教训")
    
    # 状态
    is_positive = Column(Boolean, default=True, comment="是否是正面经验")
    is_verified = Column(Boolean, default=False, comment="是否已验证")
    
    # 元数据
    tags = Column(JSON, comment="标签")
    extra_data = Column(JSON, comment="额外数据")
    
    def __repr__(self):
        return f"<ExperienceRecord(id='{self.id}', experience_type='{self.experience_type}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'experience_type': self.experience_type,
            'scenario': self.scenario,
            'context': self.context,
            'action': self.action,
            'result': self.result,
            'reward': self.reward,
            'decision_id': self.decision_id,
            'effectiveness': self.effectiveness,
            'lessons_learned': self.lessons_learned,
            'is_positive': self.is_positive,
            'is_verified': self.is_verified,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


# ==============================================================================
# 模型部署相关模型
# ==============================================================================

class ModelDeployment(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型部署记录模型"""
    __tablename__ = 'model_deployments'
    
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    deployment_id = Column(String(100), unique=True, nullable=False, index=True, comment="部署ID")
    model_id = Column(String(100), nullable=False, index=True, comment="模型ID")
    
    # 部署配置
    mode = Column(String(50), nullable=False, default='online', index=True, comment="部署模式 (online, batch, edge, hybrid)")
    release_strategy = Column(String(50), default='rolling', comment="发布策略 (rolling, canary, blue_green, ab_testing)")
    status = Column(String(20), default='pending', index=True, comment="状态 (pending, deploying, running, stopped, failed, rolled_back)")
    
    # 资源配置
    replicas = Column(Integer, default=1, comment="副本数")
    resources = Column(JSON, comment="资源配置")
    environment = Column(JSON, comment="环境变量")
    
    # 端点信息
    endpoint_url = Column(String(500), comment="访问端点")
    health_check_url = Column(String(500), comment="健康检查端点")
    
    # 配置参数
    autoscaling = Column(Boolean, default=False, comment="是否自动扩缩容")
    health_check = Column(Boolean, default=True, comment="是否启用健康检查")
    monitoring = Column(Boolean, default=True, comment="是否启用监控")
    canary_percent = Column(Integer, default=10, comment="金丝雀流量百分比")
    rolling_step = Column(Integer, default=1, comment="滚动更新步长")
    ab_percent = Column(Integer, default=50, comment="AB测试百分比")
    
    # 时间
    started_at = Column(DateTime, comment="部署开始时间")
    completed_at = Column(DateTime, comment="部署完成时间")
    stopped_at = Column(DateTime, comment="停止时间")
    
    # 统计
    deployment_time_seconds = Column(Float, comment="部署耗时(秒)")
    uptime_seconds = Column(Float, comment="运行时长(秒)")
    request_count = Column(Integer, default=0, comment="请求总数")
    error_count = Column(Integer, default=0, comment="错误次数")
    
    # 版本信息
    version = Column(String(50), comment="部署版本")
    previous_version = Column(String(50), comment="上一版本")
    
    # 错误信息
    error_message = Column(Text, comment="错误信息")
    error_details = Column(JSON, comment="错误详情")
    
    # 元数据
    tags = Column(JSON, comment="标签")
    extra_data = Column(JSON, comment="额外数据")
    
    # 关系
    logs = relationship("ModelDeploymentLog", back_populates="deployment", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ModelDeployment(id='{self.id}', deployment_id='{self.deployment_id}', status='{self.status}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'deployment_id': self.deployment_id,
            'model_id': self.model_id,
            'mode': self.mode,
            'release_strategy': self.release_strategy,
            'status': self.status,
            'replicas': self.replicas,
            'resources': self.resources,
            'environment': self.environment,
            'endpoint_url': self.endpoint_url,
            'health_check_url': self.health_check_url,
            'autoscaling': self.autoscaling,
            'health_check': self.health_check,
            'monitoring': self.monitoring,
            'canary_percent': self.canary_percent,
            'rolling_step': self.rolling_step,
            'ab_percent': self.ab_percent,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'stopped_at': self.stopped_at.isoformat() if self.stopped_at else None,
            'deployment_time_seconds': self.deployment_time_seconds,
            'uptime_seconds': self.uptime_seconds,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'version': self.version,
            'error_message': self.error_message,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ModelDeploymentLog(Base, UUIDMixin, TimestampMixin):
    """模型部署日志模型"""
    __tablename__ = 'model_deployment_logs'
    
    deployment_id = Column(GUID(), ForeignKey('model_deployments.id'), 
                          nullable=False, index=True, comment="部署记录ID")
    level = Column(String(20), default='info', index=True, comment="日志级别")
    action = Column(String(50), nullable=False, index=True, comment="动作类型")
    message = Column(Text, nullable=False, comment="日志消息")
    details = Column(JSON, comment="详细信息")
    source = Column(String(100), comment="日志来源")
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, comment="时间戳")
    
    # 关系
    deployment = relationship("ModelDeployment", back_populates="logs")
    
    def __repr__(self):
        return f"<ModelDeploymentLog(id='{self.id}', action='{self.action}', level='{self.level}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'deployment_id': str(self.deployment_id) if self.deployment_id else None,
            'level': self.level,
            'action': self.action,
            'message': self.message,
            'details': self.details,
            'source': self.source,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class ModelService(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型服务化记录模型"""
    __tablename__ = 'model_services'
    
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    service_id = Column(String(100), unique=True, nullable=False, index=True, comment="服务ID")
    model_id = Column(String(100), nullable=False, index=True, comment="模型ID")
    deployment_id = Column(String(100), index=True, comment="关联的部署ID")
    
    # 服务配置
    api_type = Column(String(50), default='rest', comment="API类型 (rest, graphql, grpc, websocket)")
    service_mesh = Column(Boolean, default=False, comment="是否使用服务网格")
    load_balancing = Column(Boolean, default=True, comment="是否启用负载均衡")
    circuit_breaker = Column(Boolean, default=True, comment="是否启用熔断")
    rate_limiting = Column(Boolean, default=True, comment="是否启用限流")
    timeout = Column(Integer, default=30, comment="超时时间(秒)")
    max_concurrent_requests = Column(Integer, default=100, comment="最大并发请求数")
    
    # 服务端点
    endpoints = Column(JSON, comment="API端点列表")
    status = Column(String(20), default='pending', index=True, comment="状态")
    
    # 时间
    started_at = Column(DateTime, comment="启动时间")
    stopped_at = Column(DateTime, comment="停止时间")
    
    # 统计
    request_count = Column(Integer, default=0, comment="请求总数")
    success_count = Column(Integer, default=0, comment="成功请求数")
    error_count = Column(Integer, default=0, comment="错误请求数")
    avg_latency_ms = Column(Float, comment="平均延迟(毫秒)")
    
    # 错误信息
    error_message = Column(Text, comment="错误信息")
    
    # 元数据
    tags = Column(JSON, comment="标签")
    extra_data = Column(JSON, comment="额外数据")
    
    def __repr__(self):
        return f"<ModelService(id='{self.id}', service_id='{self.service_id}', api_type='{self.api_type}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'service_id': self.service_id,
            'model_id': self.model_id,
            'deployment_id': self.deployment_id,
            'api_type': self.api_type,
            'service_mesh': self.service_mesh,
            'load_balancing': self.load_balancing,
            'circuit_breaker': self.circuit_breaker,
            'rate_limiting': self.rate_limiting,
            'timeout': self.timeout,
            'max_concurrent_requests': self.max_concurrent_requests,
            'endpoints': self.endpoints,
            'status': self.status,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'stopped_at': self.stopped_at.isoformat() if self.stopped_at else None,
            'request_count': self.request_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'avg_latency_ms': self.avg_latency_ms,
            'error_message': self.error_message,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class DeploymentAuditEvent(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """部署审计事件模型
    
    记录所有部署相关的操作审计事件，支持租户级别隔离。
    """
    __tablename__ = 'deployment_audit_events'
    
    # 关联信息
    deployment_id = Column(String(100), nullable=False, index=True, comment="部署ID")
    model_id = Column(String(100), index=True, comment="模型ID")
    user_id = Column(String(36), nullable=False, index=True, comment="操作用户ID")
    
    # 事件信息
    event_type = Column(String(50), nullable=False, index=True, 
                       comment="事件类型 (deploy, undeploy, scale, rollback, update, health_check, error, warning)")
    action = Column(String(100), nullable=False, index=True, comment="具体动作")
    status = Column(String(20), default='success', index=True, comment="操作状态 (success, failure, pending)")
    
    # 详细信息
    message = Column(Text, comment="事件消息")
    description = Column(Text, comment="详细描述")
    
    # 变更信息
    old_value = Column(JSON, comment="变更前的值")
    new_value = Column(JSON, comment="变更后的值")
    
    # 上下文信息
    source = Column(String(100), default='api', comment="事件来源 (api, scheduler, system, container)")
    trigger = Column(String(50), comment="触发方式 (manual, automatic, scheduled)")
    ip_address = Column(String(50), comment="操作者IP地址")
    user_agent = Column(String(500), comment="用户代理")
    
    # 性能指标
    duration_ms = Column(Float, comment="操作耗时(毫秒)")
    
    # 版本信息
    from_version = Column(String(50), comment="原版本")
    to_version = Column(String(50), comment="目标版本")
    
    # 资源变更
    resource_changes = Column(JSON, comment="资源配置变更")
    
    # 错误信息
    error_code = Column(String(50), comment="错误代码")
    error_message = Column(Text, comment="错误消息")
    stack_trace = Column(Text, comment="错误堆栈")
    
    # 关联的其他事件
    parent_event_id = Column(String(36), index=True, comment="父事件ID")
    correlation_id = Column(String(100), index=True, comment="关联ID（用于追踪一系列相关事件）")
    
    # 事件时间
    event_time = Column(DateTime, default=datetime.utcnow, index=True, comment="事件发生时间")
    
    # 元数据
    tags = Column(JSON, comment="标签")
    metadata_ = Column(JSON, comment="额外元数据")
    
    def __repr__(self):
        return f"<DeploymentAuditEvent(id='{self.id}', event_type='{self.event_type}', action='{self.action}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'tenant_id': self.tenant_id,
            'deployment_id': self.deployment_id,
            'model_id': self.model_id,
            'user_id': self.user_id,
            'event_type': self.event_type,
            'action': self.action,
            'status': self.status,
            'message': self.message,
            'description': self.description,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'source': self.source,
            'trigger': self.trigger,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'duration_ms': self.duration_ms,
            'from_version': self.from_version,
            'to_version': self.to_version,
            'resource_changes': self.resource_changes,
            'error_code': self.error_code,
            'error_message': self.error_message,
            'parent_event_id': self.parent_event_id,
            'correlation_id': self.correlation_id,
            'event_time': self.event_time.isoformat() if self.event_time else None,
            'tags': self.tags,
            'metadata': self.metadata_,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


# ==============================================================================
# 模型评估相关模型
# ==============================================================================

class ModelEvaluation(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型评估记录模型
    
    存储模型评估的结果和配置信息。
    """
    __tablename__ = 'model_evaluations'
    
    # 关联信息
    evaluation_id = Column(String(100), unique=True, nullable=False, index=True, comment="评估ID")
    model_id = Column(String(100), nullable=False, index=True, comment="模型ID")
    dataset_id = Column(String(100), nullable=False, index=True, comment="数据集ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    
    # 评估配置
    evaluation_type = Column(String(50), default='automated', index=True, 
                            comment="评估类型 (automated, manual, scheduled)")
    validation_strategy = Column(String(50), default='holdout', 
                                comment="验证策略 (holdout, cross_validation, bootstrap)")
    cross_validation_folds = Column(Integer, default=5, comment="交叉验证折数")
    test_size = Column(Float, default=0.2, comment="测试集比例")
    
    # 状态
    status = Column(String(20), default='pending', index=True, 
                   comment="状态 (pending, running, completed, failed)")
    
    # 汇总指标
    accuracy = Column(Float, comment="准确率")
    precision = Column(Float, comment="精确率")
    recall = Column(Float, comment="召回率")
    f1_score = Column(Float, comment="F1分数")
    auc = Column(Float, comment="AUC值")
    loss = Column(Float, comment="损失值")
    
    # 时间信息
    started_at = Column(DateTime, comment="评估开始时间")
    completed_at = Column(DateTime, comment="评估完成时间")
    duration_seconds = Column(Float, comment="评估耗时(秒)")
    
    # 配置和元数据
    evaluation_config = Column(JSON, comment="完整评估配置")
    metrics_summary = Column(JSON, comment="指标汇总")
    data_statistics = Column(JSON, comment="数据统计信息")
    
    # 错误信息
    error_message = Column(Text, comment="错误消息")
    error_details = Column(JSON, comment="错误详情")
    
    # 元数据
    tags = Column(JSON, comment="标签")
    metadata_ = Column(JSON, comment="额外元数据")
    
    # 关系
    metrics = relationship("ModelEvaluationMetric", back_populates="evaluation", 
                          cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ModelEvaluation(id='{self.id}', evaluation_id='{self.evaluation_id}', status='{self.status}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'tenant_id': self.tenant_id,
            'evaluation_id': self.evaluation_id,
            'model_id': self.model_id,
            'dataset_id': self.dataset_id,
            'user_id': self.user_id,
            'evaluation_type': self.evaluation_type,
            'validation_strategy': self.validation_strategy,
            'cross_validation_folds': self.cross_validation_folds,
            'test_size': self.test_size,
            'status': self.status,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc': self.auc,
            'loss': self.loss,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.duration_seconds,
            'evaluation_config': self.evaluation_config,
            'metrics_summary': self.metrics_summary,
            'data_statistics': self.data_statistics,
            'error_message': self.error_message,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ModelEvaluationMetric(Base, UUIDMixin, TimestampMixin):
    """模型评估指标模型
    
    存储单个评估指标的详细信息。
    """
    __tablename__ = 'model_evaluation_metrics'
    
    # 关联
    evaluation_id = Column(GUID(), ForeignKey('model_evaluations.id'),
                          nullable=False, index=True, comment="评估记录ID")
    
    # 指标信息
    metric_name = Column(String(100), nullable=False, index=True, comment="指标名称")
    metric_type = Column(String(50), nullable=False, index=True, 
                        comment="指标类型 (accuracy, precision, recall, f1_score, auc, custom)")
    metric_value = Column(Float, nullable=False, comment="指标值")
    
    # 置信区间
    confidence_lower = Column(Float, comment="置信区间下限")
    confidence_upper = Column(Float, comment="置信区间上限")
    confidence_level = Column(Float, default=0.95, comment="置信水平")
    
    # 额外信息
    description = Column(Text, comment="指标描述")
    unit = Column(String(50), comment="单位")
    is_primary = Column(Boolean, default=False, comment="是否为主要指标")
    
    # 详细数据
    per_class_values = Column(JSON, comment="每个类别的指标值")
    additional_info = Column(JSON, comment="附加信息")
    
    # 关系
    evaluation = relationship("ModelEvaluation", back_populates="metrics")
    
    def __repr__(self):
        return f"<ModelEvaluationMetric(id='{self.id}', metric_name='{self.metric_name}', value={self.metric_value})>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'evaluation_id': str(self.evaluation_id) if self.evaluation_id else None,
            'metric_name': self.metric_name,
            'metric_type': self.metric_type,
            'metric_value': self.metric_value,
            'confidence_lower': self.confidence_lower,
            'confidence_upper': self.confidence_upper,
            'confidence_level': self.confidence_level,
            'description': self.description,
            'unit': self.unit,
            'is_primary': self.is_primary,
            'per_class_values': self.per_class_values,
            'additional_info': self.additional_info,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class ModelComparisonRecord(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型对比记录模型
    
    存储模型对比的结果和配置。
    """
    __tablename__ = 'model_comparison_records'
    
    # 关联信息
    comparison_id = Column(String(100), unique=True, nullable=False, index=True, comment="对比ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    dataset_id = Column(String(100), nullable=False, index=True, comment="数据集ID")
    
    # 参与对比的模型
    model_ids = Column(JSON, nullable=False, comment="参与对比的模型ID列表")
    winner_model_id = Column(String(100), index=True, comment="获胜模型ID")
    
    # 状态
    status = Column(String(20), default='pending', index=True,
                   comment="状态 (pending, running, completed, failed)")
    
    # 对比配置
    comparison_config = Column(JSON, comment="对比配置")
    comparison_metrics = Column(JSON, comment="对比使用的指标")
    decision_criteria = Column(String(100), default='multi_objective', comment="决策标准")
    
    # 结果
    ranking = Column(JSON, comment="模型排名")
    recommendations = Column(JSON, comment="推荐建议列表")
    risk_assessment = Column(JSON, comment="风险评估")
    detailed_results = Column(JSON, comment="详细对比结果")
    
    # 时间
    started_at = Column(DateTime, comment="对比开始时间")
    completed_at = Column(DateTime, comment="对比完成时间")
    duration_seconds = Column(Float, comment="对比耗时(秒)")
    
    # 错误信息
    error_message = Column(Text, comment="错误消息")
    
    # 元数据
    tags = Column(JSON, comment="标签")
    metadata_ = Column(JSON, comment="额外元数据")
    
    def __repr__(self):
        return f"<ModelComparisonRecord(id='{self.id}', comparison_id='{self.comparison_id}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'tenant_id': self.tenant_id,
            'comparison_id': self.comparison_id,
            'user_id': self.user_id,
            'dataset_id': self.dataset_id,
            'model_ids': self.model_ids,
            'winner_model_id': self.winner_model_id,
            'status': self.status,
            'comparison_config': self.comparison_config,
            'comparison_metrics': self.comparison_metrics,
            'decision_criteria': self.decision_criteria,
            'ranking': self.ranking,
            'recommendations': self.recommendations,
            'risk_assessment': self.risk_assessment,
            'detailed_results': self.detailed_results,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.duration_seconds,
            'error_message': self.error_message,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


# ==============================================================================
# 模型优化相关模型
# ==============================================================================

class ModelOptimization(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型优化记录模型
    
    存储模型压缩和推理优化的结果。
    """
    __tablename__ = 'model_optimizations'
    
    # 关联信息
    optimization_id = Column(String(100), unique=True, nullable=False, index=True, comment="优化ID")
    original_model_id = Column(String(100), nullable=False, index=True, comment="原始模型ID")
    optimized_model_id = Column(String(100), index=True, comment="优化后模型ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    
    # 优化类型
    optimization_type = Column(String(50), nullable=False, index=True, 
                              comment="优化类型 (compression, inference, auto)")
    technique = Column(String(50), index=True, 
                      comment="优化技术 (quantization, pruning, knowledge_distillation, low_rank_decomposition)")
    strategy = Column(String(50), comment="压缩策略 (structured, unstructured, mixed)")
    
    # 状态
    status = Column(String(20), default='pending', index=True,
                   comment="状态 (pending, running, completed, failed)")
    
    # 压缩配置
    compression_ratio = Column(Float, comment="压缩率")
    quantization_bits = Column(Integer, comment="量化位数")
    preserve_accuracy = Column(Boolean, default=True, comment="是否保持精度")
    
    # 推理优化配置
    hardware_target = Column(String(50), comment="目标硬件 (cpu, gpu, tpu, edge)")
    graph_optimization = Column(Boolean, default=False, comment="图优化")
    operator_fusion = Column(Boolean, default=False, comment="算子融合")
    constant_folding = Column(Boolean, default=False, comment="常量折叠")
    dead_code_elimination = Column(Boolean, default=False, comment="死代码消除")
    memory_optimization = Column(Boolean, default=False, comment="内存优化")
    
    # 结果指标 - 压缩
    accuracy_preserved = Column(Float, comment="精度保持率")
    model_size_reduction = Column(Float, comment="模型大小减少率")
    inference_speedup = Column(Float, comment="推理加速比")
    original_size_mb = Column(Float, comment="原始模型大小(MB)")
    optimized_size_mb = Column(Float, comment="优化后模型大小(MB)")
    
    # 结果指标 - 推理优化
    latency_reduction = Column(Float, comment="延迟降低率")
    memory_usage_reduction = Column(Float, comment="内存使用降低率")
    throughput_improvement = Column(Float, comment="吞吐量提升倍数")
    original_latency_ms = Column(Float, comment="原始延迟(ms)")
    optimized_latency_ms = Column(Float, comment="优化后延迟(ms)")
    
    # 时间信息
    started_at = Column(DateTime, comment="优化开始时间")
    completed_at = Column(DateTime, comment="优化完成时间")
    optimization_time_seconds = Column(Float, comment="优化耗时(秒)")
    
    # 配置和详情
    optimization_config = Column(JSON, comment="完整优化配置")
    target_constraints = Column(JSON, comment="目标约束条件")
    metrics = Column(JSON, comment="详细指标")
    
    # 错误信息
    error_message = Column(Text, comment="错误消息")
    error_details = Column(JSON, comment="错误详情")
    
    # 元数据
    tags = Column(JSON, comment="标签")
    metadata_ = Column(JSON, comment="额外元数据")
    
    def __repr__(self):
        return f"<ModelOptimization(id='{self.id}', optimization_id='{self.optimization_id}', technique='{self.technique}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'tenant_id': self.tenant_id,
            'optimization_id': self.optimization_id,
            'original_model_id': self.original_model_id,
            'optimized_model_id': self.optimized_model_id,
            'user_id': self.user_id,
            'optimization_type': self.optimization_type,
            'technique': self.technique,
            'strategy': self.strategy,
            'status': self.status,
            'compression_ratio': self.compression_ratio,
            'quantization_bits': self.quantization_bits,
            'preserve_accuracy': self.preserve_accuracy,
            'hardware_target': self.hardware_target,
            'graph_optimization': self.graph_optimization,
            'operator_fusion': self.operator_fusion,
            'constant_folding': self.constant_folding,
            'dead_code_elimination': self.dead_code_elimination,
            'memory_optimization': self.memory_optimization,
            'accuracy_preserved': self.accuracy_preserved,
            'model_size_reduction': self.model_size_reduction,
            'inference_speedup': self.inference_speedup,
            'original_size_mb': self.original_size_mb,
            'optimized_size_mb': self.optimized_size_mb,
            'latency_reduction': self.latency_reduction,
            'memory_usage_reduction': self.memory_usage_reduction,
            'throughput_improvement': self.throughput_improvement,
            'original_latency_ms': self.original_latency_ms,
            'optimized_latency_ms': self.optimized_latency_ms,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'optimization_time_seconds': self.optimization_time_seconds,
            'optimization_config': self.optimization_config,
            'target_constraints': self.target_constraints,
            'metrics': self.metrics,
            'error_message': self.error_message,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


# ==============================================================================
# 模型选择相关模型
# ==============================================================================

class ModelRecommendationRecord(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型推荐记录模型
    
    存储模型推荐的历史记录，用于分析和改进推荐系统。
    """
    __tablename__ = 'model_recommendation_records'
    
    # 关联信息
    recommendation_id = Column(String(100), unique=True, nullable=False, index=True, comment="推荐ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    
    # 请求信息
    task_type = Column(String(100), nullable=False, index=True, comment="任务类型")
    requirements = Column(JSON, comment="硬件和环境要求")
    performance_requirements = Column(JSON, comment="性能要求")
    
    # 推荐结果
    recommended_models = Column(JSON, nullable=False, comment="推荐的模型列表")
    top_recommendation = Column(String(200), comment="首推模型名称")
    top_confidence = Column(Float, comment="首推置信度")
    num_recommendations = Column(Integer, default=0, comment="推荐数量")
    
    # 用户反馈
    selected_model = Column(String(200), comment="用户选择的模型")
    feedback_score = Column(Float, comment="用户反馈评分 (1-5)")
    feedback_comment = Column(Text, comment="用户反馈评论")
    is_helpful = Column(Boolean, comment="推荐是否有帮助")
    
    # 状态和时间
    status = Column(String(20), default='completed', index=True, comment="状态")
    response_time_ms = Column(Float, comment="响应时间(毫秒)")
    
    # 元数据
    source = Column(String(50), default='api', comment="请求来源")
    metadata_ = Column(JSON, comment="额外元数据")
    
    def __repr__(self):
        return f"<ModelRecommendationRecord(id='{self.id}', recommendation_id='{self.recommendation_id}', task_type='{self.task_type}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'tenant_id': self.tenant_id,
            'recommendation_id': self.recommendation_id,
            'user_id': self.user_id,
            'task_type': self.task_type,
            'requirements': self.requirements,
            'performance_requirements': self.performance_requirements,
            'recommended_models': self.recommended_models,
            'top_recommendation': self.top_recommendation,
            'top_confidence': self.top_confidence,
            'num_recommendations': self.num_recommendations,
            'selected_model': self.selected_model,
            'feedback_score': self.feedback_score,
            'feedback_comment': self.feedback_comment,
            'is_helpful': self.is_helpful,
            'status': self.status,
            'response_time_ms': self.response_time_ms,
            'source': self.source,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ModelConfigurationRecord(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型配置记录模型
    
    存储模型配置的历史记录。
    """
    __tablename__ = 'model_configuration_records'
    
    # 关联信息
    configuration_id = Column(String(100), unique=True, nullable=False, index=True, comment="配置ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    
    # 模型信息
    model_name = Column(String(200), nullable=False, index=True, comment="模型名称")
    task_type = Column(String(100), nullable=False, index=True, comment="任务类型")
    framework = Column(String(50), comment="框架")
    model_type = Column(String(50), comment="模型类型")
    
    # 数据集信息
    dataset_id = Column(String(100), index=True, comment="数据集ID")
    dataset_info = Column(JSON, comment="数据集信息")
    
    # 配置内容
    hyperparameters = Column(JSON, comment="超参数配置")
    training_config = Column(JSON, comment="训练配置")
    hardware_config = Column(JSON, comment="硬件配置")
    full_config = Column(JSON, comment="完整配置")
    
    # 配置来源
    config_source = Column(String(50), default='auto', comment="配置来源 (auto, manual, template)")
    template_id = Column(String(100), comment="模板ID（如果使用模板）")
    
    # 状态
    status = Column(String(20), default='active', index=True, comment="状态")
    is_default = Column(Boolean, default=False, comment="是否为默认配置")
    
    # 使用统计
    usage_count = Column(Integer, default=0, comment="使用次数")
    last_used_at = Column(DateTime, comment="最后使用时间")
    
    # 元数据
    tags = Column(JSON, comment="标签")
    metadata_ = Column(JSON, comment="额外元数据")
    
    def __repr__(self):
        return f"<ModelConfigurationRecord(id='{self.id}', model_name='{self.model_name}', task_type='{self.task_type}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'tenant_id': self.tenant_id,
            'configuration_id': self.configuration_id,
            'user_id': self.user_id,
            'model_name': self.model_name,
            'task_type': self.task_type,
            'framework': self.framework,
            'model_type': self.model_type,
            'dataset_id': self.dataset_id,
            'dataset_info': self.dataset_info,
            'hyperparameters': self.hyperparameters,
            'training_config': self.training_config,
            'hardware_config': self.hardware_config,
            'full_config': self.full_config,
            'config_source': self.config_source,
            'template_id': self.template_id,
            'status': self.status,
            'is_default': self.is_default,
            'usage_count': self.usage_count,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ModelCatalogEntry(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型目录条目模型
    
    存储可用的模型信息，支持租户自定义模型。
    """
    __tablename__ = 'model_catalog_entries'
    
    # 模型信息
    model_name = Column(String(200), nullable=False, index=True, comment="模型名称")
    task_type = Column(String(100), nullable=False, index=True, comment="适用任务类型")
    framework = Column(String(50), nullable=False, index=True, comment="框架")
    model_type = Column(String(50), nullable=False, index=True, comment="模型类型")
    
    # 描述信息
    display_name = Column(String(200), comment="显示名称")
    description = Column(Text, comment="模型描述")
    version = Column(String(50), default='1.0.0', comment="版本")
    
    # 性能指标
    performance_metrics = Column(JSON, comment="性能指标")
    benchmark_results = Column(JSON, comment="基准测试结果")
    
    # 资源要求
    hardware_requirements = Column(JSON, comment="硬件要求")
    min_gpu_memory = Column(String(20), comment="最小GPU内存")
    min_cpu_cores = Column(Integer, comment="最小CPU核心数")
    
    # 默认配置
    default_hyperparameters = Column(JSON, comment="默认超参数")
    default_training_config = Column(JSON, comment="默认训练配置")
    
    # 标签和分类
    tags = Column(JSON, comment="标签")
    categories = Column(JSON, comment="分类")
    recommended_for = Column(JSON, comment="推荐场景")
    
    # 可用性
    is_enabled = Column(Boolean, default=True, comment="是否启用")
    is_public = Column(Boolean, default=True, comment="是否公开")
    is_system = Column(Boolean, default=True, comment="是否系统内置")
    
    # 排序和优先级
    sort_order = Column(Integer, default=0, comment="排序顺序")
    priority = Column(Integer, default=0, comment="优先级")
    
    # 使用统计
    usage_count = Column(Integer, default=0, comment="使用次数")
    recommendation_count = Column(Integer, default=0, comment="推荐次数")
    selection_count = Column(Integer, default=0, comment="被选择次数")
    avg_rating = Column(Float, comment="平均评分")
    
    # 元数据
    source_url = Column(String(500), comment="源地址")
    documentation_url = Column(String(500), comment="文档地址")
    metadata_ = Column(JSON, comment="额外元数据")
    
    def __repr__(self):
        return f"<ModelCatalogEntry(model_name='{self.model_name}', task_type='{self.task_type}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'tenant_id': self.tenant_id,
            'model_name': self.model_name,
            'task_type': self.task_type,
            'framework': self.framework,
            'model_type': self.model_type,
            'display_name': self.display_name,
            'description': self.description,
            'version': self.version,
            'performance_metrics': self.performance_metrics,
            'benchmark_results': self.benchmark_results,
            'hardware_requirements': self.hardware_requirements,
            'min_gpu_memory': self.min_gpu_memory,
            'min_cpu_cores': self.min_cpu_cores,
            'default_hyperparameters': self.default_hyperparameters,
            'default_training_config': self.default_training_config,
            'tags': self.tags,
            'categories': self.categories,
            'recommended_for': self.recommended_for,
            'is_enabled': self.is_enabled,
            'is_public': self.is_public,
            'is_system': self.is_system,
            'sort_order': self.sort_order,
            'priority': self.priority,
            'usage_count': self.usage_count,
            'recommendation_count': self.recommendation_count,
            'selection_count': self.selection_count,
            'avg_rating': self.avg_rating,
            'source_url': self.source_url,
            'documentation_url': self.documentation_url,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class DeploymentModeConfig(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """部署模式配置模型
    
    存储各种部署模式和发布策略的配置信息，支持租户级别自定义。
    """
    __tablename__ = 'deployment_mode_configs'
    
    # 配置类型：mode（部署模式）或 strategy（发布策略）
    config_type = Column(String(20), nullable=False, index=True, comment="配置类型 (mode, strategy)")
    
    # 配置标识
    code = Column(String(50), nullable=False, index=True, comment="配置代码")
    name = Column(String(100), nullable=False, comment="显示名称")
    description = Column(Text, comment="详细描述")
    
    # 图标和分类
    icon = Column(String(100), comment="图标标识")
    category = Column(String(50), index=True, comment="分类")
    
    # 配置详情
    default_config = Column(JSON, comment="默认配置参数")
    required_resources = Column(JSON, comment="所需资源配置")
    supported_features = Column(JSON, comment="支持的特性列表")
    limitations = Column(JSON, comment="限制条件")
    
    # 状态和可用性
    is_enabled = Column(Boolean, default=True, comment="是否启用")
    is_default = Column(Boolean, default=False, comment="是否为默认选项")
    is_system = Column(Boolean, default=True, comment="是否为系统内置")
    
    # 使用限制
    min_replicas = Column(Integer, default=1, comment="最小副本数")
    max_replicas = Column(Integer, default=100, comment="最大副本数")
    requires_gpu = Column(Boolean, default=False, comment="是否需要GPU")
    
    # 推荐场景
    recommended_scenarios = Column(JSON, comment="推荐使用场景")
    
    # 使用统计
    usage_count = Column(Integer, default=0, comment="使用次数")
    
    # 排序
    sort_order = Column(Integer, default=0, comment="排序顺序")
    
    # 元数据
    tags = Column(JSON, comment="标签")
    extra_data = Column(JSON, comment="额外数据")
    
    def __repr__(self):
        return f"<DeploymentModeConfig(code='{self.code}', config_type='{self.config_type}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'tenant_id': self.tenant_id,
            'config_type': self.config_type,
            'code': self.code,
            'name': self.name,
            'description': self.description,
            'icon': self.icon,
            'category': self.category,
            'default_config': self.default_config,
            'required_resources': self.required_resources,
            'supported_features': self.supported_features,
            'limitations': self.limitations,
            'is_enabled': self.is_enabled,
            'is_default': self.is_default,
            'is_system': self.is_system,
            'min_replicas': self.min_replicas,
            'max_replicas': self.max_replicas,
            'requires_gpu': self.requires_gpu,
            'recommended_scenarios': self.recommended_scenarios,
            'usage_count': self.usage_count,
            'sort_order': self.sort_order,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


# 创建索引
Index('idx_training_sessions_user', TrainingSession.user_id)
Index('idx_training_sessions_status', TrainingSession.status)
Index('idx_training_progress_session', TrainingProgress.session_id)
Index('idx_three_stage_sessions_user', ThreeStageSession.user_id)
Index('idx_three_stage_sessions_status', ThreeStageSession.status)
Index('idx_three_stage_progress_session', ThreeStageProgress.session_id)

# 超参数优化索引
Index('idx_hp_optimization_user', HyperparameterOptimization.user_id)
Index('idx_hp_optimization_status', HyperparameterOptimization.status)
Index('idx_hp_optimization_scenario', HyperparameterOptimization.scenario_type)
Index('idx_hp_trial_optimization', HyperparameterTrial.optimization_id)
Index('idx_hp_trial_status', HyperparameterTrial.status)
Index('idx_hp_search_space_user', HyperparameterSearchSpace.user_id)
Index('idx_hp_search_space_scenario', HyperparameterSearchSpace.scenario_type)

# 智能决策索引
Index('idx_intelligent_decision_user', IntelligentDecision.user_id)
Index('idx_intelligent_decision_scenario', IntelligentDecision.scenario)
Index('idx_intelligent_decision_decision_id', IntelligentDecision.decision_id)
Index('idx_adaptive_optimization_user', AdaptiveOptimization.user_id)
Index('idx_adaptive_optimization_parameter', AdaptiveOptimization.parameter_name)
Index('idx_knowledge_base_user', KnowledgeBase.user_id)
Index('idx_knowledge_base_type', KnowledgeBase.knowledge_type)
Index('idx_experience_record_user', ExperienceRecord.user_id)
Index('idx_experience_record_scenario', ExperienceRecord.scenario)

# 模型部署索引
Index('idx_model_deployment_user', ModelDeployment.user_id)
Index('idx_model_deployment_tenant', ModelDeployment.tenant_id)
Index('idx_model_deployment_model', ModelDeployment.model_id)
Index('idx_model_deployment_status', ModelDeployment.status)
Index('idx_model_deployment_deployment_id', ModelDeployment.deployment_id)
Index('idx_model_deployment_log_deployment', ModelDeploymentLog.deployment_id)
Index('idx_model_deployment_log_action', ModelDeploymentLog.action)
Index('idx_model_service_user', ModelService.user_id)
Index('idx_model_service_tenant', ModelService.tenant_id)
Index('idx_model_service_model', ModelService.model_id)
Index('idx_model_service_status', ModelService.status)

# 部署审计事件索引
Index('idx_audit_event_tenant', DeploymentAuditEvent.tenant_id)
Index('idx_audit_event_deployment', DeploymentAuditEvent.deployment_id)
Index('idx_audit_event_user', DeploymentAuditEvent.user_id)
Index('idx_audit_event_type', DeploymentAuditEvent.event_type)
Index('idx_audit_event_action', DeploymentAuditEvent.action)
Index('idx_audit_event_status', DeploymentAuditEvent.status)
Index('idx_audit_event_time', DeploymentAuditEvent.event_time)
Index('idx_audit_event_correlation', DeploymentAuditEvent.correlation_id)

# 模型评估索引
Index('idx_model_evaluation_tenant', ModelEvaluation.tenant_id)
Index('idx_model_evaluation_model', ModelEvaluation.model_id)
Index('idx_model_evaluation_dataset', ModelEvaluation.dataset_id)
Index('idx_model_evaluation_user', ModelEvaluation.user_id)
Index('idx_model_evaluation_status', ModelEvaluation.status)
Index('idx_model_evaluation_type', ModelEvaluation.evaluation_type)
Index('idx_model_evaluation_metric_evaluation', ModelEvaluationMetric.evaluation_id)
Index('idx_model_evaluation_metric_name', ModelEvaluationMetric.metric_name)
Index('idx_model_comparison_tenant', ModelComparisonRecord.tenant_id)
Index('idx_model_comparison_user', ModelComparisonRecord.user_id)
Index('idx_model_comparison_dataset', ModelComparisonRecord.dataset_id)
Index('idx_model_comparison_status', ModelComparisonRecord.status)

# 模型优化索引
Index('idx_model_optimization_tenant', ModelOptimization.tenant_id)
Index('idx_model_optimization_user', ModelOptimization.user_id)
Index('idx_model_optimization_original_model', ModelOptimization.original_model_id)
Index('idx_model_optimization_type', ModelOptimization.optimization_type)
Index('idx_model_optimization_technique', ModelOptimization.technique)
Index('idx_model_optimization_status', ModelOptimization.status)

# 模型选择索引
Index('idx_model_recommendation_tenant', ModelRecommendationRecord.tenant_id)
Index('idx_model_recommendation_user', ModelRecommendationRecord.user_id)
Index('idx_model_recommendation_task_type', ModelRecommendationRecord.task_type)
Index('idx_model_recommendation_status', ModelRecommendationRecord.status)
Index('idx_model_configuration_tenant', ModelConfigurationRecord.tenant_id)
Index('idx_model_configuration_user', ModelConfigurationRecord.user_id)
Index('idx_model_configuration_model_name', ModelConfigurationRecord.model_name)
Index('idx_model_configuration_task_type', ModelConfigurationRecord.task_type)
Index('idx_model_catalog_tenant', ModelCatalogEntry.tenant_id)
Index('idx_model_catalog_model_name', ModelCatalogEntry.model_name)
Index('idx_model_catalog_task_type', ModelCatalogEntry.task_type)
Index('idx_model_catalog_framework', ModelCatalogEntry.framework)

# 部署模式配置索引
Index('idx_deployment_mode_config_tenant', DeploymentModeConfig.tenant_id)
Index('idx_deployment_mode_config_type', DeploymentModeConfig.config_type)
Index('idx_deployment_mode_config_code', DeploymentModeConfig.code)


# ============================================================================
# 训练流水线模型
# ============================================================================

class TrainingPipeline(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """训练流水线模型
    
    定义流水线的结构，包含多个步骤的配置
    """
    __tablename__ = 'training_pipelines'
    
    pipeline_id = Column(String(100), unique=True, nullable=False, index=True, comment="流水线ID")
    name = Column(String(200), nullable=False, index=True, comment="流水线名称")
    description = Column(Text, comment="流水线描述")
    user_id = Column(String(36), nullable=False, index=True, comment="创建者用户ID")
    
    # 流水线状态
    status = Column(String(30), default='draft', index=True, comment="状态 (draft, created, running, completed, failed)")
    
    # 关联的模型和数据集
    model_name = Column(String(200), comment="模型名称")
    model_id = Column(String(100), index=True, comment="模型ID")
    dataset_id = Column(String(100), index=True, comment="数据集ID")
    
    # 流水线配置
    steps_config = Column(JSON, nullable=False, comment="步骤配置列表")
    global_config = Column(JSON, comment="全局配置参数")
    enable_rollback = Column(Boolean, default=True, comment="是否启用回滚")
    
    # 版本控制
    version = Column(Integer, default=1, comment="版本号")
    parent_pipeline_id = Column(String(100), index=True, comment="父流水线ID（用于版本追踪）")
    
    # 标签和元数据
    tags = Column(JSON, comment="标签列表")
    metadata_ = Column(JSON, comment="额外元数据")
    
    def __repr__(self):
        return f"<TrainingPipeline(id='{self.id}', pipeline_id='{self.pipeline_id}', name='{self.name}', status='{self.status}')>"
    
    def to_dict(self):
        return {
            'id': str(self.id) if self.id else None,
            'tenant_id': str(self.tenant_id) if self.tenant_id else None,
            'pipeline_id': self.pipeline_id,
            'name': self.name,
            'description': self.description,
            'user_id': self.user_id,
            'status': self.status,
            'model_name': self.model_name,
            'model_id': self.model_id,
            'dataset_id': self.dataset_id,
            'steps_config': self.steps_config,
            'global_config': self.global_config,
            'enable_rollback': self.enable_rollback,
            'version': self.version,
            'parent_pipeline_id': self.parent_pipeline_id,
            'tags': self.tags,
            'metadata': self.metadata_,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class PipelineExecution(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """流水线执行记录模型
    
    记录每次流水线执行的详细信息
    """
    __tablename__ = 'pipeline_executions'
    
    execution_id = Column(String(100), unique=True, nullable=False, index=True, comment="执行ID")
    pipeline_id = Column(String(100), nullable=False, index=True, comment="流水线ID")
    session_id = Column(String(100), unique=True, index=True, comment="训练会话ID")
    user_id = Column(String(36), nullable=False, index=True, comment="执行者用户ID")
    
    # 执行状态
    status = Column(String(30), default='queued', index=True, comment="执行状态")
    current_step = Column(Integer, default=0, comment="当前步骤索引")
    total_steps = Column(Integer, default=0, comment="总步骤数")
    progress = Column(Float, default=0.0, comment="执行进度 (0-100)")
    
    # 时间信息
    queued_at = Column(DateTime, default=datetime.utcnow, comment="排队时间")
    started_at = Column(DateTime, comment="开始执行时间")
    completed_at = Column(DateTime, comment="完成时间")
    paused_at = Column(DateTime, comment="暂停时间")
    resumed_at = Column(DateTime, comment="恢复时间")
    
    # 执行配置（执行时的快照）
    pipeline_snapshot = Column(JSON, comment="执行时的流水线配置快照")
    runtime_config = Column(JSON, comment="运行时配置")
    
    # 执行结果
    result = Column(JSON, comment="执行结果")
    error_message = Column(Text, comment="错误信息")
    
    # 关联信息
    parent_execution_id = Column(String(100), index=True, comment="父执行ID（用于重试/恢复）")
    retry_count = Column(Integer, default=0, comment="重试次数")
    
    # 资源使用
    resource_usage = Column(JSON, comment="资源使用统计")
    
    # 元数据
    metadata_ = Column(JSON, comment="额外元数据")
    
    def __repr__(self):
        return f"<PipelineExecution(id='{self.id}', execution_id='{self.execution_id}', status='{self.status}')>"
    
    def to_dict(self):
        return {
            'id': str(self.id) if self.id else None,
            'tenant_id': str(self.tenant_id) if self.tenant_id else None,
            'execution_id': self.execution_id,
            'pipeline_id': self.pipeline_id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'status': self.status,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'progress': self.progress,
            'queued_at': self.queued_at.isoformat() if self.queued_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'paused_at': self.paused_at.isoformat() if self.paused_at else None,
            'resumed_at': self.resumed_at.isoformat() if self.resumed_at else None,
            'pipeline_snapshot': self.pipeline_snapshot,
            'runtime_config': self.runtime_config,
            'result': self.result,
            'error_message': self.error_message,
            'parent_execution_id': self.parent_execution_id,
            'retry_count': self.retry_count,
            'resource_usage': self.resource_usage,
            'metadata': self.metadata_,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class PipelineStepExecution(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """流水线步骤执行记录模型
    
    记录每个步骤的执行详情
    """
    __tablename__ = 'pipeline_step_executions'
    
    step_execution_id = Column(String(100), unique=True, nullable=False, index=True, comment="步骤执行ID")
    execution_id = Column(String(100), nullable=False, index=True, comment="流水线执行ID")
    step_index = Column(Integer, nullable=False, comment="步骤索引")
    step_name = Column(String(200), nullable=False, comment="步骤名称")
    step_type = Column(String(50), nullable=False, index=True, comment="步骤类型")
    
    # 执行状态
    status = Column(String(30), default='pending', index=True, comment="步骤状态")
    progress = Column(Float, default=0.0, comment="步骤进度 (0-100)")
    
    # 时间信息
    started_at = Column(DateTime, comment="开始时间")
    completed_at = Column(DateTime, comment="完成时间")
    duration_seconds = Column(Float, comment="执行时长（秒）")
    
    # 步骤配置和结果
    step_config = Column(JSON, comment="步骤配置")
    input_data = Column(JSON, comment="输入数据/参数")
    output_data = Column(JSON, comment="输出数据/结果")
    
    # 训练相关
    metrics = Column(JSON, comment="训练指标（loss, accuracy等）")
    checkpoint_path = Column(String(500), comment="检查点路径")
    
    # 错误处理
    error_message = Column(Text, comment="错误信息")
    failure_policy = Column(String(30), default='rollback', comment="失败策略")
    retry_count = Column(Integer, default=0, comment="重试次数")
    
    # 元数据
    metadata_ = Column(JSON, comment="额外元数据")
    
    def __repr__(self):
        return f"<PipelineStepExecution(id='{self.id}', step_name='{self.step_name}', status='{self.status}')>"
    
    def to_dict(self):
        return {
            'id': str(self.id) if self.id else None,
            'tenant_id': str(self.tenant_id) if self.tenant_id else None,
            'step_execution_id': self.step_execution_id,
            'execution_id': self.execution_id,
            'step_index': self.step_index,
            'step_name': self.step_name,
            'step_type': self.step_type,
            'status': self.status,
            'progress': self.progress,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.duration_seconds,
            'step_config': self.step_config,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'metrics': self.metrics,
            'checkpoint_path': self.checkpoint_path,
            'error_message': self.error_message,
            'failure_policy': self.failure_policy,
            'retry_count': self.retry_count,
            'metadata': self.metadata_,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class PipelineTemplate(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """流水线模板模型
    
    预定义的流水线模板，便于快速创建
    """
    __tablename__ = 'pipeline_templates'
    
    template_id = Column(String(100), unique=True, nullable=False, index=True, comment="模板ID")
    name = Column(String(200), nullable=False, index=True, comment="模板名称")
    description = Column(Text, comment="模板描述")
    user_id = Column(String(36), index=True, comment="创建者用户ID（可为null表示系统模板）")
    
    # 模板类型
    template_type = Column(String(50), default='custom', index=True, comment="模板类型 (system, custom)")
    category = Column(String(100), index=True, comment="分类 (nlp, cv, multimodal等)")
    
    # 模板内容
    steps_template = Column(JSON, nullable=False, comment="步骤模板配置")
    default_config = Column(JSON, comment="默认配置")
    required_params = Column(JSON, comment="必需参数列表")
    
    # 版本和状态
    version = Column(String(20), default='1.0.0', comment="版本号")
    is_active = Column(Boolean, default=True, comment="是否激活")
    is_public = Column(Boolean, default=False, comment="是否公开")
    
    # 使用统计
    usage_count = Column(Integer, default=0, comment="使用次数")
    
    # 标签和元数据
    tags = Column(JSON, comment="标签列表")
    metadata_ = Column(JSON, comment="额外元数据")
    
    def __repr__(self):
        return f"<PipelineTemplate(id='{self.id}', template_id='{self.template_id}', name='{self.name}')>"
    
    def to_dict(self):
        return {
            'id': str(self.id) if self.id else None,
            'tenant_id': str(self.tenant_id) if self.tenant_id else None,
            'template_id': self.template_id,
            'name': self.name,
            'description': self.description,
            'user_id': self.user_id,
            'template_type': self.template_type,
            'category': self.category,
            'steps_template': self.steps_template,
            'default_config': self.default_config,
            'required_params': self.required_params,
            'version': self.version,
            'is_active': self.is_active,
            'is_public': self.is_public,
            'usage_count': self.usage_count,
            'tags': self.tags,
            'metadata': self.metadata_,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


# 训练流水线索引
Index('idx_training_pipeline_tenant', TrainingPipeline.tenant_id)
Index('idx_training_pipeline_user', TrainingPipeline.user_id)
Index('idx_training_pipeline_status', TrainingPipeline.status)
Index('idx_training_pipeline_model', TrainingPipeline.model_id)
Index('idx_pipeline_execution_tenant', PipelineExecution.tenant_id)
Index('idx_pipeline_execution_pipeline', PipelineExecution.pipeline_id)
Index('idx_pipeline_execution_user', PipelineExecution.user_id)
Index('idx_pipeline_execution_status', PipelineExecution.status)
Index('idx_pipeline_execution_session', PipelineExecution.session_id)
Index('idx_pipeline_step_execution_tenant', PipelineStepExecution.tenant_id)
Index('idx_pipeline_step_execution_execution', PipelineStepExecution.execution_id)
Index('idx_pipeline_step_execution_type', PipelineStepExecution.step_type)
Index('idx_pipeline_step_execution_status', PipelineStepExecution.status)
Index('idx_pipeline_template_tenant', PipelineTemplate.tenant_id)
Index('idx_pipeline_template_type', PipelineTemplate.template_type)
Index('idx_pipeline_template_category', PipelineTemplate.category)
Index('idx_pipeline_template_active', PipelineTemplate.is_active)


# ==============================================================================
# 训练任务控制相关模型
# ==============================================================================

class TrainingJob(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """训练任务模型
    
    用于管理和控制训练任务的生命周期，支持开始、暂停、恢复、取消等操作。
    """
    __tablename__ = 'training_jobs'
    
    # 任务标识
    job_id = Column(String(64), unique=True, nullable=False, index=True, comment="任务唯一标识")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    
    # 任务基本信息
    name = Column(String(200), nullable=False, comment="任务名称")
    description = Column(Text, comment="任务描述")
    
    # 训练配置
    scenario_type = Column(String(50), nullable=False, index=True, comment="训练场景类型")
    training_mode = Column(String(50), default='standard', comment="训练模式(standard/distributed/multimodal/distillation)")
    config = Column(JSON, comment="完整训练配置")
    
    # 模型和数据集
    model_id = Column(String(100), comment="模型ID")
    model_name = Column(String(200), comment="模型名称")
    dataset_id = Column(String(100), comment="数据集ID")
    
    # 状态和进度
    status = Column(String(20), default=TrainingStatus.PENDING.value, index=True, comment="任务状态")
    progress = Column(Float, default=0.0, comment="进度(0-100)")
    current_epoch = Column(Integer, default=0, comment="当前轮次")
    total_epochs = Column(Integer, comment="总轮次")
    current_step = Column(Integer, default=0, comment="当前步骤")
    total_steps = Column(Integer, comment="总步骤数")
    
    # 性能指标
    metrics = Column(JSON, comment="当前指标(loss/accuracy等)")
    best_metrics = Column(JSON, comment="最佳指标")
    
    # 优先级和调度
    priority = Column(Integer, default=5, comment="优先级(1-10,1最高)")
    scheduled_at = Column(DateTime, comment="计划执行时间")
    
    # 资源配置
    resource_config = Column(JSON, comment="资源配置(GPU/CPU/内存等)")
    allocated_resources = Column(JSON, comment="已分配资源")
    
    # 检查点
    checkpoint_path = Column(String(500), comment="最新检查点路径")
    checkpoint_epoch = Column(Integer, comment="检查点对应的轮次")
    auto_checkpoint = Column(Boolean, default=True, comment="是否自动保存检查点")
    
    # 时间记录
    started_at = Column(DateTime, comment="开始时间")
    paused_at = Column(DateTime, comment="暂停时间")
    resumed_at = Column(DateTime, comment="恢复时间")
    completed_at = Column(DateTime, comment="完成时间")
    estimated_end_time = Column(DateTime, comment="预计结束时间")
    
    # 错误信息
    error_message = Column(Text, comment="错误信息")
    error_details = Column(JSON, comment="详细错误信息")
    retry_count = Column(Integer, default=0, comment="重试次数")
    max_retries = Column(Integer, default=3, comment="最大重试次数")
    
    # 结果
    result = Column(JSON, comment="训练结果")
    output_path = Column(String(500), comment="输出路径")
    
    # 关联
    session_id = Column(String(36), comment="关联的训练会话ID")
    pipeline_id = Column(String(36), comment="关联的流水线ID")
    parent_job_id = Column(String(64), comment="父任务ID(用于任务分解)")
    
    # 标签和元数据
    tags = Column(JSON, comment="标签列表")
    extra_data = Column(JSON, comment="额外数据")
    
    def __repr__(self):
        return f"<TrainingJob(id='{self.id}', job_id='{self.job_id}', name='{self.name}', status='{self.status}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'job_id': self.job_id,
            'tenant_id': str(self.tenant_id) if self.tenant_id else None,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description,
            'scenario_type': self.scenario_type,
            'training_mode': self.training_mode,
            'config': self.config,
            'model_id': self.model_id,
            'model_name': self.model_name,
            'dataset_id': self.dataset_id,
            'status': self.status,
            'progress': self.progress,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'metrics': self.metrics,
            'best_metrics': self.best_metrics,
            'priority': self.priority,
            'scheduled_at': self.scheduled_at.isoformat() if self.scheduled_at else None,
            'resource_config': self.resource_config,
            'allocated_resources': self.allocated_resources,
            'checkpoint_path': self.checkpoint_path,
            'checkpoint_epoch': self.checkpoint_epoch,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'paused_at': self.paused_at.isoformat() if self.paused_at else None,
            'resumed_at': self.resumed_at.isoformat() if self.resumed_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'estimated_end_time': self.estimated_end_time.isoformat() if self.estimated_end_time else None,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'result': self.result,
            'output_path': self.output_path,
            'session_id': self.session_id,
            'pipeline_id': self.pipeline_id,
            'parent_job_id': self.parent_job_id,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class TrainingJobLog(Base, UUIDMixin, TimestampMixin):
    """训练任务日志模型
    
    记录训练任务的详细操作日志和状态变更历史。
    """
    __tablename__ = 'training_job_logs'
    
    job_id = Column(String(64), ForeignKey('training_jobs.job_id'), nullable=False, index=True, comment="任务ID")
    
    # 日志类型和级别
    log_type = Column(String(20), nullable=False, index=True, comment="日志类型(status_change/progress/error/checkpoint/metric)")
    log_level = Column(String(10), default='info', comment="日志级别(debug/info/warning/error)")
    
    # 内容
    message = Column(Text, nullable=False, comment="日志消息")
    details = Column(JSON, comment="详细信息")
    
    # 状态变更
    from_status = Column(String(20), comment="变更前状态")
    to_status = Column(String(20), comment="变更后状态")
    
    # 指标记录
    epoch = Column(Integer, comment="轮次")
    step = Column(Integer, comment="步骤")
    metrics = Column(JSON, comment="指标快照")
    
    # 来源
    source = Column(String(50), comment="日志来源(api/system/scheduler)")
    operator_id = Column(String(36), comment="操作者ID")
    
    def __repr__(self):
        return f"<TrainingJobLog(id='{self.id}', job_id='{self.job_id}', log_type='{self.log_type}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'job_id': self.job_id,
            'log_type': self.log_type,
            'log_level': self.log_level,
            'message': self.message,
            'details': self.details,
            'from_status': self.from_status,
            'to_status': self.to_status,
            'epoch': self.epoch,
            'step': self.step,
            'metrics': self.metrics,
            'source': self.source,
            'operator_id': self.operator_id,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


# 训练任务索引
Index('idx_training_job_tenant', TrainingJob.tenant_id)
Index('idx_training_job_user', TrainingJob.user_id)
Index('idx_training_job_status', TrainingJob.status)
Index('idx_training_job_scenario', TrainingJob.scenario_type)
Index('idx_training_job_priority', TrainingJob.priority)
Index('idx_training_job_scheduled', TrainingJob.scheduled_at)
Index('idx_training_job_log_job', TrainingJobLog.job_id)
Index('idx_training_job_log_type', TrainingJobLog.log_type)