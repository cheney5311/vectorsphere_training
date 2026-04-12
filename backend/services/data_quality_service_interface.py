"""数据质量服务接口

定义数据质量服务的接口规范。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class DataQualityServiceInterface(ABC):
    """数据质量服务接口"""

    # ========================================================================
    # 质量评估相关接口
    # ========================================================================

    @abstractmethod
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
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_assessment_by_id(self, assessment_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取评估记录

        Args:
            assessment_id: 评估记录ID

        Returns:
            Optional[Dict[str, Any]]: 评估记录
        """
        pass

    # ========================================================================
    # 问题检测相关接口
    # ========================================================================

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    # ========================================================================
    # 数据清理相关接口
    # ========================================================================

    @abstractmethod
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
            Dict[str, Any]: 清理后的数据集对象
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_cleaning_record(self, cleaning_id: str) -> Optional[Dict[str, Any]]:
        """获取清理记录

        Args:
            cleaning_id: 清理记录ID

        Returns:
            Optional[Dict[str, Any]]: 清理记录
        """
        pass

    @abstractmethod
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
        pass

    # ========================================================================
    # 质量规则相关接口
    # ========================================================================

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def delete_quality_rule(self, rule_id: str) -> bool:
        """删除质量规则

        Args:
            rule_id: 规则ID

        Returns:
            bool: 是否删除成功
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    # ========================================================================
    # 质量报告相关接口
    # ========================================================================

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_report_by_id(self, report_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取报告

        Args:
            report_id: 报告ID

        Returns:
            Optional[Dict[str, Any]]: 报告内容
        """
        pass

    # ========================================================================
    # 质量监控相关接口
    # ========================================================================

    @abstractmethod
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
        pass

    @abstractmethod
    def get_monitoring_status(self, dataset_id: str) -> Dict[str, Any]:
        """获取监控状态

        Args:
            dataset_id: 数据集ID

        Returns:
            Dict[str, Any]: 监控状态
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def disable_monitoring(self, dataset_id: str) -> bool:
        """禁用监控

        Args:
            dataset_id: 数据集ID

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    # ========================================================================
    # 统计相关接口
    # ========================================================================

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass
