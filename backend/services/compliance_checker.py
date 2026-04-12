# -*- coding: utf-8 -*-
"""
合规检查服务
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from backend.modules.security.models import (
    ComplianceStandard, ComplianceLevel, DataCategory, ProcessingPurpose,
    DataProcessingRecord, ComplianceViolation, ComplianceReport
)


class ComplianceRule(ABC):
    """合规规则基类"""
    
    def __init__(self, rule_id: str, name: str, description: str, severity: str):
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.severity = severity
    
    @abstractmethod
    def check(self, context: Dict[str, Any]) -> List[ComplianceViolation]:
        """检查合规性
        
        Args:
            context: 检查上下文
            
        Returns:
            违规列表
        """
        pass


class GDPRRules:
    """GDPR规则集"""
    
    class ConsentRule(ComplianceRule):
        """同意规则"""
        
        def __init__(self):
            super().__init__(
                rule_id="gdpr_consent_001",
                name="Valid Consent Required",
                description="Processing personal data requires valid consent",
                severity="high"
            )
        
        def check(self, context: Dict[str, Any]) -> List[ComplianceViolation]:
            violations = []
            processing_records = context.get('processing_records', [])
            
            for record in processing_records:
                if not record.consent_given and ProcessingPurpose.CONSENT in record.processing_purposes:
                    violation = ComplianceViolation(
                        id=f"gdpr_consent_violation_{record.id}",
                        standard=ComplianceStandard.GDPR,
                        rule_id=self.rule_id,
                        rule_name=self.name,
                        severity=self.severity,
                        description=f"Processing personal data without valid consent for record {record.id}",
                        affected_data=[record.id],
                        remediation_steps=[
                            "Obtain explicit consent from data subject",
                            "Update consent records",
                            "Implement consent management system"
                        ],
                        detected_at=datetime.now(),
                        resolved_at=None,
                        status="open"
                    )
                    violations.append(violation)
            
            return violations
    
    class DataRetentionRule(ComplianceRule):
        """数据保留规则"""
        
        def __init__(self):
            super().__init__(
                rule_id="gdpr_retention_001",
                name="Data Retention Limits",
                description="Personal data must not be kept longer than necessary",
                severity="medium"
            )
        
        def check(self, context: Dict[str, Any]) -> List[ComplianceViolation]:
            violations = []
            processing_records = context.get('processing_records', [])
            current_time = datetime.now()
            
            for record in processing_records:
                if record.retention_period:
                    retention_deadline = record.created_at + timedelta(days=record.retention_period)
                    if current_time > retention_deadline:
                        violation = ComplianceViolation(
                            id=f"gdpr_retention_violation_{record.id}",
                            standard=ComplianceStandard.GDPR,
                            rule_id=self.rule_id,
                            rule_name=self.name,
                            severity=self.severity,
                            description=f"Data retention period exceeded for record {record.id}",
                            affected_data=[record.id],
                            remediation_steps=[
                                "Delete or anonymize expired data",
                                "Review retention policies",
                                "Implement automated data deletion"
                            ],
                            detected_at=datetime.now(),
                            resolved_at=None,
                            status="open"
                        )
                        violations.append(violation)
            
            return violations
    
    class DataMinimizationRule(ComplianceRule):
        """数据最小化规则"""
        
        def __init__(self):
            super().__init__(
                rule_id="gdpr_minimization_001",
                name="Data Minimization",
                description="Only necessary personal data should be processed",
                severity="medium"
            )
        
        def check(self, context: Dict[str, Any]) -> List[ComplianceViolation]:
            violations = []
            processing_records = context.get('processing_records', [])
            
            # 定义目的与数据类别的合理映射
            purpose_data_mapping = {
                ProcessingPurpose.CONTRACT: [DataCategory.PERSONAL_DATA, DataCategory.FINANCIAL_DATA],
                ProcessingPurpose.CONSENT: [DataCategory.PERSONAL_DATA, DataCategory.BEHAVIORAL_DATA],
                ProcessingPurpose.LEGAL_OBLIGATION: [DataCategory.PERSONAL_DATA, DataCategory.FINANCIAL_DATA],
            }
            
            for record in processing_records:
                for purpose in record.processing_purposes:
                    allowed_categories = purpose_data_mapping.get(purpose, [])
                    excessive_categories = [cat for cat in record.data_categories if cat not in allowed_categories]
                    
                    if excessive_categories:
                        violation = ComplianceViolation(
                            id=f"gdpr_minimization_violation_{record.id}",
                            standard=ComplianceStandard.GDPR,
                            rule_id=self.rule_id,
                            rule_name=self.name,
                            severity=self.severity,
                            description=f"Excessive data categories for purpose {purpose.value} in record {record.id}",
                            affected_data=[record.id],
                            remediation_steps=[
                                "Review data collection practices",
                                "Remove unnecessary data categories",
                                "Update privacy policies"
                            ],
                            detected_at=datetime.now(),
                            resolved_at=None,
                            status="open",
                            metadata={'excessive_categories': [cat.value for cat in excessive_categories]}
                        )
                        violations.append(violation)
            
            return violations


class CCPARules:
    """CCPA规则集"""
    
    class ConsumerRightsRule(ComplianceRule):
        """消费者权利规则"""
        
        def __init__(self):
            super().__init__(
                rule_id="ccpa_rights_001",
                name="Consumer Rights Implementation",
                description="Consumers must have rights to know, delete, and opt-out",
                severity="high"
            )
        
        def check(self, context: Dict[str, Any]) -> List[ComplianceViolation]:
            violations = []
            system_config = context.get('system_config', {})
            
            required_features = [
                'data_access_request',
                'data_deletion_request',
                'opt_out_mechanism',
                'privacy_policy'
            ]
            
            missing_features = [feature for feature in required_features 
                              if not system_config.get(feature, False)]
            
            if missing_features:
                violation = ComplianceViolation(
                    id="ccpa_rights_violation_001",
                    standard=ComplianceStandard.CCPA,
                    rule_id=self.rule_id,
                    rule_name=self.name,
                    severity=self.severity,
                    description=f"Missing consumer rights features: {', '.join(missing_features)}",
                    affected_data=["system_wide"],
                    remediation_steps=[
                        "Implement data access request mechanism",
                        "Implement data deletion request mechanism",
                        "Implement opt-out mechanism",
                        "Update privacy policy"
                    ],
                    detected_at=datetime.now(),
                    resolved_at=None,
                    status="open",
                    metadata={'missing_features': missing_features}
                )
                violations.append(violation)
            
            return violations


class SOC2Rules:
    """SOC 2规则集"""
    
    class AccessControlRule(ComplianceRule):
        """访问控制规则"""
        
        def __init__(self):
            super().__init__(
                rule_id="soc2_access_001",
                name="Logical Access Controls",
                description="System access must be properly controlled and monitored",
                severity="high"
            )
        
        def check(self, context: Dict[str, Any]) -> List[ComplianceViolation]:
            violations = []
            access_logs = context.get('access_logs', [])
            
            # 检查是否有未授权访问尝试
            failed_attempts = [log for log in access_logs if log.get('result') == 'failure']
            
            # 如果失败尝试过多，可能存在安全问题
            if len(failed_attempts) > 100:  # 阈值可配置
                violation = ComplianceViolation(
                    id="soc2_access_violation_001",
                    standard=ComplianceStandard.SOC2,
                    rule_id=self.rule_id,
                    rule_name=self.name,
                    severity=self.severity,
                    description=f"High number of failed access attempts: {len(failed_attempts)}",
                    affected_data=["access_control_system"],
                    remediation_steps=[
                        "Review access control policies",
                        "Implement account lockout mechanisms",
                        "Monitor for suspicious activities",
                        "Strengthen authentication requirements"
                    ],
                    detected_at=datetime.now(),
                    resolved_at=None,
                    status="open",
                    metadata={'failed_attempts_count': len(failed_attempts)}
                )
                violations.append(violation)
            
            return violations


class ComplianceChecker:
    """合规检查器
    
    提供多种合规标准的检查功能
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化规则集
        self.rules = {
            ComplianceStandard.GDPR: [
                GDPRRules.ConsentRule(),
                GDPRRules.DataRetentionRule(),
                GDPRRules.DataMinimizationRule()
            ],
            ComplianceStandard.CCPA: [
                CCPARules.ConsumerRightsRule()
            ],
            ComplianceStandard.SOC2: [
                SOC2Rules.AccessControlRule()
            ]
        }
        
        # 数据处理记录存储
        self.processing_records: Dict[str, DataProcessingRecord] = {}
        
        # 违规记录存储
        self.violations: Dict[str, ComplianceViolation] = {}
    
    def check_compliance(self, standard: ComplianceStandard, 
                        context: Optional[Dict[str, Any]] = None) -> ComplianceReport:
        """检查合规性
        
        Args:
            standard: 合规标准
            context: 检查上下文
            
        Returns:
            合规报告
        """
        if context is None:
            context = self._build_default_context()
        
        rules = self.rules.get(standard, [])
        all_violations = []
        
        # 执行所有规则检查
        for rule in rules:
            try:
                violations = rule.check(context)
                all_violations.extend(violations)
                
                # 存储违规记录
                for violation in violations:
                    self.violations[violation.id] = violation
                    
            except Exception as e:
                logging.error(f"Error checking rule {rule.rule_id}: {e}")
        
        # 计算合规分数
        total_rules = len(rules)
        compliant_rules = total_rules - len(all_violations)
        score = (compliant_rules / total_rules * 100) if total_rules > 0 else 100
        
        # 确定合规级别
        if score >= 95:
            level = ComplianceLevel.COMPLIANT
        elif score >= 70:
            level = ComplianceLevel.PARTIALLY_COMPLIANT
        else:
            level = ComplianceLevel.NON_COMPLIANT
        
        # 生成建议
        recommendations = self._generate_recommendations(standard, all_violations)
        
        # 创建报告
        report = ComplianceReport(
            id=f"compliance_report_{standard.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            standard=standard,
            overall_level=level,
            score=score,
            total_rules=total_rules,
            compliant_rules=compliant_rules,
            violations=all_violations,
            recommendations=recommendations,
            generated_at=datetime.now(),
            period_start=datetime.now() - timedelta(days=30),
            period_end=datetime.now()
        )
        
        return report
    
    def record_data_processing(self, record: DataProcessingRecord):
        """记录数据处理活动
        
        Args:
            record: 数据处理记录
        """
        self.processing_records[record.id] = record
    
    def get_processing_records(self, data_subject_id: Optional[str] = None) -> List[DataProcessingRecord]:
        """获取数据处理记录
        
        Args:
            data_subject_id: 数据主体ID
            
        Returns:
            处理记录列表
        """
        records = list(self.processing_records.values())
        
        if data_subject_id:
            records = [r for r in records if r.data_subject_id == data_subject_id]
        
        return records
    
    def handle_data_subject_request(self, request_type: str, data_subject_id: str) -> Dict[str, Any]:
        """处理数据主体请求
        
        Args:
            request_type: 请求类型（access, delete, portability, rectification）
            data_subject_id: 数据主体ID
            
        Returns:
            处理结果
        """
        records = self.get_processing_records(data_subject_id)
        
        if request_type == "access":
            return {
                'status': 'completed',
                'data': [record.__dict__ for record in records],
                'message': f'Found {len(records)} processing records'
            }
        
        elif request_type == "delete":
            # 标记删除（实际实现需要调用相应的删除服务）
            deleted_count = 0
            for record in records:
                if self._can_delete_record(record):
                    # 这里应该调用实际的删除服务
                    record.metadata['deletion_requested'] = datetime.now().isoformat()
                    deleted_count += 1
            
            return {
                'status': 'completed',
                'deleted_records': deleted_count,
                'message': f'Deleted {deleted_count} records'
            }
        
        elif request_type == "portability":
            # 数据可携带性
            portable_data = []
            for record in records:
                if self._is_portable_data(record):
                    portable_data.append({
                        'record_id': record.id,
                        'data_categories': [cat.value for cat in record.data_categories],
                        'created_at': record.created_at.isoformat(),
                        'metadata': record.metadata
                    })
            
            return {
                'status': 'completed',
                'portable_data': portable_data,
                'format': 'json',
                'message': f'Exported {len(portable_data)} portable records'
            }
        
        elif request_type == "rectification":
            return {
                'status': 'pending',
                'message': 'Rectification request received, manual review required',
                'records_found': len(records)
            }
        
        else:
            return {
                'status': 'error',
                'message': f'Unknown request type: {request_type}'
            }
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """获取合规仪表板数据
        
        Returns:
            仪表板数据
        """
        dashboard = {
            'overview': {
                'total_processing_records': len(self.processing_records),
                'total_violations': len(self.violations),
                'open_violations': len([v for v in self.violations.values() if v.status == 'open']),
                'last_check': datetime.now().isoformat()
            },
            'standards': {},
            'violations_by_severity': {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            },
            'recent_violations': [],
            'compliance_trends': []
        }
        
        # 按标准统计
        for standard in ComplianceStandard:
            standard_violations = [v for v in self.violations.values() if v.standard == standard]
            dashboard['standards'][standard.value] = {
                'total_violations': len(standard_violations),
                'open_violations': len([v for v in standard_violations if v.status == 'open']),
                'last_check': datetime.now().isoformat()
            }
        
        # 按严重程度统计
        for violation in self.violations.values():
            if violation.status == 'open':
                dashboard['violations_by_severity'][violation.severity] += 1
        
        # 最近的违规
        recent_violations = sorted(
            self.violations.values(),
            key=lambda v: v.detected_at,
            reverse=True
        )[:10]
        
        dashboard['recent_violations'] = [
            {
                'id': v.id,
                'standard': v.standard.value,
                'rule_name': v.rule_name,
                'severity': v.severity,
                'detected_at': v.detected_at.isoformat(),
                'status': v.status
            }
            for v in recent_violations
        ]
        
        return dashboard
    
    def resolve_violation(self, violation_id: str, resolution_notes: str) -> bool:
        """解决违规
        
        Args:
            violation_id: 违规ID
            resolution_notes: 解决说明
            
        Returns:
            是否成功
        """
        if violation_id in self.violations:
            violation = self.violations[violation_id]
            violation.status = 'resolved'
            violation.resolved_at = datetime.now()
            violation.metadata['resolution_notes'] = resolution_notes
            return True
        return False
    
    def _build_default_context(self) -> Dict[str, Any]:
        """构建默认检查上下文"""
        return {
            'processing_records': list(self.processing_records.values()),
            'system_config': self.config.get('system_config', {}),
            'access_logs': self.config.get('access_logs', []),
            'current_time': datetime.now()
        }
    
    def _generate_recommendations(self, standard: ComplianceStandard, 
                                violations: List[ComplianceViolation]) -> List[str]:
        """生成合规建议"""
        recommendations = []
        
        if standard == ComplianceStandard.GDPR:
            if any(v.rule_id.startswith('gdpr_consent') for v in violations):
                recommendations.append("Implement comprehensive consent management system")
            if any(v.rule_id.startswith('gdpr_retention') for v in violations):
                recommendations.append("Establish automated data retention and deletion policies")
            if any(v.rule_id.startswith('gdpr_minimization') for v in violations):
                recommendations.append("Review and minimize data collection practices")
        
        elif standard == ComplianceStandard.CCPA:
            if any(v.rule_id.startswith('ccpa_rights') for v in violations):
                recommendations.append("Implement consumer rights request handling system")
        
        elif standard == ComplianceStandard.SOC2:
            if any(v.rule_id.startswith('soc2_access') for v in violations):
                recommendations.append("Strengthen access control and monitoring systems")
        
        # 通用建议
        if violations:
            recommendations.extend([
                "Conduct regular compliance training for staff",
                "Implement automated compliance monitoring",
                "Establish incident response procedures",
                "Regular third-party compliance audits"
            ])
        
        return recommendations
    
    def _can_delete_record(self, record: DataProcessingRecord) -> bool:
        """检查是否可以删除记录"""
        # 检查法律义务
        if ProcessingPurpose.LEGAL_OBLIGATION in record.processing_purposes:
            return False
        
        # 检查合同义务
        if ProcessingPurpose.CONTRACT in record.processing_purposes:
            # 需要检查合同是否仍然有效
            return False
        
        return True
    
    def _is_portable_data(self, record: DataProcessingRecord) -> bool:
        """检查数据是否可携带"""
        # 基于同意或合同的数据通常是可携带的
        portable_purposes = [ProcessingPurpose.CONSENT, ProcessingPurpose.CONTRACT]
        return any(purpose in record.processing_purposes for purpose in portable_purposes)