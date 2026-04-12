# -*- coding: utf-8 -*-
"""
访问控制服务
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Set

from backend.modules.security.models import (
    Permission, Role, AccessPolicy, AccessRequest,
    AccessResult
)


class AccessControlManager:
    """访问控制管理器
    
    实现基于角色的访问控制(RBAC)和基于属性的访问控制(ABAC)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 角色权限映射
        self.role_permissions = self._init_role_permissions()
        
        # 用户角色映射
        self.user_roles: Dict[str, Set[Role]] = {}
        
        # 访问策略
        self.access_policies: Dict[str, AccessPolicy] = {}
        
        # 资源层次结构
        self.resource_hierarchy: Dict[str, List[str]] = {}
        
        # 初始化默认策略
        self._init_default_policies()
    
    def check_permission(self, user_id: str, resource: str, 
                        action: str, context: Optional[Dict[str, Any]] = None) -> AccessResult:
        """检查权限
        
        Args:
            user_id: 用户ID
            resource: 资源
            action: 操作
            context: 上下文信息
            
        Returns:
            访问结果
        """
        if context is None:
            context = {}
        
        request = AccessRequest(
            user_id=user_id,
            resource=resource,
            action=action,
            context=context,
            timestamp=datetime.now()
        )
        
        # 1. 检查RBAC权限
        rbac_result = self._check_rbac_permission(request)
        
        # 2. 检查ABAC策略
        abac_result = self._check_abac_policies(request)
        
        # 3. 合并结果
        final_result = self._merge_access_results(rbac_result, abac_result)
        
        # 4. 记录访问日志
        self._log_access_attempt(request, final_result)
        
        return final_result
    
    def assign_role(self, user_id: str, role: Role) -> bool:
        """分配角色
        
        Args:
            user_id: 用户ID
            role: 角色
            
        Returns:
            是否成功
        """
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        
        self.user_roles[user_id].add(role)
        return True
    
    def revoke_role(self, user_id: str, role: Role) -> bool:
        """撤销角色
        
        Args:
            user_id: 用户ID
            role: 角色
            
        Returns:
            是否成功
        """
        if user_id in self.user_roles and role in self.user_roles[user_id]:
            self.user_roles[user_id].remove(role)
            return True
        return False
    
    def get_user_roles(self, user_id: str) -> List[Role]:
        """获取用户角色
        
        Args:
            user_id: 用户ID
            
        Returns:
            角色列表
        """
        return list(self.user_roles.get(user_id, set()))
    
    def get_user_permissions(self, user_id: str) -> List[Permission]:
        """获取用户权限
        
        Args:
            user_id: 用户ID
            
        Returns:
            权限列表
        """
        permissions = set()
        user_roles = self.user_roles.get(user_id, set())
        
        for role in user_roles:
            role_perms = self.role_permissions.get(role, set())
            permissions.update(role_perms)
        
        return list(permissions)
    
    def create_policy(self, policy: AccessPolicy) -> bool:
        """创建访问策略
        
        Args:
            policy: 访问策略
            
        Returns:
            是否成功
        """
        try:
            self.access_policies[policy.id] = policy
            return True
        except Exception:
            return False
    
    def update_policy(self, policy_id: str, updates: Dict[str, Any]) -> bool:
        """更新访问策略
        
        Args:
            policy_id: 策略ID
            updates: 更新内容
            
        Returns:
            是否成功
        """
        if policy_id not in self.access_policies:
            return False
        
        try:
            policy = self.access_policies[policy_id]
            for key, value in updates.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
            policy.updated_at = datetime.now()
            return True
        except Exception:
            return False
    
    def delete_policy(self, policy_id: str) -> bool:
        """删除访问策略
        
        Args:
            policy_id: 策略ID
            
        Returns:
            是否成功
        """
        if policy_id in self.access_policies:
            del self.access_policies[policy_id]
            return True
        return False
    
    def get_policies(self) -> List[AccessPolicy]:
        """获取所有策略
        
        Returns:
            策略列表
        """
        return list(self.access_policies.values())
    
    def _init_role_permissions(self) -> Dict[Role, Set[Permission]]:
        """初始化角色权限映射"""
        return {
            Role.SUPER_ADMIN: {
                Permission.SYSTEM_ADMIN,
                Permission.USER_MANAGE,
                Permission.RESOURCE_MANAGE,
                Permission.AUDIT_READ,
                Permission.COST_MANAGE,
                Permission.BUDGET_MANAGE,
                Permission.TRAINING_CREATE,
                Permission.TRAINING_READ,
                Permission.TRAINING_UPDATE,
                Permission.TRAINING_DELETE,
                Permission.TRAINING_EXECUTE,
                Permission.TRAINING_STOP,
                Permission.MODEL_CREATE,
                Permission.MODEL_READ,
                Permission.MODEL_UPDATE,
                Permission.MODEL_DELETE,
                Permission.MODEL_DEPLOY,
                Permission.MODEL_DOWNLOAD,
                Permission.DATA_CREATE,
                Permission.DATA_READ,
                Permission.DATA_UPDATE,
                Permission.DATA_DELETE,
                Permission.DATA_UPLOAD,
            },
            Role.ADMIN: {
                Permission.USER_MANAGE,
                Permission.RESOURCE_MANAGE,
                Permission.AUDIT_READ,
                Permission.COST_READ,
                Permission.TRAINING_CREATE,
                Permission.TRAINING_READ,
                Permission.TRAINING_UPDATE,
                Permission.TRAINING_DELETE,
                Permission.TRAINING_EXECUTE,
                Permission.TRAINING_STOP,
                Permission.MODEL_CREATE,
                Permission.MODEL_READ,
                Permission.MODEL_UPDATE,
                Permission.MODEL_DELETE,
                Permission.MODEL_DEPLOY,
                Permission.MODEL_DOWNLOAD,
                Permission.DATA_CREATE,
                Permission.DATA_READ,
                Permission.DATA_UPDATE,
                Permission.DATA_DELETE,
                Permission.DATA_UPLOAD,
            },
            Role.MANAGER: {
                Permission.COST_READ,
                Permission.TRAINING_CREATE,
                Permission.TRAINING_READ,
                Permission.TRAINING_UPDATE,
                Permission.TRAINING_EXECUTE,
                Permission.TRAINING_STOP,
                Permission.MODEL_CREATE,
                Permission.MODEL_READ,
                Permission.MODEL_UPDATE,
                Permission.MODEL_DEPLOY,
                Permission.MODEL_DOWNLOAD,
                Permission.DATA_CREATE,
                Permission.DATA_READ,
                Permission.DATA_UPDATE,
                Permission.DATA_UPLOAD,
            },
            Role.DEVELOPER: {
                Permission.TRAINING_CREATE,
                Permission.TRAINING_READ,
                Permission.TRAINING_UPDATE,
                Permission.TRAINING_EXECUTE,
                Permission.MODEL_CREATE,
                Permission.MODEL_READ,
                Permission.MODEL_UPDATE,
                Permission.MODEL_DOWNLOAD,
                Permission.DATA_CREATE,
                Permission.DATA_READ,
                Permission.DATA_UPDATE,
                Permission.DATA_UPLOAD,
            },
            Role.ANALYST: {
                Permission.TRAINING_READ,
                Permission.MODEL_READ,
                Permission.MODEL_DOWNLOAD,
                Permission.DATA_READ,
                Permission.COST_READ,
            },
            Role.VIEWER: {
                Permission.TRAINING_READ,
                Permission.MODEL_READ,
                Permission.DATA_READ,
            },
            Role.GUEST: {
                Permission.TRAINING_READ,
                Permission.MODEL_READ,
            }
        }
    
    def _init_default_policies(self):
        """初始化默认策略"""
        # 资源所有者策略
        owner_policy = AccessPolicy(
            id="resource_owner",
            name="Resource Owner Policy",
            description="Resource owners have full access to their resources",
            effect="allow",
            principals=["*"],
            resources=["*"],
            actions=["*"],
            conditions={
                "resource_owner": "${user_id}"
            },
            priority=100,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self.access_policies[owner_policy.id] = owner_policy
        
        # 时间限制策略
        time_policy = AccessPolicy(
            id="business_hours",
            name="Business Hours Policy",
            description="Restrict access to business hours for certain operations",
            effect="deny",
            principals=["role:developer", "role:analyst"],
            resources=["training:*", "model:deploy"],
            actions=["create", "execute", "deploy"],
            conditions={
                "time_range": "22:00-06:00",
                "weekday": "saturday,sunday"
            },
            priority=200,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self.access_policies[time_policy.id] = time_policy
        
        # 成本限制策略
        cost_policy = AccessPolicy(
            id="cost_limit",
            name="Cost Limit Policy",
            description="Restrict expensive operations based on budget",
            effect="deny",
            principals=["role:developer"],
            resources=["training:*"],
            actions=["create", "execute"],
            conditions={
                "estimated_cost": ">1000",
                "budget_remaining": "<500"
            },
            priority=150,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self.access_policies[cost_policy.id] = cost_policy
        
        # Artifact 访问策略
        artifact_policy = AccessPolicy(
            id="artifact_access",
            name="Artifact Access Policy",
            description="Allow developers and managers to access artifacts",
            effect="allow",
            principals=["role:developer", "role:manager", "role:admin"],
            resources=["artifact:*"],
            actions=["create", "read", "update", "upload", "download"],
            conditions={},
            priority=120,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self.access_policies[artifact_policy.id] = artifact_policy
        
        # File 访问策略
        file_policy = AccessPolicy(
            id="file_access",
            name="File Access Policy",
            description="Allow developers and managers to access files",
            effect="allow",
            principals=["role:developer", "role:manager", "role:admin"],
            resources=["file:*"],
            actions=["create", "read", "update", "delete", "upload", "download"],
            conditions={},
            priority=120,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        self.access_policies[file_policy.id] = file_policy
    
    def _check_rbac_permission(self, request: AccessRequest) -> AccessResult:
        """检查RBAC权限"""
        user_roles = self.user_roles.get(request.user_id, set())
        user_permissions = set()
        
        for role in user_roles:
            role_perms = self.role_permissions.get(role, set())
            user_permissions.update(role_perms)
        
        # 检查是否有匹配的权限
        required_permission = self._get_required_permission(request.resource, request.action)
        
        if required_permission in user_permissions:
            return AccessResult(
                allowed=True,
                reason="RBAC permission granted",
                matched_policies=["rbac"],
                conditions_met=True,
                risk_score=0.0
            )
        else:
            return AccessResult(
                allowed=False,
                reason="RBAC permission denied",
                matched_policies=[],
                conditions_met=False,
                risk_score=0.5
            )
    
    def _check_abac_policies(self, request: AccessRequest) -> AccessResult:
        """检查ABAC策略"""
        matched_policies = []
        allow_policies = []
        deny_policies = []
        
        # 按优先级排序策略
        sorted_policies = sorted(
            self.access_policies.values(),
            key=lambda p: p.priority,
            reverse=True
        )
        
        for policy in sorted_policies:
            if self._policy_matches(policy, request):
                matched_policies.append(policy.id)
                
                if policy.effect == "allow":
                    allow_policies.append(policy)
                elif policy.effect == "deny":
                    deny_policies.append(policy)
        
        # 拒绝策略优先
        if deny_policies:
            return AccessResult(
                allowed=False,
                reason=f"Denied by policy: {deny_policies[0].name}",
                matched_policies=matched_policies,
                conditions_met=True,
                risk_score=0.8
            )
        
        # 检查允许策略
        if allow_policies:
            return AccessResult(
                allowed=True,
                reason=f"Allowed by policy: {allow_policies[0].name}",
                matched_policies=matched_policies,
                conditions_met=True,
                risk_score=0.2
            )
        
        # 默认拒绝
        return AccessResult(
            allowed=False,
            reason="No matching policy found",
            matched_policies=matched_policies,
            conditions_met=False,
            risk_score=0.6
        )
    
    def _policy_matches(self, policy: AccessPolicy, request: AccessRequest) -> bool:
        """检查策略是否匹配请求"""
        # 检查主体
        if not self._matches_principals(policy.principals, request.user_id):
            return False
        
        # 检查资源
        if not self._matches_resources(policy.resources, request.resource):
            return False
        
        # 检查操作
        if not self._matches_actions(policy.actions, request.action):
            return False
        
        # 检查条件
        if not self._matches_conditions(policy.conditions, request):
            return False
        
        return True
    
    def _matches_principals(self, principals: List[str], user_id: str) -> bool:
        """检查主体是否匹配"""
        for principal in principals:
            if principal == "*":
                return True
            elif principal == user_id:
                return True
            elif principal.startswith("role:"):
                role_name = principal[5:]
                try:
                    role = Role(role_name)
                    if role in self.user_roles.get(user_id, set()):
                        return True
                except ValueError:
                    continue
        return False
    
    def _matches_resources(self, resources: List[str], resource: str) -> bool:
        """检查资源是否匹配"""
        for resource_pattern in resources:
            if resource_pattern == "*":
                return True
            elif self._pattern_matches(resource_pattern, resource):
                return True
        return False
    
    def _matches_actions(self, actions: List[str], action: str) -> bool:
        """检查操作是否匹配"""
        return "*" in actions or action in actions
    
    def _matches_conditions(self, conditions: Dict[str, Any], request: AccessRequest) -> bool:
        """检查条件是否匹配"""
        for condition_key, condition_value in conditions.items():
            if not self._evaluate_condition(condition_key, condition_value, request):
                return False
        return True
    
    def _evaluate_condition(self, key: str, value: Any, request: AccessRequest) -> bool:
        """评估单个条件"""
        context = request.context
        
        if key == "resource_owner":
            resource_owner = context.get("resource_owner")
            if value == "${user_id}":
                return resource_owner == request.user_id
            return resource_owner == value
        
        elif key == "time_range":
            current_time = request.timestamp.strftime("%H:%M")
            start_time, end_time = value.split("-")
            return start_time <= current_time <= end_time
        
        elif key == "weekday":
            current_weekday = request.timestamp.strftime("%A").lower()
            allowed_weekdays = [day.strip() for day in value.split(",")]
            return current_weekday not in allowed_weekdays
        
        elif key == "estimated_cost":
            estimated_cost = context.get("estimated_cost", 0)
            if value.startswith(">"):
                threshold = float(value[1:])
                return estimated_cost > threshold
            elif value.startswith("<"):
                threshold = float(value[1:])
                return estimated_cost < threshold
        
        elif key == "budget_remaining":
            budget_remaining = context.get("budget_remaining", float('inf'))
            if value.startswith("<"):
                threshold = float(value[1:])
                return budget_remaining < threshold
        
        return True
    
    def _pattern_matches(self, pattern: str, text: str) -> bool:
        """检查模式是否匹配文本"""
        # 将通配符模式转换为正则表达式
        regex_pattern = pattern.replace("*", ".*").replace("?", ".")
        return re.match(f"^{regex_pattern}$", text) is not None
    
    def _get_required_permission(self, resource: str, action: str) -> Optional[Permission]:
        """获取所需权限"""
        resource_type = resource.split(":")[0] if ":" in resource else resource
        
        permission_map = {
            ("training", "create"): Permission.TRAINING_CREATE,
            ("training", "read"): Permission.TRAINING_READ,
            ("training", "update"): Permission.TRAINING_UPDATE,
            ("training", "delete"): Permission.TRAINING_DELETE,
            ("training", "execute"): Permission.TRAINING_EXECUTE,
            ("training", "stop"): Permission.TRAINING_STOP,
            ("model", "create"): Permission.MODEL_CREATE,
            ("model", "read"): Permission.MODEL_READ,
            ("model", "update"): Permission.MODEL_UPDATE,
            ("model", "delete"): Permission.MODEL_DELETE,
            ("model", "deploy"): Permission.MODEL_DEPLOY,
            ("model", "download"): Permission.MODEL_DOWNLOAD,
            ("data", "create"): Permission.DATA_CREATE,
            ("data", "read"): Permission.DATA_READ,
            ("data", "update"): Permission.DATA_UPDATE,
            ("data", "delete"): Permission.DATA_DELETE,
            ("data", "upload"): Permission.DATA_UPLOAD,
            # 工件相关权限映射
            ("artifact", "create"): Permission.DATA_CREATE,
            ("artifact", "read"): Permission.DATA_READ,
            ("artifact", "update"): Permission.DATA_UPDATE,
            ("artifact", "delete"): Permission.DATA_DELETE,
            ("artifact", "upload"): Permission.DATA_UPLOAD,
            ("artifact", "download"): Permission.DATA_READ,
            # 文件相关权限映射
            ("file", "create"): Permission.DATA_CREATE,
            ("file", "read"): Permission.DATA_READ,
            ("file", "update"): Permission.DATA_UPDATE,
            ("file", "delete"): Permission.DATA_DELETE,
            ("file", "upload"): Permission.DATA_UPLOAD,
            ("file", "download"): Permission.DATA_READ,
        }
        
        return permission_map.get((resource_type, action))
    
    def _merge_access_results(self, rbac_result: AccessResult, 
                             abac_result: AccessResult) -> AccessResult:
        """合并访问结果"""
        # ABAC策略优先级更高
        if not abac_result.allowed:
            return abac_result
        
        # 如果ABAC允许，检查RBAC
        if not rbac_result.allowed:
            return AccessResult(
                allowed=False,
                reason="RBAC permission required",
                matched_policies=rbac_result.matched_policies + abac_result.matched_policies,
                conditions_met=abac_result.conditions_met,
                risk_score=max(rbac_result.risk_score, abac_result.risk_score)
            )
        
        # 两者都允许
        return AccessResult(
            allowed=True,
            reason="Both RBAC and ABAC allow access",
            matched_policies=rbac_result.matched_policies + abac_result.matched_policies,
            conditions_met=True,
            risk_score=min(rbac_result.risk_score, abac_result.risk_score)
        )
    
    def _log_access_attempt(self, request: AccessRequest, result: AccessResult):
        """记录访问尝试"""
        # 这里应该记录到审计日志
        log_entry = {
            'timestamp': request.timestamp.isoformat(),
            'user_id': request.user_id,
            'resource': request.resource,
            'action': request.action,
            'allowed': result.allowed,
            'reason': result.reason,
            'matched_policies': result.matched_policies,
            'risk_score': result.risk_score,
            'context': request.context
        }
        
        # 暂时打印到控制台，实际应该写入日志系统
        print(f"Access Log: {json.dumps(log_entry, indent=2)}")