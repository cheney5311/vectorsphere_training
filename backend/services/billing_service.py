"""计费服务

实现计费相关的业务逻辑，支持数据库持久化。
"""

import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any, Tuple

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

try:
    from backend.core.exceptions import ValidationError, BusinessLogicError
except ImportError:
    from backend.core.exceptions import ValidationError, BusinessLogicError

logger = logging.getLogger(__name__)


# ============================================================================
# 计费枚举（兼容旧代码）
# ============================================================================

class BillingPeriod:
    """计费周期"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class InvoiceStatus:
    """发票状态"""
    DRAFT = "draft"
    PENDING = "pending"
    SENT = "sent"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


class PaymentStatus:
    """支付状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    PAID = "paid"
    FAILED = "failed"
    CANCELLED = "cancelled"
    OVERDUE = "overdue"


# ============================================================================
# 数据类定义（兼容旧代码，用于没有数据库时的内存存储）
# ============================================================================

from dataclasses import dataclass, field


@dataclass
class BillingRule:
    """计费规则数据类"""
    id: str
    name: str
    resource_type: str
    billing_model: str = "pay_as_you_go"
    billing_period: str = BillingPeriod.HOURLY
    unit_price: Decimal = Decimal('0')
    unit: str = "unit"
    currency: str = "CNY"
    free_tier: Decimal = Decimal('0')
    minimum_charge: Optional[Decimal] = None
    discount_rate: Decimal = Decimal('0')
    is_active: bool = True
    effective_date: datetime = field(default_factory=datetime.utcnow)
    expiry_date: Optional[datetime] = None


@dataclass
class UsageRecord:
    """使用记录数据类"""
    id: str
    user_id: str
    resource_type: str
    resource_id: str
    usage_amount: Decimal
    unit: str
    start_time: datetime
    end_time: datetime
    tenant_id: str = ""
    cost: Decimal = Decimal('0')
    billing_rule_id: str = ""
    billing_status: str = "pending"
    invoice_id: str = ""


@dataclass
class BillingItem:
    """计费项目数据类"""
    id: str
    invoice_id: str
    usage_record_id: str
    description: str
    quantity: Decimal
    unit_price: Decimal
    total_amount: Decimal
    unit: str = "unit"
    tax_amount: Decimal = Decimal('0')


@dataclass
class Invoice:
    """发票数据类"""
    id: str
    user_id: str
    billing_period_start: datetime
    billing_period_end: datetime
    invoice_number: str = ""
    tenant_id: str = ""
    subtotal: Decimal = Decimal('0')
    discount_amount: Decimal = Decimal('0')
    tax_amount: Decimal = Decimal('0')
    total_amount: Decimal = Decimal('0')
    currency: str = "CNY"
    status: str = InvoiceStatus.DRAFT
    payment_status: str = PaymentStatus.PENDING
    due_date: Optional[datetime] = None
    items: List[BillingItem] = field(default_factory=list)
    
    def calculate_totals(self):
        """计算发票总额"""
        self.subtotal = sum(item.total_amount for item in self.items)
        self.tax_amount = sum(item.tax_amount for item in self.items)
        self.total_amount = self.subtotal + self.tax_amount - self.discount_amount


@dataclass
class Payment:
    """支付记录数据类"""
    id: str
    invoice_id: str
    user_id: str
    amount: Decimal
    payment_method: str
    tenant_id: str = ""
    currency: str = "CNY"
    transaction_id: str = ""
    status: str = PaymentStatus.PENDING
    payment_date: Optional[datetime] = None


# ============================================================================
# 计费计算器
# ============================================================================

class BillingCalculator:
    """计费计算器"""
    
    def __init__(self):
        self.tax_rate = Decimal('0.1')  # 默认税率10%
    
    def calculate_cost(self, usage_amount: Decimal, unit_price: Decimal,
                      free_tier: Decimal = Decimal('0'),
                      discount_rate: Decimal = Decimal('0'),
                      minimum_charge: Optional[Decimal] = None) -> Decimal:
        """计算使用成本"""
        try:
            # 基础成本计算
            base_cost = usage_amount * unit_price
            
            # 应用免费额度
            if free_tier and usage_amount <= free_tier:
                return Decimal('0')
            elif free_tier:
                billable_amount = usage_amount - free_tier
                base_cost = billable_amount * unit_price
            
            # 应用折扣
            if discount_rate > 0:
                base_cost = base_cost * (Decimal('1') - discount_rate)
            
            # 应用最低收费
            if minimum_charge and base_cost < minimum_charge:
                base_cost = minimum_charge
            
            return base_cost.quantize(Decimal('0.01'))  # 保留两位小数
            
        except Exception as e:
            logger.error(f"Failed to calculate cost: {e}")
            return Decimal('0')
    
    def calculate_cost_from_rule(self, usage_amount: Decimal, billing_rule: Dict) -> Decimal:
        """根据计费规则计算成本"""
        unit_price = Decimal(str(billing_rule.get('unit_price', 0)))
        free_tier = Decimal(str(billing_rule.get('free_tier', 0)))
        discount_rate = Decimal(str(billing_rule.get('discount_rate', 0)))
        minimum_charge = billing_rule.get('minimum_charge')
        if minimum_charge:
            minimum_charge = Decimal(str(minimum_charge))
        
        return self.calculate_cost(usage_amount, unit_price, free_tier, discount_rate, minimum_charge)
    
    def calculate_tax(self, amount: Decimal, tax_rate: Optional[Decimal] = None) -> Decimal:
        """计算税费"""
        rate = tax_rate or self.tax_rate
        return (amount * rate).quantize(Decimal('0.01'))
    
    def apply_discount(self, amount: Decimal, discount_rate: Decimal) -> Decimal:
        """应用折扣"""
        return (amount * discount_rate).quantize(Decimal('0.01'))


# ============================================================================
# 计费服务
# ============================================================================

class BillingService:
    """计费服务
    
    支持两种模式：
    1. 内存模式 (use_memory_storage=True): 数据存储在内存中，适用于测试
    2. 数据库模式 (use_memory_storage=False): 数据持久化到数据库
    """
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化计费服务
        
        Args:
            use_memory_storage: 是否使用内存存储
        """
        self._use_memory_storage = use_memory_storage
        self.calculator = BillingCalculator()
        
        # 初始化 Repository
        self._init_repositories()
        
        # 初始化默认规则
        self._initialize_default_rules()
    
    def _init_repositories(self):
        """初始化数据访问层"""
        try:
            from backend.repositories.billing_repository import (
                BillingRuleRepository,
                UsageRecordRepository,
                InvoiceRepository,
                BillingItemRepository,
                PaymentRepository,
                WalletRepository,
                WalletTransactionRepository
            )
            
            self.billing_rule_repo = BillingRuleRepository(use_memory_storage=self._use_memory_storage)
            self.usage_record_repo = UsageRecordRepository(use_memory_storage=self._use_memory_storage)
            self.invoice_repo = InvoiceRepository(use_memory_storage=self._use_memory_storage)
            self.billing_item_repo = BillingItemRepository(use_memory_storage=self._use_memory_storage)
            self.payment_repo = PaymentRepository(use_memory_storage=self._use_memory_storage)
            self.wallet_repo = WalletRepository(use_memory_storage=self._use_memory_storage)
            self.wallet_transaction_repo = WalletTransactionRepository(use_memory_storage=self._use_memory_storage)
            
            logger.info(f"BillingService repositories initialized (memory_storage={self._use_memory_storage})")
            
        except ImportError as e:
            logger.warning(f"Failed to import billing repositories: {e}, falling back to legacy mode")
            # 回退到旧的内存存储模式
            self._use_memory_storage = True
            self.billing_rules: Dict[str, BillingRule] = {}
            self.usage_records: Dict[str, UsageRecord] = {}
            self.invoices: Dict[str, Invoice] = {}
            self.payments: Dict[str, Payment] = {}
            
            self.billing_rule_repo = None
            self.usage_record_repo = None
            self.invoice_repo = None
            self.billing_item_repo = None
            self.payment_repo = None
            self.wallet_repo = None
            self.wallet_transaction_repo = None
    
    def _initialize_default_rules(self):
        """初始化默认计费规则"""
        default_rules = [
            {
                'id': 'gpu_hourly',
                'name': 'GPU按小时计费',
                'resource_type': 'gpu',
                'billing_model': 'pay_as_you_go',
                'billing_period': BillingPeriod.HOURLY,
                'unit_price': Decimal('2.50'),
                'unit': 'hour',
                'free_tier': Decimal('1')
            },
            {
                'id': 'cpu_hourly',
                'name': 'CPU按小时计费',
                'resource_type': 'cpu',
                'billing_model': 'pay_as_you_go',
                'billing_period': BillingPeriod.HOURLY,
                'unit_price': Decimal('0.10'),
                'unit': 'hour'
            },
            {
                'id': 'memory_hourly',
                'name': '内存按小时计费',
                'resource_type': 'memory',
                'billing_model': 'pay_as_you_go',
                'billing_period': BillingPeriod.HOURLY,
                'unit_price': Decimal('0.02'),
                'unit': 'GB-hour'
            },
            {
                'id': 'storage_monthly',
                'name': '存储按月计费',
                'resource_type': 'storage',
                'billing_model': 'subscription',
                'billing_period': BillingPeriod.MONTHLY,
                'unit_price': Decimal('0.05'),
                'unit': 'GB-month'
            },
            {
                'id': 'api_calls',
                'name': 'API调用计费',
                'resource_type': 'api_calls',
                'billing_model': 'pay_as_you_go',
                'billing_period': BillingPeriod.MONTHLY,
                'unit_price': Decimal('0.001'),
                'unit': 'call',
                'free_tier': Decimal('1000')
            },
            {
                'id': 'training_hours',
                'name': '训练时长计费',
                'resource_type': 'training_hours',
                'billing_model': 'pay_as_you_go',
                'billing_period': BillingPeriod.HOURLY,
                'unit_price': Decimal('5.00'),
                'unit': 'hour'
            }
        ]
        
        for rule_data in default_rules:
            try:
                if self.billing_rule_repo:
                    existing = self.billing_rule_repo.get_by_id(rule_data['id'])
                    if not existing:
                        self.billing_rule_repo.create(rule_data)
                else:
                    # 旧模式
                    self.billing_rules[rule_data['id']] = BillingRule(**rule_data)
            except Exception as e:
                logger.debug(f"Default rule {rule_data['id']} may already exist: {e}")
    
    # ========== 计费规则管理 ==========
    
    def add_billing_rule(self, rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """添加计费规则"""
        try:
            if self.billing_rule_repo:
                rule = self.billing_rule_repo.create(rule_data)
                logger.info(f"Added billing rule: {rule_data.get('name')}")
                return self._rule_to_dict(rule)
            else:
                rule = BillingRule(**rule_data)
                self.billing_rules[rule.id] = rule
                return self._rule_to_dict(rule)
                
        except Exception as e:
            logger.error(f"Failed to add billing rule: {e}")
            raise BusinessLogicError(f"Failed to add billing rule: {e}")
    
    def update_billing_rule(self, rule_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新计费规则"""
        try:
            if self.billing_rule_repo:
                rule = self.billing_rule_repo.update(rule_id, updates)
                if rule:
                    logger.info(f"Updated billing rule: {rule_id}")
                    return self._rule_to_dict(rule)
                return None
            else:
                if rule_id not in self.billing_rules:
                    return None
                rule = self.billing_rules[rule_id]
                for key, value in updates.items():
                    if hasattr(rule, key):
                        setattr(rule, key, value)
                return self._rule_to_dict(rule)
                
        except Exception as e:
            logger.error(f"Failed to update billing rule: {e}")
            raise BusinessLogicError(f"Failed to update billing rule: {e}")
    
    def get_billing_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """获取计费规则"""
        try:
            if self.billing_rule_repo:
                rule = self.billing_rule_repo.get_by_id(rule_id)
                return self._rule_to_dict(rule) if rule else None
            else:
                rule = self.billing_rules.get(rule_id)
                return self._rule_to_dict(rule) if rule else None
                
        except Exception as e:
            logger.error(f"Failed to get billing rule: {e}")
            return None
    
    def list_billing_rules(self, active_only: bool = True, 
                          limit: int = 100, offset: int = 0) -> Tuple[List[Dict], int]:
        """获取计费规则列表"""
        try:
            if self.billing_rule_repo:
                rules, total = self.billing_rule_repo.list_all(active_only, limit, offset)
                return [self._rule_to_dict(r) for r in rules], total
            else:
                rules = list(self.billing_rules.values())
                if active_only:
                    rules = [r for r in rules if r.is_active]
                total = len(rules)
                return [self._rule_to_dict(r) for r in rules[offset:offset+limit]], total
                
        except Exception as e:
            logger.error(f"Failed to list billing rules: {e}")
            return [], 0
    
    def delete_billing_rule(self, rule_id: str) -> bool:
        """删除计费规则"""
        try:
            if self.billing_rule_repo:
                return self.billing_rule_repo.delete(rule_id)
            else:
                if rule_id in self.billing_rules:
                    del self.billing_rules[rule_id]
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete billing rule: {e}")
            return False
    
    # ========== 使用记录管理 ==========
    
    def record_usage(self, usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """记录资源使用"""
        try:
            # 查找适用的计费规则
            resource_type = usage_data.get('resource_type')
            billing_rule = self._find_applicable_rule(resource_type)
            
            if not billing_rule:
                logger.warning(f"No applicable billing rule found for: {resource_type}")
                # 仍然记录使用，但不计费
                usage_data['billing_status'] = 'no_rule'
            else:
            # 计算成本
                usage_amount = Decimal(str(usage_data.get('usage_amount', 0)))
                cost = self.calculator.calculate_cost_from_rule(usage_amount, billing_rule)
                usage_data['cost'] = cost
                usage_data['billing_rule_id'] = billing_rule.get('id')
                usage_data['billing_status'] = 'pending'
            
            # 保存使用记录
            if self.usage_record_repo:
                record = self.usage_record_repo.create(usage_data)
                logger.info(f"Recorded usage: {record.get('id') if isinstance(record, dict) else record.id}, cost: {usage_data.get('cost', 0)}")
                return self._record_to_dict(record)
            else:
                record_id = usage_data.get('id') or str(uuid.uuid4())
                usage_data['id'] = record_id
                record = UsageRecord(**usage_data)
                self.usage_records[record_id] = record
                return self._record_to_dict(record)
            
        except Exception as e:
            logger.error(f"Failed to record usage: {e}")
            raise BusinessLogicError(f"Failed to record usage: {e}")
    
    def get_usage_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """获取使用记录"""
        try:
            if self.usage_record_repo:
                record = self.usage_record_repo.get_by_id(record_id)
                return self._record_to_dict(record) if record else None
            else:
                record = self.usage_records.get(record_id)
                return self._record_to_dict(record) if record else None
                
        except Exception as e:
            logger.error(f"Failed to get usage record: {e}")
        return None
    
    def get_user_usage_records(self, user_id: str, 
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None,
                               resource_type: Optional[str] = None,
                               limit: int = 100, offset: int = 0) -> Tuple[List[Dict], int]:
        """获取用户的使用记录"""
        try:
            if self.usage_record_repo:
                records, total = self.usage_record_repo.get_by_user(
                    user_id, start_time, end_time, resource_type, limit, offset
                )
                return [self._record_to_dict(r) for r in records], total
            else:
                records = [r for r in self.usage_records.values() if r.user_id == user_id]
                if start_time:
                    records = [r for r in records if r.start_time >= start_time]
                if end_time:
                    records = [r for r in records if r.end_time <= end_time]
                if resource_type:
                    records = [r for r in records if r.resource_type == resource_type]
                total = len(records)
                return [self._record_to_dict(r) for r in records[offset:offset+limit]], total
                
        except Exception as e:
            logger.error(f"Failed to get user usage records: {e}")
            return [], 0
    
    def get_usage_summary(self, user_id: str, start_time: datetime, 
                         end_time: datetime) -> Dict[str, Any]:
        """获取用户使用汇总"""
        try:
            if self.usage_record_repo:
                return self.usage_record_repo.get_usage_summary(user_id, start_time, end_time)
            else:
                records = [
                    r for r in self.usage_records.values()
                    if r.user_id == user_id
                    and r.start_time >= start_time
                    and r.end_time <= end_time
                ]
                
                summary = {}
                total_cost = Decimal('0')
                
                for record in records:
                    resource_type = record.resource_type
                    if resource_type not in summary:
                        summary[resource_type] = {
                            'usage_amount': Decimal('0'),
                            'cost': Decimal('0'),
                            'unit': record.unit,
                            'record_count': 0
                        }
                    
                    summary[resource_type]['usage_amount'] += record.usage_amount
                    summary[resource_type]['cost'] += record.cost
                    summary[resource_type]['record_count'] += 1
                    total_cost += record.cost
                
                return {
                    'user_id': user_id,
                    'period_start': start_time.isoformat(),
                    'period_end': end_time.isoformat(),
                    'total_cost': float(total_cost),
                    'by_resource_type': {
                        k: {
                            'usage_amount': float(v['usage_amount']),
                            'cost': float(v['cost']),
                            'unit': v['unit'],
                            'record_count': v['record_count']
                        } for k, v in summary.items()
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get usage summary: {e}")
            return {}
    
    # ========== 发票管理 ==========
    
    def generate_invoice(self, user_id: str, period_start: datetime, 
                        period_end: datetime, tenant_id: str = None) -> Optional[Dict[str, Any]]:
        """生成发票"""
        try:
            # 获取未计费的使用记录
            if self.usage_record_repo:
                unbilled_records = self.usage_record_repo.get_unbilled_records(
                    user_id, period_start, period_end
                )
            else:
                unbilled_records = [
                    r for r in self.usage_records.values()
                    if r.user_id == user_id
                    and r.billing_status == 'pending'
                    and r.start_time >= period_start
                    and r.end_time <= period_end
                ]
            
            if not unbilled_records:
                logger.info(f"No unbilled records for user {user_id} in the specified period")
                return None
            
            # 创建发票
            invoice_data = {
                'user_id': user_id,
                'tenant_id': tenant_id,
                'billing_period_start': period_start,
                'billing_period_end': period_end,
                'due_date': period_end + timedelta(days=30),
                'status': InvoiceStatus.PENDING,
                'payment_status': PaymentStatus.PENDING
            }
            
            if self.invoice_repo:
                invoice = self.invoice_repo.create(invoice_data)
                invoice_id = invoice.get('id') if isinstance(invoice, dict) else invoice.id
            else:
                invoice_id = f"inv_{user_id}_{int(period_start.timestamp())}"
                invoice_data['id'] = invoice_id
                invoice_data['invoice_number'] = invoice_id
                invoice = Invoice(**invoice_data)
                self.invoices[invoice_id] = invoice
            
            # 创建计费项目
            subtotal = Decimal('0')
            tax_total = Decimal('0')
            record_ids = []
            
            for record in unbilled_records:
                record_dict = self._record_to_dict(record)
                record_id = record_dict.get('id')
                record_ids.append(record_id)
                
                billing_rule = self._find_rule_by_id(record_dict.get('billing_rule_id'))
                rule_name = billing_rule.get('name', 'Unknown') if billing_rule else 'Unknown'
                
                cost = Decimal(str(record_dict.get('cost', 0)))
                tax_amount = self.calculator.calculate_tax(cost)
                
                item_data = {
                    'invoice_id': invoice_id,
                    'usage_record_id': record_id,
                    'description': f"{rule_name} - {record_dict.get('resource_id', '')}",
                    'resource_type': record_dict.get('resource_type'),
                    'quantity': Decimal(str(record_dict.get('usage_amount', 0))),
                    'unit': record_dict.get('unit', 'unit'),
                    'unit_price': Decimal(str(billing_rule.get('unit_price', 0))) if billing_rule else Decimal('0'),
                    'subtotal': cost,
                    'tax_amount': tax_amount,
                    'total_amount': cost + tax_amount,
                    'period_start': record_dict.get('start_time'),
                    'period_end': record_dict.get('end_time')
                }
                
                if self.billing_item_repo:
                    self.billing_item_repo.create(item_data)
                else:
                    item = BillingItem(**{
                        'id': f"item_{record_id}",
                        **item_data
                    })
                    invoice.items.append(item)
                
                subtotal += cost
                tax_total += tax_amount
            
            # 更新发票总额
            total_amount = subtotal + tax_total
            update_data = {
                'subtotal': subtotal,
                'tax_amount': tax_total,
                'total_amount': total_amount
            }
            
            if self.invoice_repo:
                invoice = self.invoice_repo.update(invoice_id, update_data)
                # 标记使用记录为已计费
                if self.usage_record_repo:
                    self.usage_record_repo.mark_as_billed(record_ids, invoice_id)
            else:
                invoice.subtotal = subtotal
                invoice.tax_amount = tax_total
                invoice.total_amount = total_amount
                for rid in record_ids:
                    if rid in self.usage_records:
                        self.usage_records[rid].billing_status = 'billed'
                        self.usage_records[rid].invoice_id = invoice_id
            
            logger.info(f"Generated invoice: {invoice_id}, total: {total_amount}")
            return self._invoice_to_dict(invoice)
            
        except Exception as e:
            logger.error(f"Failed to generate invoice: {e}")
            raise BusinessLogicError(f"Failed to generate invoice: {e}")
    
    def get_invoice(self, invoice_id: str) -> Optional[Dict[str, Any]]:
        """获取发票"""
        try:
            if self.invoice_repo:
                invoice = self.invoice_repo.get_by_id(invoice_id)
                if invoice:
                    invoice_dict = self._invoice_to_dict(invoice)
                    # 获取发票项目
                    if self.billing_item_repo:
                        items = self.billing_item_repo.get_by_invoice(invoice_id)
                        invoice_dict['items'] = [self._item_to_dict(i) for i in items]
                    return invoice_dict
                return None
            else:
                invoice = self.invoices.get(invoice_id)
                return self._invoice_to_dict(invoice) if invoice else None
                
        except Exception as e:
            logger.error(f"Failed to get invoice: {e}")
            return None
    
    def get_user_invoices(self, user_id: str, status: Optional[str] = None,
                         payment_status: Optional[str] = None,
                         limit: int = 100, offset: int = 0) -> Tuple[List[Dict], int]:
        """获取用户的发票列表"""
        try:
            if self.invoice_repo:
                invoices, total = self.invoice_repo.get_by_user(
                    user_id, status, payment_status, limit, offset
                )
                return [self._invoice_to_dict(i) for i in invoices], total
            else:
                invoices = [i for i in self.invoices.values() if i.user_id == user_id]
                if status:
                    invoices = [i for i in invoices if i.status == status]
                if payment_status:
                    invoices = [i for i in invoices if i.payment_status == payment_status]
                total = len(invoices)
                return [self._invoice_to_dict(i) for i in invoices[offset:offset+limit]], total
                
        except Exception as e:
            logger.error(f"Failed to get user invoices: {e}")
            return [], 0
    
    def get_overdue_invoices(self) -> List[Dict]:
        """获取逾期发票"""
        try:
            if self.invoice_repo:
                invoices = self.invoice_repo.get_overdue()
                # 更新状态为逾期
                for inv in invoices:
                    inv_id = inv.get('id') if isinstance(inv, dict) else inv.id
                    self.invoice_repo.update(inv_id, {
                        'status': InvoiceStatus.OVERDUE,
                        'payment_status': PaymentStatus.OVERDUE
                    })
                return [self._invoice_to_dict(i) for i in invoices]
            else:
                now = datetime.utcnow()
                overdue = []
                for invoice in self.invoices.values():
                    if (invoice.due_date and invoice.due_date < now 
                        and invoice.payment_status != PaymentStatus.PAID):
                        invoice.status = InvoiceStatus.OVERDUE
                        invoice.payment_status = PaymentStatus.OVERDUE
                        overdue.append(invoice)
                return [self._invoice_to_dict(i) for i in overdue]
                
        except Exception as e:
            logger.error(f"Failed to get overdue invoices: {e}")
            return []
    
    # ========== 支付管理 ==========
    
    def process_payment(self, payment_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理支付"""
        try:
            invoice_id = payment_data.get('invoice_id')
            
            # 验证发票
            invoice = self.get_invoice(invoice_id)
            if not invoice:
                raise ValidationError(f"Invoice not found: {invoice_id}")
            
            # 验证支付金额
            payment_amount = Decimal(str(payment_data.get('amount', 0)))
            invoice_amount = Decimal(str(invoice.get('total_amount', 0)))
            
            if payment_amount != invoice_amount:
                raise ValidationError(f"Payment amount mismatch: {payment_amount} != {invoice_amount}")
            
            # 创建支付记录
            payment_data['status'] = PaymentStatus.PROCESSING
            
            if self.payment_repo:
                payment = self.payment_repo.create(payment_data)
                payment_id = payment.get('id') if isinstance(payment, dict) else payment.id
                
                # 模拟支付处理（实际应用中对接支付网关）
                # 假设支付成功
                self.payment_repo.mark_as_paid(payment_id, payment_data.get('transaction_id'))
                payment = self.payment_repo.get_by_id(payment_id)
                
                # 更新发票状态
                self.invoice_repo.mark_as_paid(invoice_id)
            else:
                payment_id = payment_data.get('id') or str(uuid.uuid4())
                payment_data['id'] = payment_id
                payment_data['status'] = PaymentStatus.PAID
                payment_data['payment_date'] = datetime.utcnow()
                payment = Payment(**payment_data)
                self.payments[payment_id] = payment
            
            # 更新发票状态
                if invoice_id in self.invoices:
                    self.invoices[invoice_id].status = InvoiceStatus.PAID
                    self.invoices[invoice_id].payment_status = PaymentStatus.PAID
            
            logger.info(f"Processed payment: {payment_id}")
            return self._payment_to_dict(payment)
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to process payment: {e}")
            raise BusinessLogicError(f"Failed to process payment: {e}")
    
    def get_payment(self, payment_id: str) -> Optional[Dict[str, Any]]:
        """获取支付记录"""
        try:
            if self.payment_repo:
                payment = self.payment_repo.get_by_id(payment_id)
                return self._payment_to_dict(payment) if payment else None
            else:
                payment = self.payments.get(payment_id)
                return self._payment_to_dict(payment) if payment else None
                
        except Exception as e:
            logger.error(f"Failed to get payment: {e}")
            return None
    
    def get_user_payments(self, user_id: str, status: Optional[str] = None,
                         limit: int = 100, offset: int = 0) -> Tuple[List[Dict], int]:
        """获取用户的支付记录"""
        try:
            if self.payment_repo:
                payments, total = self.payment_repo.get_by_user(user_id, status, limit, offset)
                return [self._payment_to_dict(p) for p in payments], total
            else:
                payments = [p for p in self.payments.values() if p.user_id == user_id]
                if status:
                    payments = [p for p in payments if p.status == status]
                total = len(payments)
                return [self._payment_to_dict(p) for p in payments[offset:offset+limit]], total
                
        except Exception as e:
            logger.error(f"Failed to get user payments: {e}")
            return [], 0
    
    # ========== 钱包管理 ==========
    
    def get_or_create_wallet(self, user_id: str, tenant_id: str = None) -> Dict[str, Any]:
        """获取或创建用户钱包"""
        try:
            if self.wallet_repo:
                wallet = self.wallet_repo.get_or_create(user_id, tenant_id)
                return self._wallet_to_dict(wallet)
            else:
                return {
                    'user_id': user_id,
                    'balance': 0,
                    'frozen_balance': 0,
                    'currency': 'CNY',
                    'status': 'active'
                }
                
        except Exception as e:
            logger.error(f"Failed to get or create wallet: {e}")
            raise BusinessLogicError(f"Failed to get or create wallet: {e}")
    
    def add_funds(self, user_id: str, amount: Decimal, description: str = None,
                 reference_type: str = None, reference_id: str = None) -> Dict[str, Any]:
        """充值"""
        try:
            if self.wallet_repo and self.wallet_transaction_repo:
                wallet = self.wallet_repo.get_by_user(user_id)
                if not wallet:
                    wallet = self.wallet_repo.create({'user_id': user_id})
                
                wallet_id = wallet.get('id') if isinstance(wallet, dict) else wallet.id
                balance_before = Decimal(str(wallet.get('balance') if isinstance(wallet, dict) else wallet.balance))
                balance_after = balance_before + amount
                
                # 更新钱包余额
                self.wallet_repo.update_balance(user_id, amount, 'add')
                
                # 创建交易记录
                transaction_data = {
                    'wallet_id': wallet_id,
                    'user_id': user_id,
                    'transaction_type': 'deposit',
                    'amount': amount,
                    'balance_before': balance_before,
                    'balance_after': balance_after,
                    'description': description or '账户充值',
                    'reference_type': reference_type,
                    'reference_id': reference_id
                }
                self.wallet_transaction_repo.create(transaction_data)
                
                return self._wallet_to_dict(self.wallet_repo.get_by_user(user_id))
            else:
                return {
                    'user_id': user_id,
                    'balance': float(amount),
                    'currency': 'CNY',
                    'status': 'active'
                }
                
        except Exception as e:
            logger.error(f"Failed to add funds: {e}")
            raise BusinessLogicError(f"Failed to add funds: {e}")
    
    def deduct_funds(self, user_id: str, amount: Decimal, description: str = None,
                    reference_type: str = None, reference_id: str = None) -> Dict[str, Any]:
        """扣款"""
        try:
            if self.wallet_repo and self.wallet_transaction_repo:
                wallet = self.wallet_repo.get_by_user(user_id)
                if not wallet:
                    raise ValidationError("Wallet not found")
                
                balance = Decimal(str(wallet.get('balance') if isinstance(wallet, dict) else wallet.balance))
                if balance < amount:
                    raise ValidationError("Insufficient balance")
                
                wallet_id = wallet.get('id') if isinstance(wallet, dict) else wallet.id
                balance_before = balance
                balance_after = balance_before - amount
                
                # 更新钱包余额
                self.wallet_repo.update_balance(user_id, amount, 'subtract')
                
                # 创建交易记录
                transaction_data = {
                    'wallet_id': wallet_id,
                    'user_id': user_id,
                    'transaction_type': 'deduction',
                    'amount': amount,
                    'balance_before': balance_before,
                    'balance_after': balance_after,
                    'description': description or '账户扣款',
                    'reference_type': reference_type,
                    'reference_id': reference_id
                }
                self.wallet_transaction_repo.create(transaction_data)
                
                return self._wallet_to_dict(self.wallet_repo.get_by_user(user_id))
            else:
                raise ValidationError("Wallet functionality not available")
                
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to deduct funds: {e}")
            raise BusinessLogicError(f"Failed to deduct funds: {e}")
    
    def get_wallet_transactions(self, user_id: str, transaction_type: Optional[str] = None,
                               limit: int = 100, offset: int = 0) -> Tuple[List[Dict], int]:
        """获取钱包交易记录"""
        try:
            if self.wallet_transaction_repo:
                transactions, total = self.wallet_transaction_repo.get_by_user(
                    user_id, transaction_type, limit, offset
                )
                return [self._transaction_to_dict(t) for t in transactions], total
            else:
                return [], 0
            
        except Exception as e:
            logger.error(f"Failed to get wallet transactions: {e}")
            return [], 0
    
    # ========== 统计和报表 ==========
    
    def get_user_billing_summary(self, user_id: str, period_start: datetime, 
                                period_end: datetime) -> Dict[str, Any]:
        """获取用户计费摘要"""
        try:
            # 获取使用汇总
            usage_summary = self.get_usage_summary(user_id, period_start, period_end)
            
            # 获取发票统计
            invoices, _ = self.get_user_invoices(user_id)
            
            paid_invoices = [i for i in invoices if i.get('payment_status') == PaymentStatus.PAID]
            pending_invoices = [i for i in invoices if i.get('payment_status') == PaymentStatus.PENDING]
            
            return {
                'user_id': user_id,
                'period_start': period_start.isoformat(),
                'period_end': period_end.isoformat(),
                'total_cost': usage_summary.get('total_cost', 0),
                'resource_summary': usage_summary.get('by_resource_type', {}),
                'invoice_count': len(invoices),
                'paid_invoices': len(paid_invoices),
                'pending_invoices': len(pending_invoices),
                'total_paid': sum(Decimal(str(i.get('total_amount', 0))) for i in paid_invoices),
                'total_pending': sum(Decimal(str(i.get('total_amount', 0))) for i in pending_invoices)
            }
            
        except Exception as e:
            logger.error(f"Failed to get user billing summary: {e}")
            return {}
    
    def get_billing_dashboard_data(self) -> Dict[str, Any]:
        """获取计费仪表板数据"""
        try:
            now = datetime.utcnow()
            current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            # 当月收入统计
            current_month_revenue = Decimal('0')
            revenue_by_resource = {}
            
            if self.usage_record_repo:
                # 从数据库获取统计
                records, _ = self.usage_record_repo.get_by_tenant('', current_month_start, now)
                for record in records:
                    record_dict = self._record_to_dict(record)
                    cost = Decimal(str(record_dict.get('cost', 0)))
                    resource_type = record_dict.get('resource_type')
                    
                    current_month_revenue += cost
                    if resource_type not in revenue_by_resource:
                        revenue_by_resource[resource_type] = Decimal('0')
                    revenue_by_resource[resource_type] += cost
            else:
                for record in self.usage_records.values():
                    if record.start_time >= current_month_start:
                        current_month_revenue += record.cost
                        if record.resource_type not in revenue_by_resource:
                            revenue_by_resource[record.resource_type] = Decimal('0')
                        revenue_by_resource[record.resource_type] += record.cost
            
            # 发票统计
            if self.invoice_repo:
                all_invoices, total_invoices = self.invoice_repo.get_by_user('', limit=10000)
            else:
                all_invoices = list(self.invoices.values())
                total_invoices = len(all_invoices)
            
            paid_count = len([i for i in all_invoices 
                             if (i.get('payment_status') if isinstance(i, dict) else i.payment_status) == PaymentStatus.PAID])
            pending_count = len([i for i in all_invoices 
                                if (i.get('payment_status') if isinstance(i, dict) else i.payment_status) == PaymentStatus.PENDING])
            overdue_count = len(self.get_overdue_invoices())
            
            # 活跃规则数
            rules, _ = self.list_billing_rules(active_only=True)
            
            return {
                'current_month_revenue': float(current_month_revenue),
                'total_invoices': total_invoices,
                'paid_invoices': paid_count,
                'pending_invoices': pending_count,
                'overdue_invoices': overdue_count,
                'payment_rate': (paid_count / total_invoices * 100) if total_invoices > 0 else 0,
                'revenue_by_resource': {k: float(v) for k, v in revenue_by_resource.items()},
                'active_billing_rules': len(rules),
                'total_usage_records': len(self.usage_records) if not self.usage_record_repo else 'N/A'
            }
            
        except Exception as e:
            logger.error(f"Failed to get billing dashboard data: {e}")
            return {}

    # ========== 辅助方法 ==========
    
    def _find_applicable_rule(self, resource_type: str) -> Optional[Dict]:
        """查找适用的计费规则"""
        try:
            if self.billing_rule_repo:
                rules = self.billing_rule_repo.get_by_resource_type(resource_type, active_only=True)
                if rules:
                    return self._rule_to_dict(rules[0])
                return None
            else:
                now = datetime.utcnow()
                for rule in self.billing_rules.values():
                    if (rule.resource_type == resource_type
                        and rule.is_active
                        and rule.effective_date <= now
                        and (rule.expiry_date is None or rule.expiry_date >= now)):
                        return self._rule_to_dict(rule)
                return None
                
        except Exception as e:
            logger.error(f"Failed to find applicable rule: {e}")
            return None
    
    def _find_rule_by_id(self, rule_id: str) -> Optional[Dict]:
        """根据ID查找计费规则"""
        if not rule_id:
            return None
        return self.get_billing_rule(rule_id)
    
    def _rule_to_dict(self, rule) -> Optional[Dict]:
        """将规则转换为字典"""
        if rule is None:
            return None
        if isinstance(rule, dict):
            # 确保字典有默认值
            rule.setdefault('currency', 'CNY')
            rule.setdefault('unit', 'unit')
            rule.setdefault('is_active', True)
            rule.setdefault('billing_model', 'pay_as_you_go')
            rule.setdefault('billing_period', BillingPeriod.HOURLY)
            rule.setdefault('free_tier', 0)
            rule.setdefault('discount_rate', 0)
            return rule
        return {
            'id': getattr(rule, 'id', None),
            'name': getattr(rule, 'name', None),
            'description': getattr(rule, 'description', None),
            'resource_type': getattr(rule, 'resource_type', None),
            'billing_model': getattr(rule, 'billing_model', None),
            'billing_period': getattr(rule, 'billing_period', None),
            'unit_price': float(getattr(rule, 'unit_price', 0)),
            'unit': getattr(rule, 'unit', 'unit'),
            'currency': getattr(rule, 'currency', 'CNY'),
            'free_tier': float(getattr(rule, 'free_tier', 0)),
            'minimum_charge': float(getattr(rule, 'minimum_charge', 0)) if getattr(rule, 'minimum_charge', None) else None,
            'discount_rate': float(getattr(rule, 'discount_rate', 0)),
            'is_active': getattr(rule, 'is_active', True),
            'effective_date': getattr(rule, 'effective_date', datetime.utcnow()).isoformat() if getattr(rule, 'effective_date', None) else None,
            'expiry_date': getattr(rule, 'expiry_date', None).isoformat() if getattr(rule, 'expiry_date', None) else None,
            'created_at': getattr(rule, 'created_at', datetime.utcnow()).isoformat() if getattr(rule, 'created_at', None) else None,
            'updated_at': getattr(rule, 'updated_at', datetime.utcnow()).isoformat() if getattr(rule, 'updated_at', None) else None
        }
    
    def _record_to_dict(self, record) -> Optional[Dict]:
        """将使用记录转换为字典"""
        if record is None:
            return None
        if isinstance(record, dict):
            return record
        return {
            'id': getattr(record, 'id', None),
            'tenant_id': getattr(record, 'tenant_id', None),
            'user_id': getattr(record, 'user_id', None),
            'resource_type': getattr(record, 'resource_type', None),
            'resource_id': getattr(record, 'resource_id', None),
            'resource_name': getattr(record, 'resource_name', None),
            'usage_amount': float(getattr(record, 'usage_amount', 0)),
            'unit': getattr(record, 'unit', 'unit'),
            'start_time': getattr(record, 'start_time', None).isoformat() if getattr(record, 'start_time', None) else None,
            'end_time': getattr(record, 'end_time', None).isoformat() if getattr(record, 'end_time', None) else None,
            'cost': float(getattr(record, 'cost', 0)),
            'billing_rule_id': getattr(record, 'billing_rule_id', None),
            'billing_status': getattr(record, 'billing_status', 'pending'),
            'invoice_id': getattr(record, 'invoice_id', None),
            'created_at': getattr(record, 'created_at', datetime.utcnow()).isoformat() if getattr(record, 'created_at', None) else None
        }
    
    def _invoice_to_dict(self, invoice) -> Optional[Dict]:
        """将发票转换为字典"""
        if invoice is None:
            return None
        if isinstance(invoice, dict):
            return invoice
        return {
            'id': getattr(invoice, 'id', None),
            'invoice_number': getattr(invoice, 'invoice_number', None),
            'tenant_id': getattr(invoice, 'tenant_id', None),
            'user_id': getattr(invoice, 'user_id', None),
            'billing_period_start': getattr(invoice, 'billing_period_start', None).isoformat() if getattr(invoice, 'billing_period_start', None) else None,
            'billing_period_end': getattr(invoice, 'billing_period_end', None).isoformat() if getattr(invoice, 'billing_period_end', None) else None,
            'subtotal': float(getattr(invoice, 'subtotal', 0)),
            'discount_amount': float(getattr(invoice, 'discount_amount', 0)),
            'tax_amount': float(getattr(invoice, 'tax_amount', 0)),
            'total_amount': float(getattr(invoice, 'total_amount', 0)),
            'currency': getattr(invoice, 'currency', 'CNY'),
            'status': getattr(invoice, 'status', InvoiceStatus.DRAFT),
            'payment_status': getattr(invoice, 'payment_status', PaymentStatus.PENDING),
            'due_date': getattr(invoice, 'due_date', None).isoformat() if getattr(invoice, 'due_date', None) else None,
            'paid_date': getattr(invoice, 'paid_date', None).isoformat() if getattr(invoice, 'paid_date', None) else None,
            'items': [self._item_to_dict(i) for i in getattr(invoice, 'items', [])] if hasattr(invoice, 'items') else [],
            'created_at': getattr(invoice, 'created_at', datetime.utcnow()).isoformat() if getattr(invoice, 'created_at', None) else None
        }
    
    def _item_to_dict(self, item) -> Optional[Dict]:
        """将计费项目转换为字典"""
        if item is None:
            return None
        if isinstance(item, dict):
            return item
        return {
            'id': getattr(item, 'id', None),
            'invoice_id': getattr(item, 'invoice_id', None),
            'usage_record_id': getattr(item, 'usage_record_id', None),
            'description': getattr(item, 'description', None),
            'resource_type': getattr(item, 'resource_type', None),
            'quantity': float(getattr(item, 'quantity', 0)),
            'unit': getattr(item, 'unit', 'unit'),
            'unit_price': float(getattr(item, 'unit_price', 0)),
            'subtotal': float(getattr(item, 'subtotal', 0)),
            'discount_amount': float(getattr(item, 'discount_amount', 0)),
            'tax_amount': float(getattr(item, 'tax_amount', 0)),
            'total_amount': float(getattr(item, 'total_amount', 0))
        }
    
    def _payment_to_dict(self, payment) -> Optional[Dict]:
        """将支付记录转换为字典"""
        if payment is None:
            return None
        if isinstance(payment, dict):
            return payment
        return {
            'id': getattr(payment, 'id', None),
            'payment_number': getattr(payment, 'payment_number', None),
            'tenant_id': getattr(payment, 'tenant_id', None),
            'invoice_id': getattr(payment, 'invoice_id', None),
            'user_id': getattr(payment, 'user_id', None),
            'amount': float(getattr(payment, 'amount', 0)),
            'currency': getattr(payment, 'currency', 'CNY'),
            'payment_method': getattr(payment, 'payment_method', None),
            'transaction_id': getattr(payment, 'transaction_id', None),
            'status': getattr(payment, 'status', PaymentStatus.PENDING),
            'payment_date': getattr(payment, 'payment_date', None).isoformat() if getattr(payment, 'payment_date', None) else None,
            'created_at': getattr(payment, 'created_at', datetime.utcnow()).isoformat() if getattr(payment, 'created_at', None) else None
        }
    
    def _wallet_to_dict(self, wallet) -> Optional[Dict]:
        """将钱包转换为字典"""
        if wallet is None:
            return None
        if isinstance(wallet, dict):
            # 确保字典有默认值
            wallet.setdefault('balance', 0)
            wallet.setdefault('frozen_balance', 0)
            wallet.setdefault('currency', 'CNY')
            wallet.setdefault('credit_limit', 0)
            wallet.setdefault('status', 'active')
            return wallet
        return {
            'id': getattr(wallet, 'id', None),
            'tenant_id': getattr(wallet, 'tenant_id', None),
            'user_id': getattr(wallet, 'user_id', None),
            'balance': float(getattr(wallet, 'balance', 0)),
            'frozen_balance': float(getattr(wallet, 'frozen_balance', 0)),
            'currency': getattr(wallet, 'currency', 'CNY'),
            'credit_limit': float(getattr(wallet, 'credit_limit', 0)),
            'status': getattr(wallet, 'status', 'active'),
            'last_transaction_at': getattr(wallet, 'last_transaction_at', None).isoformat() if getattr(wallet, 'last_transaction_at', None) else None
        }
    
    def _transaction_to_dict(self, transaction) -> Optional[Dict]:
        """将钱包交易转换为字典"""
        if transaction is None:
            return None
        if isinstance(transaction, dict):
            return transaction
        return {
            'id': getattr(transaction, 'id', None),
            'wallet_id': getattr(transaction, 'wallet_id', None),
            'user_id': getattr(transaction, 'user_id', None),
            'transaction_type': getattr(transaction, 'transaction_type', None),
            'amount': float(getattr(transaction, 'amount', 0)),
            'balance_before': float(getattr(transaction, 'balance_before', 0)),
            'balance_after': float(getattr(transaction, 'balance_after', 0)),
            'reference_type': getattr(transaction, 'reference_type', None),
            'reference_id': getattr(transaction, 'reference_id', None),
            'description': getattr(transaction, 'description', None),
            'status': getattr(transaction, 'status', 'completed'),
            'created_at': getattr(transaction, 'created_at', datetime.utcnow()).isoformat() if getattr(transaction, 'created_at', None) else None
        }


# ============================================================================
# 全局实例和工厂函数
# ============================================================================

_global_billing_service: Optional[BillingService] = None


def get_billing_service(use_memory_storage: bool = False) -> BillingService:
    """获取全局计费服务实例
    
    Args:
        use_memory_storage: 是否使用内存存储（仅在首次创建时生效）
    
    Returns:
        BillingService 实例
    """
    global _global_billing_service
    if _global_billing_service is None:
        _global_billing_service = BillingService(use_memory_storage=use_memory_storage)
    return _global_billing_service


def reset_billing_service():
    """重置全局计费服务实例（主要用于测试）"""
    global _global_billing_service
    _global_billing_service = None
