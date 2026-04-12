"""计费数据库模型

定义计费相关的数据库模型，包括：
- 计费规则 (BillingRule)
- 使用记录 (UsageRecord)
- 计费项目 (BillingItem)
- 发票 (Invoice)
- 支付记录 (Payment)
"""

from sqlalchemy import Column, String, Text, Boolean, Integer, Float, DateTime, Numeric, ForeignKey, func
from sqlalchemy.orm import relationship
from decimal import Decimal
from datetime import datetime
from enum import Enum

from .base_models import Base, UUIDMixin, TimestampMixin, TenantMixin


# ============================================================================
# 计费相关枚举
# ============================================================================

class BillingPeriod(str, Enum):
    """计费周期枚举"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class BillingModel(str, Enum):
    """计费模式枚举"""
    PAY_AS_YOU_GO = "pay_as_you_go"  # 按量付费
    SUBSCRIPTION = "subscription"    # 订阅模式
    PREPAID = "prepaid"              # 预付费
    TIERED = "tiered"                # 阶梯计费


class InvoiceStatus(str, Enum):
    """发票状态枚举"""
    DRAFT = "draft"
    PENDING = "pending"
    SENT = "sent"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class PaymentStatus(str, Enum):
    """支付状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    PAID = "paid"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    OVERDUE = "overdue"


class PaymentMethod(str, Enum):
    """支付方式枚举"""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BANK_TRANSFER = "bank_transfer"
    ALIPAY = "alipay"
    WECHAT_PAY = "wechat_pay"
    PAYPAL = "paypal"
    WALLET = "wallet"


class ResourceType(str, Enum):
    """资源类型枚举"""
    GPU = "gpu"
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    API_CALLS = "api_calls"
    TRAINING_HOURS = "training_hours"
    INFERENCE_CALLS = "inference_calls"


# ============================================================================
# 计费数据库模型
# ============================================================================

class BillingRule(Base, UUIDMixin, TimestampMixin):
    """计费规则模型"""
    __tablename__ = 'billing_rules'
    
    name = Column(String(100), nullable=False, comment="规则名称")
    description = Column(Text, comment="规则描述")
    resource_type = Column(String(50), nullable=False, index=True, comment="资源类型")
    billing_model = Column(String(50), default='pay_as_you_go', comment="计费模式")
    billing_period = Column(String(20), default='hourly', comment="计费周期")
    unit_price = Column(Numeric(10, 4), nullable=False, comment="单价")
    unit = Column(String(20), default='unit', comment="计量单位")
    currency = Column(String(10), default='CNY', comment="货币")
    free_tier = Column(Numeric(10, 4), default=0, comment="免费额度")
    minimum_charge = Column(Numeric(10, 4), comment="最低收费")
    discount_rate = Column(Numeric(5, 4), default=0, comment="折扣率")
    is_active = Column(Boolean, default=True, index=True, comment="是否激活")
    effective_date = Column(DateTime, default=func.now(), comment="生效日期")
    expiry_date = Column(DateTime, comment="失效日期")
    tier_config = Column(Text, comment="阶梯配置(JSON)")
    extra_data = Column(Text, comment="额外数据(JSON)")
    
    def __repr__(self):
        return f"<BillingRule(id='{self.id}', name='{self.name}', resource_type='{self.resource_type}')>"


class UsageRecord(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """资源使用记录模型"""
    __tablename__ = 'usage_records'
    
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    resource_type = Column(String(50), nullable=False, index=True, comment="资源类型")
    resource_id = Column(String(100), comment="资源ID")
    resource_name = Column(String(200), comment="资源名称")
    usage_amount = Column(Numeric(20, 6), nullable=False, comment="使用量")
    unit = Column(String(20), default='unit', comment="计量单位")
    start_time = Column(DateTime, nullable=False, index=True, comment="开始时间")
    end_time = Column(DateTime, nullable=False, index=True, comment="结束时间")
    cost = Column(Numeric(15, 4), default=0, comment="费用")
    billing_rule_id = Column(String(36), index=True, comment="计费规则ID")
    billing_status = Column(String(20), default='pending', index=True, comment="计费状态")
    invoice_id = Column(String(36), index=True, comment="发票ID")
    extra_data = Column(Text, comment="额外数据(JSON)")
    
    def __repr__(self):
        return f"<UsageRecord(id='{self.id}', resource_type='{self.resource_type}', usage_amount={self.usage_amount})>"


class Invoice(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """发票模型"""
    __tablename__ = 'invoices'
    
    invoice_number = Column(String(50), unique=True, nullable=False, index=True, comment="发票编号")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    billing_period_start = Column(DateTime, nullable=False, comment="账单周期开始")
    billing_period_end = Column(DateTime, nullable=False, comment="账单周期结束")
    subtotal = Column(Numeric(15, 4), default=0, comment="小计")
    discount_amount = Column(Numeric(15, 4), default=0, comment="折扣金额")
    tax_amount = Column(Numeric(15, 4), default=0, comment="税额")
    total_amount = Column(Numeric(15, 4), default=0, comment="总金额")
    currency = Column(String(10), default='CNY', comment="货币")
    status = Column(String(20), default='draft', index=True, comment="发票状态")
    payment_status = Column(String(20), default='pending', index=True, comment="支付状态")
    due_date = Column(DateTime, comment="到期日期")
    paid_date = Column(DateTime, comment="支付日期")
    notes = Column(Text, comment="备注")
    billing_address = Column(Text, comment="账单地址(JSON)")
    extra_data = Column(Text, comment="额外数据(JSON)")
    
    def __repr__(self):
        return f"<Invoice(id='{self.id}', invoice_number='{self.invoice_number}', total_amount={self.total_amount})>"


class BillingItem(Base, UUIDMixin, TimestampMixin):
    """计费项目模型（发票明细）"""
    __tablename__ = 'billing_items'
    
    invoice_id = Column(String(36), nullable=False, index=True, comment="发票ID")
    usage_record_id = Column(String(36), index=True, comment="使用记录ID")
    description = Column(String(500), nullable=False, comment="描述")
    resource_type = Column(String(50), comment="资源类型")
    quantity = Column(Numeric(20, 6), nullable=False, comment="数量")
    unit = Column(String(20), default='unit', comment="单位")
    unit_price = Column(Numeric(10, 4), nullable=False, comment="单价")
    subtotal = Column(Numeric(15, 4), nullable=False, comment="小计")
    discount_amount = Column(Numeric(15, 4), default=0, comment="折扣金额")
    tax_amount = Column(Numeric(15, 4), default=0, comment="税额")
    total_amount = Column(Numeric(15, 4), nullable=False, comment="总金额")
    period_start = Column(DateTime, comment="计费周期开始")
    period_end = Column(DateTime, comment="计费周期结束")
    extra_data = Column(Text, comment="额外数据(JSON)")
    
    def __repr__(self):
        return f"<BillingItem(id='{self.id}', invoice_id='{self.invoice_id}', total_amount={self.total_amount})>"


class Payment(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """支付记录模型"""
    __tablename__ = 'payments'
    
    payment_number = Column(String(50), unique=True, nullable=False, index=True, comment="支付编号")
    invoice_id = Column(String(36), nullable=False, index=True, comment="发票ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    amount = Column(Numeric(15, 4), nullable=False, comment="支付金额")
    currency = Column(String(10), default='CNY', comment="货币")
    payment_method = Column(String(50), nullable=False, comment="支付方式")
    transaction_id = Column(String(100), index=True, comment="交易ID")
    status = Column(String(20), default='pending', index=True, comment="支付状态")
    payment_date = Column(DateTime, comment="支付时间")
    failure_reason = Column(Text, comment="失败原因")
    refund_amount = Column(Numeric(15, 4), default=0, comment="退款金额")
    refund_date = Column(DateTime, comment="退款时间")
    refund_reason = Column(Text, comment="退款原因")
    gateway_response = Column(Text, comment="网关响应(JSON)")
    extra_data = Column(Text, comment="额外数据(JSON)")
    
    def __repr__(self):
        return f"<Payment(id='{self.id}', payment_number='{self.payment_number}', amount={self.amount})>"


class Wallet(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """钱包/账户余额模型"""
    __tablename__ = 'wallets'
    
    user_id = Column(String(36), nullable=False, unique=True, index=True, comment="用户ID")
    balance = Column(Numeric(15, 4), default=0, comment="余额")
    frozen_balance = Column(Numeric(15, 4), default=0, comment="冻结余额")
    currency = Column(String(10), default='CNY', comment="货币")
    credit_limit = Column(Numeric(15, 4), default=0, comment="信用额度")
    status = Column(String(20), default='active', index=True, comment="状态")
    last_transaction_at = Column(DateTime, comment="最后交易时间")
    extra_data = Column(Text, comment="额外数据(JSON)")
    
    def __repr__(self):
        return f"<Wallet(id='{self.id}', user_id='{self.user_id}', balance={self.balance})>"


class WalletTransaction(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """钱包交易记录模型"""
    __tablename__ = 'wallet_transactions'
    
    wallet_id = Column(String(36), nullable=False, index=True, comment="钱包ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    transaction_type = Column(String(50), nullable=False, index=True, comment="交易类型")
    amount = Column(Numeric(15, 4), nullable=False, comment="金额")
    balance_before = Column(Numeric(15, 4), nullable=False, comment="交易前余额")
    balance_after = Column(Numeric(15, 4), nullable=False, comment="交易后余额")
    reference_type = Column(String(50), comment="关联类型")
    reference_id = Column(String(36), comment="关联ID")
    description = Column(String(500), comment="描述")
    status = Column(String(20), default='completed', index=True, comment="状态")
    extra_data = Column(Text, comment="额外数据(JSON)")
    
    def __repr__(self):
        return f"<WalletTransaction(id='{self.id}', transaction_type='{self.transaction_type}', amount={self.amount})>"

