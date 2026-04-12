"""计费数据访问层

提供计费相关的数据库访问功能，包括：
- 计费规则 (BillingRule)
- 使用记录 (UsageRecord)
- 发票 (Invoice)
- 计费项目 (BillingItem)
- 支付记录 (Payment)
- 钱包 (Wallet)
- 钱包交易 (WalletTransaction)
"""

import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from decimal import Decimal
import uuid

from backend.core.exceptions import ValidationError, DatabaseError
from backend.schemas.billing_models import (
    BillingRule, UsageRecord, Invoice, BillingItem, Payment,
    Wallet, WalletTransaction, BillingPeriod, BillingModel,
    InvoiceStatus, PaymentStatus
)

logger = logging.getLogger(__name__)


def _generate_id() -> str:
    """生成唯一ID"""
    return str(uuid.uuid4())


def _generate_invoice_number() -> str:
    """生成发票编号"""
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    suffix = str(uuid.uuid4())[:6].upper()
    return f"INV-{timestamp}-{suffix}"


def _generate_payment_number() -> str:
    """生成支付编号"""
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    suffix = str(uuid.uuid4())[:6].upper()
    return f"PAY-{timestamp}-{suffix}"


class BillingRuleRepository:
    """计费规则数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化计费规则仓库
        
        Args:
            use_memory_storage: 是否使用内存存储
        """
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._rules: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._rules: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, rule_data: Dict[str, Any]) -> BillingRule:
        """创建计费规则"""
        try:
            rule_id = rule_data.get('id') or _generate_id()
            
            if self._use_memory_storage:
                rule_data['id'] = rule_id
                rule_data['created_at'] = datetime.utcnow()
                rule_data['updated_at'] = datetime.utcnow()
                self._rules[rule_id] = rule_data
                return rule_data
            
            with self._db_manager.get_db_session() as db:
                tier_config = rule_data.get('tier_config', {})
                if isinstance(tier_config, dict):
                    tier_config = json.dumps(tier_config)
                
                extra_data = rule_data.get('extra_data', rule_data.get('metadata', {}))
                if isinstance(extra_data, dict):
                    extra_data = json.dumps(extra_data)
                
                rule = BillingRule(
                    id=rule_id,
                    name=rule_data['name'],
                    description=rule_data.get('description'),
                    resource_type=rule_data['resource_type'],
                    billing_model=rule_data.get('billing_model', 'pay_as_you_go'),
                    billing_period=rule_data.get('billing_period', 'hourly'),
                    unit_price=Decimal(str(rule_data['unit_price'])),
                    unit=rule_data.get('unit', 'unit'),
                    currency=rule_data.get('currency', 'CNY'),
                    free_tier=Decimal(str(rule_data.get('free_tier', 0))),
                    minimum_charge=Decimal(str(rule_data.get('minimum_charge', 0))) if rule_data.get('minimum_charge') else None,
                    discount_rate=Decimal(str(rule_data.get('discount_rate', 0))),
                    is_active=rule_data.get('is_active', True),
                    effective_date=rule_data.get('effective_date', datetime.utcnow()),
                    expiry_date=rule_data.get('expiry_date'),
                    tier_config=tier_config,
                    extra_data=extra_data
                )
                db.add(rule)
                db.commit()
                db.refresh(rule)
                return rule
                
        except Exception as e:
            logger.error(f"Failed to create billing rule: {e}")
            raise DatabaseError(f"Failed to create billing rule: {e}", operation="create_billing_rule")
    
    def get_by_id(self, rule_id: str) -> Optional[BillingRule]:
        """根据ID获取计费规则"""
        try:
            if self._use_memory_storage:
                return self._rules.get(rule_id)
            
            with self._db_manager.get_db_session() as db:
                return db.query(BillingRule).filter(BillingRule.id == rule_id).first()
                
        except Exception as e:
            logger.error(f"Failed to get billing rule: {e}")
            return None
    
    def get_by_resource_type(self, resource_type: str, active_only: bool = True) -> List[BillingRule]:
        """根据资源类型获取计费规则"""
        try:
            now = datetime.utcnow()
            
            if self._use_memory_storage:
                rules = [r for r in self._rules.values() if r.get('resource_type') == resource_type]
                if active_only:
                    rules = [r for r in rules if r.get('is_active', True)]
                    rules = [r for r in rules if r.get('effective_date', datetime.min) <= now]
                    rules = [r for r in rules if r.get('expiry_date') is None or r.get('expiry_date') >= now]
                return rules
            
            with self._db_manager.get_db_session() as db:
                query = db.query(BillingRule).filter(BillingRule.resource_type == resource_type)
                if active_only:
                    query = query.filter(BillingRule.is_active == True)
                    query = query.filter(BillingRule.effective_date <= now)
                    query = query.filter(
                        (BillingRule.expiry_date == None) | (BillingRule.expiry_date >= now)
                    )
                return query.all()
                
        except Exception as e:
            logger.error(f"Failed to get billing rules by resource type: {e}")
            return []
    
    def update(self, rule_id: str, update_data: Dict[str, Any]) -> Optional[BillingRule]:
        """更新计费规则"""
        try:
            if self._use_memory_storage:
                if rule_id in self._rules:
                    self._rules[rule_id].update(update_data)
                    self._rules[rule_id]['updated_at'] = datetime.utcnow()
                    return self._rules[rule_id]
                return None
            
            with self._db_manager.get_db_session() as db:
                rule = db.query(BillingRule).filter(BillingRule.id == rule_id).first()
                if not rule:
                    return None
                
                for key, value in update_data.items():
                    if key in ('tier_config', 'extra_data', 'metadata') and isinstance(value, dict):
                        value = json.dumps(value)
                    if key == 'metadata':
                        key = 'extra_data'
                    if key in ('unit_price', 'free_tier', 'minimum_charge', 'discount_rate') and value is not None:
                        value = Decimal(str(value))
                    if hasattr(rule, key):
                        setattr(rule, key, value)
                
                rule.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(rule)
                return rule
                
        except Exception as e:
            logger.error(f"Failed to update billing rule: {e}")
            raise DatabaseError(f"Failed to update billing rule: {e}", operation="update_billing_rule")
    
    def delete(self, rule_id: str) -> bool:
        """删除计费规则"""
        try:
            if self._use_memory_storage:
                if rule_id in self._rules:
                    del self._rules[rule_id]
                    return True
                return False
            
            with self._db_manager.get_db_session() as db:
                rule = db.query(BillingRule).filter(BillingRule.id == rule_id).first()
                if not rule:
                    return False
                
                db.delete(rule)
                db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete billing rule: {e}")
            return False
    
    def list_all(self, active_only: bool = True, limit: int = 100, offset: int = 0) -> Tuple[List[BillingRule], int]:
        """获取计费规则列表"""
        try:
            if self._use_memory_storage:
                rules = list(self._rules.values())
                if active_only:
                    rules = [r for r in rules if r.get('is_active', True)]
                total = len(rules)
                return rules[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(BillingRule)
                if active_only:
                    query = query.filter(BillingRule.is_active == True)
                
                total = query.count()
                rules = query.order_by(BillingRule.created_at.desc()).offset(offset).limit(limit).all()
                return rules, total
                
        except Exception as e:
            logger.error(f"Failed to list billing rules: {e}")
            return [], 0


class UsageRecordRepository:
    """使用记录数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化使用记录仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._records: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._records: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, record_data: Dict[str, Any]) -> UsageRecord:
        """创建使用记录"""
        try:
            record_id = record_data.get('id') or _generate_id()
            
            if self._use_memory_storage:
                record_data['id'] = record_id
                record_data['created_at'] = datetime.utcnow()
                record_data['updated_at'] = datetime.utcnow()
                self._records[record_id] = record_data
                return record_data
            
            with self._db_manager.get_db_session() as db:
                extra_data = record_data.get('extra_data', record_data.get('metadata', {}))
                if isinstance(extra_data, dict):
                    extra_data = json.dumps(extra_data)
                
                record = UsageRecord(
                    id=record_id,
                    tenant_id=record_data.get('tenant_id'),
                    user_id=record_data['user_id'],
                    resource_type=record_data['resource_type'],
                    resource_id=record_data.get('resource_id'),
                    resource_name=record_data.get('resource_name'),
                    usage_amount=Decimal(str(record_data['usage_amount'])),
                    unit=record_data.get('unit', 'unit'),
                    start_time=record_data['start_time'],
                    end_time=record_data['end_time'],
                    cost=Decimal(str(record_data.get('cost', 0))),
                    billing_rule_id=record_data.get('billing_rule_id'),
                    billing_status=record_data.get('billing_status', 'pending'),
                    invoice_id=record_data.get('invoice_id'),
                    extra_data=extra_data
                )
                db.add(record)
                db.commit()
                db.refresh(record)
                return record
                
        except Exception as e:
            logger.error(f"Failed to create usage record: {e}")
            raise DatabaseError(f"Failed to create usage record: {e}", operation="create_usage_record")
    
    def get_by_id(self, record_id: str) -> Optional[UsageRecord]:
        """根据ID获取使用记录"""
        try:
            if self._use_memory_storage:
                return self._records.get(record_id)
            
            with self._db_manager.get_db_session() as db:
                return db.query(UsageRecord).filter(UsageRecord.id == record_id).first()
                
        except Exception as e:
            logger.error(f"Failed to get usage record: {e}")
            return None
    
    def get_by_user(self, user_id: str, start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None, resource_type: Optional[str] = None,
                   limit: int = 100, offset: int = 0) -> Tuple[List[UsageRecord], int]:
        """获取用户的使用记录"""
        try:
            if self._use_memory_storage:
                records = [r for r in self._records.values() if r.get('user_id') == user_id]
                if start_time:
                    records = [r for r in records if r.get('start_time', datetime.max) >= start_time]
                if end_time:
                    records = [r for r in records if r.get('end_time', datetime.min) <= end_time]
                if resource_type:
                    records = [r for r in records if r.get('resource_type') == resource_type]
                total = len(records)
                return records[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(UsageRecord).filter(UsageRecord.user_id == user_id)
                if start_time:
                    query = query.filter(UsageRecord.start_time >= start_time)
                if end_time:
                    query = query.filter(UsageRecord.end_time <= end_time)
                if resource_type:
                    query = query.filter(UsageRecord.resource_type == resource_type)
                
                total = query.count()
                records = query.order_by(UsageRecord.start_time.desc()).offset(offset).limit(limit).all()
                return records, total
                
        except Exception as e:
            logger.error(f"Failed to get usage records by user: {e}")
            return [], 0
    
    def get_by_tenant(self, tenant_id: str, start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None, limit: int = 100, 
                     offset: int = 0) -> Tuple[List[UsageRecord], int]:
        """获取租户的使用记录"""
        try:
            if self._use_memory_storage:
                records = [r for r in self._records.values() if r.get('tenant_id') == tenant_id]
                if start_time:
                    records = [r for r in records if r.get('start_time', datetime.max) >= start_time]
                if end_time:
                    records = [r for r in records if r.get('end_time', datetime.min) <= end_time]
                total = len(records)
                return records[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(UsageRecord).filter(UsageRecord.tenant_id == tenant_id)
                if start_time:
                    query = query.filter(UsageRecord.start_time >= start_time)
                if end_time:
                    query = query.filter(UsageRecord.end_time <= end_time)
                
                total = query.count()
                records = query.order_by(UsageRecord.start_time.desc()).offset(offset).limit(limit).all()
                return records, total
                
        except Exception as e:
            logger.error(f"Failed to get usage records by tenant: {e}")
            return [], 0
    
    def get_unbilled_records(self, user_id: str, start_time: datetime, 
                            end_time: datetime) -> List[UsageRecord]:
        """获取未计费的使用记录"""
        try:
            if self._use_memory_storage:
                records = [
                    r for r in self._records.values() 
                    if r.get('user_id') == user_id
                    and r.get('billing_status') == 'pending'
                    and r.get('start_time', datetime.max) >= start_time
                    and r.get('end_time', datetime.min) <= end_time
                ]
                return records
            
            with self._db_manager.get_db_session() as db:
                return db.query(UsageRecord).filter(
                    UsageRecord.user_id == user_id,
                    UsageRecord.billing_status == 'pending',
                    UsageRecord.start_time >= start_time,
                    UsageRecord.end_time <= end_time
                ).all()
                
        except Exception as e:
            logger.error(f"Failed to get unbilled records: {e}")
            return []
    
    def update(self, record_id: str, update_data: Dict[str, Any]) -> Optional[UsageRecord]:
        """更新使用记录"""
        try:
            if self._use_memory_storage:
                if record_id in self._records:
                    self._records[record_id].update(update_data)
                    self._records[record_id]['updated_at'] = datetime.utcnow()
                    return self._records[record_id]
                return None
            
            with self._db_manager.get_db_session() as db:
                record = db.query(UsageRecord).filter(UsageRecord.id == record_id).first()
                if not record:
                    return None
                
                for key, value in update_data.items():
                    if key in ('extra_data', 'metadata') and isinstance(value, dict):
                        value = json.dumps(value)
                    if key == 'metadata':
                        key = 'extra_data'
                    if key in ('usage_amount', 'cost') and value is not None:
                        value = Decimal(str(value))
                    if hasattr(record, key):
                        setattr(record, key, value)
                
                record.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(record)
                return record
                
        except Exception as e:
            logger.error(f"Failed to update usage record: {e}")
            raise DatabaseError(f"Failed to update usage record: {e}", operation="update_usage_record")
    
    def mark_as_billed(self, record_ids: List[str], invoice_id: str) -> int:
        """将记录标记为已计费"""
        try:
            updated_count = 0
            
            if self._use_memory_storage:
                for rid in record_ids:
                    if rid in self._records:
                        self._records[rid]['billing_status'] = 'billed'
                        self._records[rid]['invoice_id'] = invoice_id
                        self._records[rid]['updated_at'] = datetime.utcnow()
                        updated_count += 1
                return updated_count
            
            with self._db_manager.get_db_session() as db:
                updated_count = db.query(UsageRecord).filter(
                    UsageRecord.id.in_(record_ids)
                ).update({
                    'billing_status': 'billed',
                    'invoice_id': invoice_id,
                    'updated_at': datetime.utcnow()
                }, synchronize_session=False)
                db.commit()
                return updated_count
                
        except Exception as e:
            logger.error(f"Failed to mark records as billed: {e}")
            return 0
    
    def get_usage_summary(self, user_id: str, start_time: datetime, 
                         end_time: datetime) -> Dict[str, Any]:
        """获取使用汇总"""
        try:
            if self._use_memory_storage:
                records = [
                    r for r in self._records.values()
                    if r.get('user_id') == user_id
                    and r.get('start_time', datetime.max) >= start_time
                    and r.get('end_time', datetime.min) <= end_time
                ]
                
                summary = {}
                total_cost = Decimal('0')
                
                for record in records:
                    resource_type = record.get('resource_type')
                    if resource_type not in summary:
                        summary[resource_type] = {
                            'usage_amount': Decimal('0'),
                            'cost': Decimal('0'),
                            'unit': record.get('unit', 'unit'),
                            'record_count': 0
                        }
                    
                    usage_amount = Decimal(str(record.get('usage_amount', 0)))
                    cost = Decimal(str(record.get('cost', 0)))
                    
                    summary[resource_type]['usage_amount'] += usage_amount
                    summary[resource_type]['cost'] += cost
                    summary[resource_type]['record_count'] += 1
                    total_cost += cost
                
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
            
            with self._db_manager.get_db_session() as db:
                from sqlalchemy import func as sql_func
                
                # pylint: disable=not-callable
                results = db.query(
                    UsageRecord.resource_type,
                    UsageRecord.unit,
                    sql_func.sum(UsageRecord.usage_amount).label('total_usage'),
                    sql_func.sum(UsageRecord.cost).label('total_cost'),
                    sql_func.count(UsageRecord.id).label('record_count')
                ).filter(
                    UsageRecord.user_id == user_id,
                    UsageRecord.start_time >= start_time,
                    UsageRecord.end_time <= end_time
                ).group_by(UsageRecord.resource_type, UsageRecord.unit).all()
                
                summary = {}
                total_cost = Decimal('0')
                
                for row in results:
                    summary[row.resource_type] = {
                        'usage_amount': float(row.total_usage or 0),
                        'cost': float(row.total_cost or 0),
                        'unit': row.unit,
                        'record_count': row.record_count
                    }
                    total_cost += (row.total_cost or Decimal('0'))
                
                return {
                    'user_id': user_id,
                    'period_start': start_time.isoformat(),
                    'period_end': end_time.isoformat(),
                    'total_cost': float(total_cost),
                    'by_resource_type': summary
                }
                
        except Exception as e:
            logger.error(f"Failed to get usage summary: {e}")
            return {}


class InvoiceRepository:
    """发票数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化发票仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._invoices: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._invoices: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, invoice_data: Dict[str, Any]) -> Invoice:
        """创建发票"""
        try:
            invoice_id = invoice_data.get('id') or _generate_id()
            invoice_number = invoice_data.get('invoice_number') or _generate_invoice_number()
            
            if self._use_memory_storage:
                invoice_data['id'] = invoice_id
                invoice_data['invoice_number'] = invoice_number
                invoice_data['created_at'] = datetime.utcnow()
                invoice_data['updated_at'] = datetime.utcnow()
                self._invoices[invoice_id] = invoice_data
                return invoice_data
            
            with self._db_manager.get_db_session() as db:
                billing_address = invoice_data.get('billing_address', {})
                if isinstance(billing_address, dict):
                    billing_address = json.dumps(billing_address)
                
                extra_data = invoice_data.get('extra_data', invoice_data.get('metadata', {}))
                if isinstance(extra_data, dict):
                    extra_data = json.dumps(extra_data)
                
                invoice = Invoice(
                    id=invoice_id,
                    invoice_number=invoice_number,
                    tenant_id=invoice_data.get('tenant_id'),
                    user_id=invoice_data['user_id'],
                    billing_period_start=invoice_data['billing_period_start'],
                    billing_period_end=invoice_data['billing_period_end'],
                    subtotal=Decimal(str(invoice_data.get('subtotal', 0))),
                    discount_amount=Decimal(str(invoice_data.get('discount_amount', 0))),
                    tax_amount=Decimal(str(invoice_data.get('tax_amount', 0))),
                    total_amount=Decimal(str(invoice_data.get('total_amount', 0))),
                    currency=invoice_data.get('currency', 'CNY'),
                    status=invoice_data.get('status', 'draft'),
                    payment_status=invoice_data.get('payment_status', 'pending'),
                    due_date=invoice_data.get('due_date'),
                    paid_date=invoice_data.get('paid_date'),
                    notes=invoice_data.get('notes'),
                    billing_address=billing_address,
                    extra_data=extra_data
                )
                db.add(invoice)
                db.commit()
                db.refresh(invoice)
                return invoice
                
        except Exception as e:
            logger.error(f"Failed to create invoice: {e}")
            raise DatabaseError(f"Failed to create invoice: {e}", operation="create_invoice")
    
    def get_by_id(self, invoice_id: str) -> Optional[Invoice]:
        """根据ID获取发票"""
        try:
            if self._use_memory_storage:
                return self._invoices.get(invoice_id)
            
            with self._db_manager.get_db_session() as db:
                return db.query(Invoice).filter(Invoice.id == invoice_id).first()
                
        except Exception as e:
            logger.error(f"Failed to get invoice: {e}")
            return None
    
    def get_by_number(self, invoice_number: str) -> Optional[Invoice]:
        """根据编号获取发票"""
        try:
            if self._use_memory_storage:
                for invoice in self._invoices.values():
                    if invoice.get('invoice_number') == invoice_number:
                        return invoice
                return None
            
            with self._db_manager.get_db_session() as db:
                return db.query(Invoice).filter(Invoice.invoice_number == invoice_number).first()
                
        except Exception as e:
            logger.error(f"Failed to get invoice by number: {e}")
            return None
    
    def get_by_user(self, user_id: str, status: Optional[str] = None,
                   payment_status: Optional[str] = None,
                   limit: int = 100, offset: int = 0) -> Tuple[List[Invoice], int]:
        """获取用户的发票"""
        try:
            if self._use_memory_storage:
                invoices = [i for i in self._invoices.values() if i.get('user_id') == user_id]
                if status:
                    invoices = [i for i in invoices if i.get('status') == status]
                if payment_status:
                    invoices = [i for i in invoices if i.get('payment_status') == payment_status]
                total = len(invoices)
                return invoices[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(Invoice).filter(Invoice.user_id == user_id)
                if status:
                    query = query.filter(Invoice.status == status)
                if payment_status:
                    query = query.filter(Invoice.payment_status == payment_status)
                
                total = query.count()
                invoices = query.order_by(Invoice.created_at.desc()).offset(offset).limit(limit).all()
                return invoices, total
                
        except Exception as e:
            logger.error(f"Failed to get invoices by user: {e}")
            return [], 0
    
    def get_overdue(self) -> List[Invoice]:
        """获取逾期发票"""
        try:
            now = datetime.utcnow()
            
            if self._use_memory_storage:
                return [
                    i for i in self._invoices.values()
                    if i.get('due_date') and i.get('due_date') < now
                    and i.get('payment_status') not in ('paid', 'refunded', 'cancelled')
                ]
            
            with self._db_manager.get_db_session() as db:
                return db.query(Invoice).filter(
                    Invoice.due_date < now,
                    ~Invoice.payment_status.in_(['paid', 'refunded', 'cancelled'])
                ).all()
                
        except Exception as e:
            logger.error(f"Failed to get overdue invoices: {e}")
            return []
    
    def update(self, invoice_id: str, update_data: Dict[str, Any]) -> Optional[Invoice]:
        """更新发票"""
        try:
            if self._use_memory_storage:
                if invoice_id in self._invoices:
                    self._invoices[invoice_id].update(update_data)
                    self._invoices[invoice_id]['updated_at'] = datetime.utcnow()
                    return self._invoices[invoice_id]
                return None
            
            with self._db_manager.get_db_session() as db:
                invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
                if not invoice:
                    return None
                
                for key, value in update_data.items():
                    if key in ('billing_address', 'extra_data', 'metadata') and isinstance(value, dict):
                        value = json.dumps(value)
                    if key == 'metadata':
                        key = 'extra_data'
                    if key in ('subtotal', 'discount_amount', 'tax_amount', 'total_amount') and value is not None:
                        value = Decimal(str(value))
                    if hasattr(invoice, key):
                        setattr(invoice, key, value)
                
                invoice.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(invoice)
                return invoice
                
        except Exception as e:
            logger.error(f"Failed to update invoice: {e}")
            raise DatabaseError(f"Failed to update invoice: {e}", operation="update_invoice")
    
    def mark_as_paid(self, invoice_id: str) -> Optional[Invoice]:
        """标记发票为已支付"""
        return self.update(invoice_id, {
            'status': 'paid',
            'payment_status': 'paid',
            'paid_date': datetime.utcnow()
        })
    
    def delete(self, invoice_id: str) -> bool:
        """删除发票"""
        try:
            if self._use_memory_storage:
                if invoice_id in self._invoices:
                    del self._invoices[invoice_id]
                    return True
                return False
            
            with self._db_manager.get_db_session() as db:
                invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
                if not invoice:
                    return False
                
                db.delete(invoice)
                db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete invoice: {e}")
            return False


class BillingItemRepository:
    """计费项目数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化计费项目仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._items: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._items: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, item_data: Dict[str, Any]) -> BillingItem:
        """创建计费项目"""
        try:
            item_id = item_data.get('id') or _generate_id()
            
            if self._use_memory_storage:
                item_data['id'] = item_id
                item_data['created_at'] = datetime.utcnow()
                item_data['updated_at'] = datetime.utcnow()
                self._items[item_id] = item_data
                return item_data
            
            with self._db_manager.get_db_session() as db:
                extra_data = item_data.get('extra_data', item_data.get('metadata', {}))
                if isinstance(extra_data, dict):
                    extra_data = json.dumps(extra_data)
                
                item = BillingItem(
                    id=item_id,
                    invoice_id=item_data['invoice_id'],
                    usage_record_id=item_data.get('usage_record_id'),
                    description=item_data['description'],
                    resource_type=item_data.get('resource_type'),
                    quantity=Decimal(str(item_data['quantity'])),
                    unit=item_data.get('unit', 'unit'),
                    unit_price=Decimal(str(item_data['unit_price'])),
                    subtotal=Decimal(str(item_data['subtotal'])),
                    discount_amount=Decimal(str(item_data.get('discount_amount', 0))),
                    tax_amount=Decimal(str(item_data.get('tax_amount', 0))),
                    total_amount=Decimal(str(item_data['total_amount'])),
                    period_start=item_data.get('period_start'),
                    period_end=item_data.get('period_end'),
                    extra_data=extra_data
                )
                db.add(item)
                db.commit()
                db.refresh(item)
                return item
                
        except Exception as e:
            logger.error(f"Failed to create billing item: {e}")
            raise DatabaseError(f"Failed to create billing item: {e}", operation="create_billing_item")
    
    def get_by_invoice(self, invoice_id: str) -> List[BillingItem]:
        """获取发票的计费项目"""
        try:
            if self._use_memory_storage:
                return [i for i in self._items.values() if i.get('invoice_id') == invoice_id]
            
            with self._db_manager.get_db_session() as db:
                return db.query(BillingItem).filter(BillingItem.invoice_id == invoice_id).all()
                
        except Exception as e:
            logger.error(f"Failed to get billing items by invoice: {e}")
            return []
    
    def delete_by_invoice(self, invoice_id: str) -> int:
        """删除发票的所有计费项目"""
        try:
            if self._use_memory_storage:
                to_delete = [k for k, v in self._items.items() if v.get('invoice_id') == invoice_id]
                for key in to_delete:
                    del self._items[key]
                return len(to_delete)
            
            with self._db_manager.get_db_session() as db:
                count = db.query(BillingItem).filter(BillingItem.invoice_id == invoice_id).delete()
                db.commit()
                return count
                
        except Exception as e:
            logger.error(f"Failed to delete billing items: {e}")
            return 0


class PaymentRepository:
    """支付记录数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化支付仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._payments: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._payments: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, payment_data: Dict[str, Any]) -> Payment:
        """创建支付记录"""
        try:
            payment_id = payment_data.get('id') or _generate_id()
            payment_number = payment_data.get('payment_number') or _generate_payment_number()
            
            if self._use_memory_storage:
                payment_data['id'] = payment_id
                payment_data['payment_number'] = payment_number
                payment_data['created_at'] = datetime.utcnow()
                payment_data['updated_at'] = datetime.utcnow()
                self._payments[payment_id] = payment_data
                return payment_data
            
            with self._db_manager.get_db_session() as db:
                gateway_response = payment_data.get('gateway_response', {})
                if isinstance(gateway_response, dict):
                    gateway_response = json.dumps(gateway_response)
                
                extra_data = payment_data.get('extra_data', payment_data.get('metadata', {}))
                if isinstance(extra_data, dict):
                    extra_data = json.dumps(extra_data)
                
                payment = Payment(
                    id=payment_id,
                    payment_number=payment_number,
                    tenant_id=payment_data.get('tenant_id'),
                    invoice_id=payment_data['invoice_id'],
                    user_id=payment_data['user_id'],
                    amount=Decimal(str(payment_data['amount'])),
                    currency=payment_data.get('currency', 'CNY'),
                    payment_method=payment_data['payment_method'],
                    transaction_id=payment_data.get('transaction_id'),
                    status=payment_data.get('status', 'pending'),
                    payment_date=payment_data.get('payment_date'),
                    failure_reason=payment_data.get('failure_reason'),
                    refund_amount=Decimal(str(payment_data.get('refund_amount', 0))),
                    refund_date=payment_data.get('refund_date'),
                    refund_reason=payment_data.get('refund_reason'),
                    gateway_response=gateway_response,
                    extra_data=extra_data
                )
                db.add(payment)
                db.commit()
                db.refresh(payment)
                return payment
                
        except Exception as e:
            logger.error(f"Failed to create payment: {e}")
            raise DatabaseError(f"Failed to create payment: {e}", operation="create_payment")
    
    def get_by_id(self, payment_id: str) -> Optional[Payment]:
        """根据ID获取支付记录"""
        try:
            if self._use_memory_storage:
                return self._payments.get(payment_id)
            
            with self._db_manager.get_db_session() as db:
                return db.query(Payment).filter(Payment.id == payment_id).first()
                
        except Exception as e:
            logger.error(f"Failed to get payment: {e}")
            return None
    
    def get_by_invoice(self, invoice_id: str) -> List[Payment]:
        """获取发票的支付记录"""
        try:
            if self._use_memory_storage:
                return [p for p in self._payments.values() if p.get('invoice_id') == invoice_id]
            
            with self._db_manager.get_db_session() as db:
                return db.query(Payment).filter(Payment.invoice_id == invoice_id).all()
                
        except Exception as e:
            logger.error(f"Failed to get payments by invoice: {e}")
            return []
    
    def get_by_user(self, user_id: str, status: Optional[str] = None,
                   limit: int = 100, offset: int = 0) -> Tuple[List[Payment], int]:
        """获取用户的支付记录"""
        try:
            if self._use_memory_storage:
                payments = [p for p in self._payments.values() if p.get('user_id') == user_id]
                if status:
                    payments = [p for p in payments if p.get('status') == status]
                total = len(payments)
                return payments[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(Payment).filter(Payment.user_id == user_id)
                if status:
                    query = query.filter(Payment.status == status)
                
                total = query.count()
                payments = query.order_by(Payment.created_at.desc()).offset(offset).limit(limit).all()
                return payments, total
                
        except Exception as e:
            logger.error(f"Failed to get payments by user: {e}")
            return [], 0
    
    def update(self, payment_id: str, update_data: Dict[str, Any]) -> Optional[Payment]:
        """更新支付记录"""
        try:
            if self._use_memory_storage:
                if payment_id in self._payments:
                    self._payments[payment_id].update(update_data)
                    self._payments[payment_id]['updated_at'] = datetime.utcnow()
                    return self._payments[payment_id]
                return None
            
            with self._db_manager.get_db_session() as db:
                payment = db.query(Payment).filter(Payment.id == payment_id).first()
                if not payment:
                    return None
                
                for key, value in update_data.items():
                    if key in ('gateway_response', 'extra_data', 'metadata') and isinstance(value, dict):
                        value = json.dumps(value)
                    if key == 'metadata':
                        key = 'extra_data'
                    if key in ('amount', 'refund_amount') and value is not None:
                        value = Decimal(str(value))
                    if hasattr(payment, key):
                        setattr(payment, key, value)
                
                payment.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(payment)
                return payment
                
        except Exception as e:
            logger.error(f"Failed to update payment: {e}")
            raise DatabaseError(f"Failed to update payment: {e}", operation="update_payment")
    
    def mark_as_paid(self, payment_id: str, transaction_id: str = None) -> Optional[Payment]:
        """标记支付为已完成"""
        update_data = {
            'status': 'paid',
            'payment_date': datetime.utcnow()
        }
        if transaction_id:
            update_data['transaction_id'] = transaction_id
        return self.update(payment_id, update_data)


class WalletRepository:
    """钱包数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化钱包仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._wallets: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._wallets: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, wallet_data: Dict[str, Any]) -> Wallet:
        """创建钱包"""
        try:
            wallet_id = wallet_data.get('id') or _generate_id()
            
            if self._use_memory_storage:
                wallet_data['id'] = wallet_id
                wallet_data['created_at'] = datetime.utcnow()
                wallet_data['updated_at'] = datetime.utcnow()
                self._wallets[wallet_id] = wallet_data
                return wallet_data
            
            with self._db_manager.get_db_session() as db:
                extra_data = wallet_data.get('extra_data', wallet_data.get('metadata', {}))
                if isinstance(extra_data, dict):
                    extra_data = json.dumps(extra_data)
                
                wallet = Wallet(
                    id=wallet_id,
                    tenant_id=wallet_data.get('tenant_id'),
                    user_id=wallet_data['user_id'],
                    balance=Decimal(str(wallet_data.get('balance', 0))),
                    frozen_balance=Decimal(str(wallet_data.get('frozen_balance', 0))),
                    currency=wallet_data.get('currency', 'CNY'),
                    credit_limit=Decimal(str(wallet_data.get('credit_limit', 0))),
                    status=wallet_data.get('status', 'active'),
                    extra_data=extra_data
                )
                db.add(wallet)
                db.commit()
                db.refresh(wallet)
                return wallet
                
        except Exception as e:
            logger.error(f"Failed to create wallet: {e}")
            raise DatabaseError(f"Failed to create wallet: {e}", operation="create_wallet")
    
    def get_by_user(self, user_id: str) -> Optional[Wallet]:
        """根据用户ID获取钱包"""
        try:
            if self._use_memory_storage:
                for wallet in self._wallets.values():
                    if wallet.get('user_id') == user_id:
                        return wallet
                return None
            
            with self._db_manager.get_db_session() as db:
                return db.query(Wallet).filter(Wallet.user_id == user_id).first()
                
        except Exception as e:
            logger.error(f"Failed to get wallet by user: {e}")
            return None
    
    def get_or_create(self, user_id: str, tenant_id: str = None) -> Wallet:
        """获取或创建钱包"""
        wallet = self.get_by_user(user_id)
        if not wallet:
            wallet = self.create({
                'user_id': user_id,
                'tenant_id': tenant_id
            })
        return wallet
    
    def update_balance(self, user_id: str, amount: Decimal, operation: str = 'add') -> Optional[Wallet]:
        """更新钱包余额
        
        Args:
            user_id: 用户ID
            amount: 金额
            operation: 操作类型 ('add'增加, 'subtract'减少, 'set'设置)
        """
        try:
            wallet = self.get_by_user(user_id)
            if not wallet:
                return None
            
            if self._use_memory_storage:
                current_balance = Decimal(str(wallet.get('balance', 0)))
                if operation == 'add':
                    wallet['balance'] = float(current_balance + amount)
                elif operation == 'subtract':
                    wallet['balance'] = float(current_balance - amount)
                elif operation == 'set':
                    wallet['balance'] = float(amount)
                wallet['updated_at'] = datetime.utcnow()
                wallet['last_transaction_at'] = datetime.utcnow()
                return wallet
            
            with self._db_manager.get_db_session() as db:
                db_wallet = db.query(Wallet).filter(Wallet.user_id == user_id).first()
                if not db_wallet:
                    return None
                
                if operation == 'add':
                    db_wallet.balance = db_wallet.balance + amount
                elif operation == 'subtract':
                    db_wallet.balance = db_wallet.balance - amount
                elif operation == 'set':
                    db_wallet.balance = amount
                
                db_wallet.updated_at = datetime.utcnow()
                db_wallet.last_transaction_at = datetime.utcnow()
                db.commit()
                db.refresh(db_wallet)
                return db_wallet
                
        except Exception as e:
            logger.error(f"Failed to update wallet balance: {e}")
            return None


class WalletTransactionRepository:
    """钱包交易记录数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化钱包交易仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._transactions: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._transactions: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, transaction_data: Dict[str, Any]) -> WalletTransaction:
        """创建钱包交易记录"""
        try:
            transaction_id = transaction_data.get('id') or _generate_id()
            
            if self._use_memory_storage:
                transaction_data['id'] = transaction_id
                transaction_data['created_at'] = datetime.utcnow()
                transaction_data['updated_at'] = datetime.utcnow()
                self._transactions[transaction_id] = transaction_data
                return transaction_data
            
            with self._db_manager.get_db_session() as db:
                extra_data = transaction_data.get('extra_data', transaction_data.get('metadata', {}))
                if isinstance(extra_data, dict):
                    extra_data = json.dumps(extra_data)
                
                transaction = WalletTransaction(
                    id=transaction_id,
                    tenant_id=transaction_data.get('tenant_id'),
                    wallet_id=transaction_data['wallet_id'],
                    user_id=transaction_data['user_id'],
                    transaction_type=transaction_data['transaction_type'],
                    amount=Decimal(str(transaction_data['amount'])),
                    balance_before=Decimal(str(transaction_data['balance_before'])),
                    balance_after=Decimal(str(transaction_data['balance_after'])),
                    reference_type=transaction_data.get('reference_type'),
                    reference_id=transaction_data.get('reference_id'),
                    description=transaction_data.get('description'),
                    status=transaction_data.get('status', 'completed'),
                    extra_data=extra_data
                )
                db.add(transaction)
                db.commit()
                db.refresh(transaction)
                return transaction
                
        except Exception as e:
            logger.error(f"Failed to create wallet transaction: {e}")
            raise DatabaseError(f"Failed to create wallet transaction: {e}", operation="create_wallet_transaction")
    
    def get_by_wallet(self, wallet_id: str, limit: int = 100, offset: int = 0) -> Tuple[List[WalletTransaction], int]:
        """获取钱包的交易记录"""
        try:
            if self._use_memory_storage:
                transactions = [t for t in self._transactions.values() if t.get('wallet_id') == wallet_id]
                total = len(transactions)
                return transactions[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(WalletTransaction).filter(WalletTransaction.wallet_id == wallet_id)
                total = query.count()
                transactions = query.order_by(WalletTransaction.created_at.desc()).offset(offset).limit(limit).all()
                return transactions, total
                
        except Exception as e:
            logger.error(f"Failed to get wallet transactions: {e}")
            return [], 0
    
    def get_by_user(self, user_id: str, transaction_type: Optional[str] = None,
                   limit: int = 100, offset: int = 0) -> Tuple[List[WalletTransaction], int]:
        """获取用户的交易记录"""
        try:
            if self._use_memory_storage:
                transactions = [t for t in self._transactions.values() if t.get('user_id') == user_id]
                if transaction_type:
                    transactions = [t for t in transactions if t.get('transaction_type') == transaction_type]
                total = len(transactions)
                return transactions[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(WalletTransaction).filter(WalletTransaction.user_id == user_id)
                if transaction_type:
                    query = query.filter(WalletTransaction.transaction_type == transaction_type)
                
                total = query.count()
                transactions = query.order_by(WalletTransaction.created_at.desc()).offset(offset).limit(limit).all()
                return transactions, total
                
        except Exception as e:
            logger.error(f"Failed to get user transactions: {e}")
            return [], 0

