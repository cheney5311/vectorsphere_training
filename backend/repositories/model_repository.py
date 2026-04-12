from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import sys
import os
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.utils.validation import validate_id
from backend.schemas.model import Model
from backend.modules.model.exceptions.model_exceptions import ModelNotFoundError
from backend.modules.database.service import DatabaseService
from backend.schemas.model_models import ModelDB
from backend.schemas.enums import ModelType, ModelFramework, ModelStatus

logger = logging.getLogger(__name__)

class ModelRepository:
    """模型仓库"""
    
    def __init__(self):
        """初始化模型仓库"""
        self.db_service = DatabaseService()
        
    def _to_dataclass(self, model_db: ModelDB) -> Model:
        """将 SQLAlchemy 模型转换为 dataclass
        
        Args:
            model_db: SQLAlchemy 模型实例
            
        Returns:
            Model: dataclass 模型实例
        """
        # 在会话内访问所有需要的属性，避免延迟加载问题
        return Model(
            id=str(model_db.id),
            user_id=model_db.user_id,
            name=model_db.name,
            description=model_db.description,
            version=model_db.version,
            model_type=getattr(model_db, 'model_type', None),
            framework=getattr(model_db, 'framework', None),
            status=getattr(model_db, 'status', None),
            storage_path=getattr(model_db, 'storage_path', None),
            config=model_db.config or {},
            training_session_id=getattr(model_db, 'training_session_id', None),
            created_at=model_db.created_at,
            updated_at=model_db.updated_at
        )
        
    def _to_sqlalchemy(self, model: Model, model_db: Optional[ModelDB] = None) -> ModelDB:
        """将 dataclass 转换为 SQLAlchemy 模型
        
        Args:
            model: dataclass 模型实例
            model_db: 可选的现有 SQLAlchemy 模型实例（用于更新）
            
        Returns:
            ModelDB: SQLAlchemy 模型实例
        """
        if model_db is None:
            model_db = ModelDB()
            if model.id:
                model_db.id = uuid.UUID(model.id) if isinstance(model.id, str) else model.id
        
        model_db.user_id = model.user_id
        model_db.name = model.name
        model_db.description = model.description
        model_db.version = model.version
        model_db.config = model.config
        
        # 设置默认值
        if not hasattr(model_db, 'model_type') or model_db.model_type is None:
            model_db.model_type = ModelType.NLP
        if not hasattr(model_db, 'framework') or model_db.framework is None:
            model_db.framework = ModelFramework.PYTORCH
        if not hasattr(model_db, 'status') or model_db.status is None:
            model_db.status = ModelStatus.DRAFT
        if not hasattr(model_db, 'tenant_id') or model_db.tenant_id is None:
            model_db.tenant_id = "default"
            
        return model_db
        
    def create(self, model: Model) -> Model:
        """创建模型
        
        Args:
            model: 模型对象
            
        Returns:
            Model: 创建的模型对象
        """
        try:
            model_db = self._to_sqlalchemy(model)
            created_model_db = self.db_service.create(model_db)
            logger.info(f"成功创建模型: {created_model_db.name} (ID: {created_model_db.id})")
            return self._to_dataclass(created_model_db)
        except Exception as e:
            logger.error(f"创建模型失败: {e}")
            raise
        
    def get_by_id(self, model_id: str) -> Optional[Model]:
        """根据ID获取模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            Model: 模型对象，如果不存在则返回None
        """
        try:
            model_db = self.db_service.get_by_id(ModelDB, model_id)
            if model_db:
                return self._to_dataclass(model_db)
            return None
        except Exception as e:
            logger.error(f"根据ID获取模型失败: {e}")
            raise
        
    def list_by_user(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Model]:
        """根据用户ID获取模型列表
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            List[Model]: 模型列表
        """
        try:
            model_dbs = self.db_service.filter_by(ModelDB, user_id=user_id)
            # 应用分页
            paginated_models = model_dbs[offset:offset + limit]
            # 确保在会话内访问所有属性
            return [self._to_dataclass(model_db) for model_db in paginated_models]
        except Exception as e:
            logger.error(f"根据用户ID获取模型列表失败: {e}")
            raise
        
    def update(self, model: Model) -> Model:
        """更新模型
        
        Args:
            model: 模型对象
            
        Returns:
            Model: 更新后的模型对象
            
        Raises:
            ModelNotFoundError: 当模型不存在时
        """
        try:
            # 先获取现有模型
            existing_model_db = self.db_service.get_by_id(ModelDB, model.id)
            if not existing_model_db:
                raise ModelNotFoundError(f"模型 {model.id} 不存在")
            
            # 更新模型数据
            updated_model_db = self._to_sqlalchemy(model, existing_model_db)
            
            # 更新数据库
            update_data = {
                'name': updated_model_db.name,
                'description': updated_model_db.description,
                'version': updated_model_db.version,
                'config': updated_model_db.config,
                'updated_at': datetime.utcnow()
            }
            
            result_model_db = self.db_service.update(existing_model_db, update_data)
            logger.info(f"成功更新模型: {result_model_db.name} (ID: {result_model_db.id})")
            return self._to_dataclass(result_model_db)
        except ModelNotFoundError:
            raise
        except Exception as e:
            logger.error(f"更新模型失败: {e}")
            raise
        
    def delete(self, model_id: str) -> bool:
        """删除模型
        
        Args:
            model_id: 模型ID
            
        Returns:
            bool: 删除成功返回True，否则返回False
        """
        try:
            model_db = self.db_service.get_by_id(ModelDB, model_id)
            if model_db:
                success = self.db_service.delete(model_db)
                if success:
                    logger.info(f"成功删除模型: {model_db.name} (ID: {model_id})")
                return success
            return False
        except Exception as e:
            logger.error(f"删除模型失败: {e}")
            raise
        
    def exists(self, model_id: str) -> bool:
        """检查模型是否存在
        
        Args:
            model_id: 模型ID
            
        Returns:
            bool: 存在返回True，否则返回False
        """
        try:
            model_db = self.db_service.get_by_id(ModelDB, model_id)
            return model_db is not None
        except Exception as e:
            logger.error(f"检查模型是否存在失败: {e}")
            return False
            
    def count_by_tenant(self, tenant_id: str) -> int:
        """统计租户模型数量
        
        Args:
            tenant_id: 租户ID
            
        Returns:
            int: 模型数量
        """
        try:
            return self.db_service.get_db_session().query(ModelDB).filter(
                ModelDB.tenant_id == tenant_id
            ).count()
        except Exception as e:
            logger.error(f"统计租户模型数量失败: {e}")
            return 0
        
    def get_all_models(self) -> List[Model]:
        """获取所有模型
        
        Returns:
            List[Model]: 所有模型的列表，如果没有模型则返回空列表
        """
        try:
            model_dbs = self.db_service.list_all(ModelDB, limit=1000)  # 设置合理的限制
            # 确保在会话内访问所有属性
            return [self._to_dataclass(model_db) for model_db in model_dbs]
        except Exception as e:
            logger.error(f"获取所有模型失败: {e}")
            return []