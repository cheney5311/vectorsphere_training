"""认证服务实现"""

import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from backend.modules.database.manager import get_database_manager
from backend.schemas import User
from backend.modules.model.user import Token
from ..exceptions.auth_exceptions import (
    AuthenticationError, AuthorizationError, UserNotFoundError,
    InvalidTokenError, ExpiredTokenError, InvalidCredentialsError,
    UserAlreadyExistsError
)


class AuthService:
    """认证服务"""
    
    def __init__(self, db_manager=None, secret_key: str = "default_secret_key"):
        self.db_manager = db_manager or get_database_manager()
        self.secret_key = secret_key
        self.token_expiration = timedelta(hours=24)
        self.refresh_token_expiration = timedelta(days=30)
    
    def _hash_password(self, password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """哈希密码"""
        if salt is None:
            salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return password_hash.hex(), salt
    
    def _verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """验证密码"""
        hashed, _ = self._hash_password(password, salt)
        return secrets.compare_digest(hashed, password_hash)
    
    def register_user(self, username: str, email: str, password: str, 
                     full_name: Optional[str] = None, phone: Optional[str] = None) -> User:
        """注册用户"""
        with self.db_manager.get_session() as db:
            # 检查用户是否已存在
            existing_user = db.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                raise UserAlreadyExistsError("用户名或邮箱已存在")
            
            # 创建新用户
            password_hash, salt = self._hash_password(password)
            user = User(
                id=secrets.token_hex(16),
                username=username,
                email=email,
                password_hash=password_hash,
                full_name=full_name,
                phone=phone
            )
            
            # 保存到数据库
            db.add(user)
            db.commit()
            db.refresh(user)
            
            return user
    
    def authenticate_user(self, identifier: str, password: str) -> User:
        """认证用户"""
        with self.db_manager.get_session() as db:
            # 根据用户名或邮箱查找用户
            user = db.query(User).filter(
                (User.username == identifier) | (User.email == identifier)
            ).first()
            
            if not user:
                raise UserNotFoundError("用户不存在")
            
            # 使用getattr访问is_active属性以避免类型检查问题
            is_active = getattr(user, 'is_active')
            if not is_active:
                raise AuthenticationError("用户已被禁用")
            
            # 验证密码
            user_password_hash = getattr(user, 'password_hash')
            if not self._verify_password(password, user_password_hash, user.id):
                raise InvalidCredentialsError("用户名或密码错误")
            
            # 更新最后登录时间
            setattr(user, 'last_login', datetime.utcnow())
            db.commit()
            
            return user
    
    def generate_token(self, user: User) -> tuple[str, str]:
        """生成访问令牌和刷新令牌"""
        # 生成访问令牌
        access_payload = {
            'user_id': user.id,
            'username': user.username,
            'exp': datetime.utcnow() + self.token_expiration,
            'iat': datetime.utcnow()
        }
        access_token = jwt.encode(access_payload, self.secret_key, algorithm='HS256')
        
        # 生成刷新令牌
        refresh_payload = {
            'user_id': user.id,
            'exp': datetime.utcnow() + self.refresh_token_expiration,
            'iat': datetime.utcnow()
        }
        refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm='HS256')
        
        # 保存令牌到数据库
        with self.db_manager.get_session() as db:
            token = Token(
                token=refresh_token,
                user_id=user.id,
                expires_at=datetime.utcnow() + self.refresh_token_expiration
            )
            db.add(token)
            db.commit()
        
        return access_token, refresh_token
    
    def refresh_token(self, refresh_token: str) -> tuple[str, str]:
        """刷新令牌"""
        try:
            # 解码刷新令牌
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=['HS256'])
            user_id = payload.get('user_id')
            
            if not user_id:
                raise InvalidTokenError("无效的刷新令牌")
            
            # 检查令牌是否存在于数据库且未被撤销
            with self.db_manager.get_session() as db:
                # 使用getattr访问Token属性以避免类型检查问题
                token = db.query(Token).filter(
                    getattr(Token, 'token') == refresh_token,
                    getattr(Token, 'user_id') == user_id,
                    getattr(Token, 'revoked').is_(False)
                ).first()
                
                if not token:
                    raise InvalidTokenError("刷新令牌不存在或已被撤销")
                
                if getattr(token, 'expires_at') < datetime.utcnow():
                    raise ExpiredTokenError("刷新令牌已过期")
                
                # 获取用户信息
                user = db.query(User).filter(User.id == user_id).first()
                if not user:
                    raise UserNotFoundError("用户不存在")
                
                # 生成新的访问令牌和刷新令牌
                return self.generate_token(user)
        except jwt.ExpiredSignatureError:
            raise ExpiredTokenError("刷新令牌已过期")
        except jwt.InvalidTokenError:
            raise InvalidTokenError("无效的刷新令牌")
    
    def revoke_token(self, refresh_token: str) -> bool:
        """撤销令牌"""
        with self.db_manager.get_session() as db:
            # 使用getattr访问Token属性以避免类型检查问题
            token = db.query(Token).filter(getattr(Token, 'token') == refresh_token).first()
            if token:
                setattr(token, 'revoked', True)
                db.commit()
                return True
            return False
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """根据ID获取用户"""
        with self.db_manager.get_session() as db:
            return db.query(User).filter(User.id == user_id).first()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        with self.db_manager.get_session() as db:
            return db.query(User).filter(User.username == username).first()
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """验证令牌"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise ExpiredTokenError("访问令牌已过期")
        except jwt.InvalidTokenError:
            raise InvalidTokenError("无效的访问令牌")