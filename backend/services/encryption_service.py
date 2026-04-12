# -*- coding: utf-8 -*-
"""
加密服务
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from backend.modules.security.models import EncryptionKey, EncryptionResult, EncryptionAlgorithm, KeyType


class KeyManager:
    """密钥管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.keys: Dict[str, EncryptionKey] = {}
        self.key_storage_path = config.get('key_storage_path', 'data/keys')
        
        # 确保密钥存储目录存在
        os.makedirs(self.key_storage_path, exist_ok=True)
        
        # 加载现有密钥
        self._load_keys()
    
    def generate_key(self, name: str, algorithm: EncryptionAlgorithm,
                    expires_in_days: Optional[int] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """生成新密钥
        
        Args:
            name: 密钥名称
            algorithm: 加密算法
            expires_in_days: 过期天数
            metadata: 元数据
            
        Returns:
            密钥ID
        """
        if metadata is None:
            metadata = {}
        
        key_id = self._generate_key_id()
        
        # 计算过期时间
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        # 生成密钥数据
        if algorithm == EncryptionAlgorithm.FERNET:
            key_data = Fernet.generate_key()
            key_type = KeyType.SYMMETRIC
        elif algorithm in [EncryptionAlgorithm.AES_256_GCM, EncryptionAlgorithm.AES_256_CBC]:
            key_data = secrets.token_bytes(32)  # 256 bits
            key_type = KeyType.SYMMETRIC
        elif algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
            key_size = 2048 if algorithm == EncryptionAlgorithm.RSA_2048 else 4096
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
                backend=default_backend()
            )
            key_data = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            key_type = KeyType.ASYMMETRIC_PRIVATE
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # 创建密钥对象
        key = EncryptionKey(
            id=key_id,
            name=name,
            key_type=key_type,
            algorithm=algorithm,
            key_data=key_data,
            created_at=datetime.now(),
            expires_at=expires_at,
            metadata=metadata
        )
        
        # 存储密钥
        self.keys[key_id] = key
        self._save_key(key)
        
        return key_id
    
    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """获取密钥
        
        Args:
            key_id: 密钥ID
            
        Returns:
            密钥对象
        """
        key = self.keys.get(key_id)
        if key and key.is_expired():
            logging.warning(f"Key {key_id} is expired")
            return None
        return key
    
    def list_keys(self, include_expired: bool = False) -> List[EncryptionKey]:
        """列出所有密钥
        
        Args:
            include_expired: 是否包含过期密钥
            
        Returns:
            密钥列表
        """
        keys = list(self.keys.values())
        if not include_expired:
            keys = [key for key in keys if not key.is_expired()]
        return keys
    
    def delete_key(self, key_id: str) -> bool:
        """删除密钥
        
        Args:
            key_id: 密钥ID
            
        Returns:
            是否成功
        """
        if key_id in self.keys:
            del self.keys[key_id]
            
            # 删除文件
            key_file = os.path.join(self.key_storage_path, f"{key_id}.key")
            if os.path.exists(key_file):
                os.remove(key_file)
            
            return True
        return False
    
    def rotate_key(self, old_key_id: str, expires_in_days: Optional[int] = None) -> str:
        """轮换密钥
        
        Args:
            old_key_id: 旧密钥ID
            expires_in_days: 新密钥过期天数
            
        Returns:
            新密钥ID
        """
        old_key = self.keys.get(old_key_id)
        if not old_key:
            raise ValueError(f"Key {old_key_id} not found")
        
        # 生成新密钥
        new_key_id = self.generate_key(
            name=f"{old_key.name}_rotated",
            algorithm=old_key.algorithm,
            expires_in_days=expires_in_days,
            metadata={**old_key.metadata, 'rotated_from': old_key_id}
        )
        
        # 标记旧密钥为已轮换
        old_key.metadata['rotated_to'] = new_key_id
        old_key.metadata['rotated_at'] = datetime.now().isoformat()
        self._save_key(old_key)
        
        return new_key_id
    
    def derive_key(self, password: str, salt: Optional[bytes] = None,
                  algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM) -> Tuple[bytes, bytes]:
        """派生密钥
        
        Args:
            password: 密码
            salt: 盐值
            algorithm: 算法
            
        Returns:
            (密钥, 盐值)
        """
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # 使用Scrypt进行密钥派生
        kdf = Scrypt(
            length=32,
            salt=salt,
            n=2**14,
            r=8,
            p=1,
            backend=default_backend()
        )
        
        key = kdf.derive(password.encode('utf-8'))
        return key, salt
    
    def _generate_key_id(self) -> str:
        """生成密钥ID"""
        return secrets.token_hex(16)
    
    def _save_key(self, key: EncryptionKey):
        """保存密钥到文件"""
        key_file = os.path.join(self.key_storage_path, f"{key.id}.key")
        
        key_data = {
            'id': key.id,
            'name': key.name,
            'key_type': key.key_type.value,
            'algorithm': key.algorithm.value,
            'key_data': base64.b64encode(key.key_data).decode('utf-8'),
            'created_at': key.created_at.isoformat(),
            'expires_at': key.expires_at.isoformat() if key.expires_at else None,
            'metadata': key.metadata
        }
        
        with open(key_file, 'w') as f:
            json.dump(key_data, f, indent=2)
    
    def _load_keys(self):
        """从文件加载密钥"""
        if not os.path.exists(self.key_storage_path):
            return
        
        for filename in os.listdir(self.key_storage_path):
            if filename.endswith('.key'):
                key_file = os.path.join(self.key_storage_path, filename)
                try:
                    with open(key_file, 'r') as f:
                        key_data = json.load(f)
                    
                    key = EncryptionKey(
                        id=key_data['id'],
                        name=key_data['name'],
                        key_type=KeyType(key_data['key_type']),
                        algorithm=EncryptionAlgorithm(key_data['algorithm']),
                        key_data=base64.b64decode(key_data['key_data']),
                        created_at=datetime.fromisoformat(key_data['created_at']),
                        expires_at=datetime.fromisoformat(key_data['expires_at']) if key_data['expires_at'] else None,
                        metadata=key_data['metadata']
                    )
                    
                    self.keys[key.id] = key
                    
                except Exception as e:
                    logging.error(f"Failed to load key from {key_file}: {e}")


class EncryptionService:
    """加密服务
    
    提供数据加密、解密、密钥管理等功能
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.key_manager = KeyManager(config)
        
        # 默认密钥ID
        self.default_key_id = config.get('default_key_id')
        
        # 如果没有默认密钥，创建一个
        if not self.default_key_id or not self.key_manager.get_key(self.default_key_id):
            self.default_key_id = self.key_manager.generate_key(
                name="default_encryption_key",
                algorithm=EncryptionAlgorithm.FERNET,
                expires_in_days=365
            )
    
    def encrypt(self, data: Union[str, bytes], key_id: Optional[str] = None,
               algorithm: Optional[EncryptionAlgorithm] = None) -> EncryptionResult:
        """加密数据
        
        Args:
            data: 要加密的数据
            key_id: 密钥ID
            algorithm: 加密算法
            
        Returns:
            加密结果
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if key_id is None:
            key_id = self.default_key_id
        
        if key_id is None:
            raise ValueError("No key ID provided and no default key configured")
        
        key = self.key_manager.get_key(key_id)
        if not key:
            raise ValueError(f"Key {key_id} not found or expired")
        
        if algorithm is None:
            algorithm = key.algorithm
        
        if algorithm == EncryptionAlgorithm.FERNET:
            return self._encrypt_fernet(data, key)
        elif algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._encrypt_aes_gcm(data, key)
        elif algorithm == EncryptionAlgorithm.AES_256_CBC:
            return self._encrypt_aes_cbc(data, key)
        elif algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
            return self._encrypt_rsa(data, key)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def decrypt(self, encrypted_result: Union[EncryptionResult, str]) -> bytes:
        """解密数据
        
        Args:
            encrypted_result: 加密结果或Base64字符串
            
        Returns:
            解密后的数据
        """
        if isinstance(encrypted_result, str):
            # 解析Base64字符串
            try:
                decoded = json.loads(base64.b64decode(encrypted_result).decode('utf-8'))
                
                encrypted_result = EncryptionResult(
                    encrypted_data=base64.b64decode(decoded['data']),
                    key_id=decoded['key_id'],
                    algorithm=EncryptionAlgorithm(decoded['algorithm']),
                    iv=base64.b64decode(decoded['iv']) if 'iv' in decoded else None,
                    tag=base64.b64decode(decoded['tag']) if 'tag' in decoded else None,
                    metadata=json.loads(decoded.get('metadata', '{}')) if 'metadata' in decoded and decoded['metadata'] else None
                )
            except Exception as e:
                raise ValueError(f"Invalid encryption result format: {e}")
        
        key = self.key_manager.get_key(encrypted_result.key_id)
        if not key:
            raise ValueError(f"Key {encrypted_result.key_id} not found or expired")
        
        algorithm = encrypted_result.algorithm
        
        if algorithm == EncryptionAlgorithm.FERNET:
            return self._decrypt_fernet(encrypted_result, key)
        elif algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._decrypt_aes_gcm(encrypted_result, key)
        elif algorithm == EncryptionAlgorithm.AES_256_CBC:
            return self._decrypt_aes_cbc(encrypted_result, key)
        elif algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
            return self._decrypt_rsa(encrypted_result, key)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def encrypt_field(self, data: Any, field_name: str) -> str:
        """加密字段
        
        Args:
            data: 数据
            field_name: 字段名
            
        Returns:
            加密后的Base64字符串
        """
        json_data = json.dumps(data, ensure_ascii=False)
        result = self.encrypt(json_data)
        
        # 转换为Base64字符串
        result_dict = {
            'data': base64.b64encode(result.encrypted_data).decode('utf-8'),
            'key_id': result.key_id,
            'algorithm': result.algorithm.value
        }
        
        if result.iv:
            result_dict['iv'] = base64.b64encode(result.iv).decode('utf-8')
        if result.tag:
            result_dict['tag'] = base64.b64encode(result.tag).decode('utf-8')
        if result.metadata:
            result_dict['metadata'] = json.dumps(result.metadata)
        
        return base64.b64encode(json.dumps(result_dict).encode()).decode('utf-8')
    
    def decrypt_field(self, encrypted_data: str, field_name: str) -> Any:
        """解密字段
        
        Args:
            encrypted_data: 加密的Base64字符串
            field_name: 字段名
            
        Returns:
            解密后的数据
        """
        decrypted_bytes = self.decrypt(encrypted_data)
        json_data = decrypted_bytes.decode('utf-8')
        return json.loads(json_data)
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """哈希密码
        
        Args:
            password: 密码
            salt: 盐值
            
        Returns:
            (哈希值, 盐值)
        """
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # 使用PBKDF2进行密码哈希
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = kdf.derive(password.encode('utf-8'))
        
        return (
            base64.b64encode(key).decode('utf-8'),
            base64.b64encode(salt).decode('utf-8')
        )
    
    def verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """验证密码
        
        Args:
            password: 密码
            hashed_password: 哈希值
            salt: 盐值
            
        Returns:
            是否匹配
        """
        try:
            salt_bytes = base64.b64decode(salt)
            expected_hash = base64.b64decode(hashed_password)
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt_bytes,
                iterations=100000,
                backend=default_backend()
            )
            
            kdf.verify(password.encode('utf-8'), expected_hash)
            return True
        except Exception:
            return False
    
    def generate_hmac(self, data: Union[str, bytes], key_id: Optional[str] = None) -> str:
        """生成HMAC
        
        Args:
            data: 数据
            key_id: 密钥ID
            
        Returns:
            HMAC值
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if key_id is None:
            key_id = self.default_key_id
        
        if key_id is None:
            raise ValueError("No key ID provided and no default key configured")
        
        key = self.key_manager.get_key(key_id)
        if not key:
            raise ValueError(f"Key {key_id} not found or expired")
        
        mac = hmac.new(key.key_data, data, hashlib.sha256)
        return base64.b64encode(mac.digest()).decode('utf-8')
    
    def verify_hmac(self, data: Union[str, bytes], signature: str, key_id: Optional[str] = None) -> bool:
        """验证HMAC
        
        Args:
            data: 数据
            signature: 签名
            key_id: 密钥ID
            
        Returns:
            是否有效
        """
        try:
            expected_signature = self.generate_hmac(data, key_id)
            return hmac.compare_digest(signature, expected_signature)
        except Exception:
            return False
    
    def _encrypt_fernet(self, data: bytes, key: EncryptionKey) -> EncryptionResult:
        """Fernet加密"""
        f = Fernet(key.key_data)
        encrypted_data = f.encrypt(data)
        
        return EncryptionResult(
            encrypted_data=encrypted_data,
            key_id=key.id,
            algorithm=EncryptionAlgorithm.FERNET
        )
    
    def _decrypt_fernet(self, result: EncryptionResult, key: EncryptionKey) -> bytes:
        """Fernet解密"""
        f = Fernet(key.key_data)
        return f.decrypt(result.encrypted_data)
    
    def _encrypt_aes_gcm(self, data: bytes, key: EncryptionKey) -> EncryptionResult:
        """AES-GCM加密"""
        iv = secrets.token_bytes(12)  # 96-bit IV for GCM
        
        cipher = Cipher(
            algorithms.AES(key.key_data),
            modes.GCM(iv),
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(data) + encryptor.finalize()
        
        return EncryptionResult(
            encrypted_data=encrypted_data,
            key_id=key.id,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            iv=iv,
            tag=encryptor.tag
        )
    
    def _decrypt_aes_gcm(self, result: EncryptionResult, key: EncryptionKey) -> bytes:
        """AES-GCM解密"""
        if result.iv is None or result.tag is None:
            raise ValueError("IV and tag are required for AES-GCM decryption")
        
        cipher = Cipher(
            algorithms.AES(key.key_data),
            modes.GCM(result.iv, result.tag),
            backend=default_backend()
        )
        
        decryptor = cipher.decryptor()
        return decryptor.update(result.encrypted_data) + decryptor.finalize()
    
    def _encrypt_aes_cbc(self, data: bytes, key: EncryptionKey) -> EncryptionResult:
        """AES-CBC加密"""
        # 填充数据到16字节边界
        padding_length = 16 - (len(data) % 16)
        padded_data = data + bytes([padding_length] * padding_length)
        
        iv = secrets.token_bytes(16)  # 128-bit IV for CBC
        
        cipher = Cipher(
            algorithms.AES(key.key_data),
            modes.CBC(iv),
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        return EncryptionResult(
            encrypted_data=encrypted_data,
            key_id=key.id,
            algorithm=EncryptionAlgorithm.AES_256_CBC,
            iv=iv
        )
    
    def _decrypt_aes_cbc(self, result: EncryptionResult, key: EncryptionKey) -> bytes:
        """AES-CBC解密"""
        if result.iv is None:
            raise ValueError("IV is required for AES-CBC decryption")
        
        cipher = Cipher(
            algorithms.AES(key.key_data),
            modes.CBC(result.iv),
            backend=default_backend()
        )
        
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(result.encrypted_data) + decryptor.finalize()
        
        # 移除填充
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def _encrypt_rsa(self, data: bytes, key: EncryptionKey) -> EncryptionResult:
        """RSA加密"""
        # RSA加密有长度限制，对于长数据需要分块或使用混合加密
        private_key = serialization.load_pem_private_key(
            key.key_data,
            password=None,
            backend=default_backend()
        )
        
        public_key = private_key.public_key()
        
        # 计算最大块大小
        key_size = private_key.key_size
        max_chunk_size = (key_size // 8) - 2 * (hashes.SHA256().digest_size) - 2
        
        if len(data) > max_chunk_size:
            # 对于大数据，使用混合加密（RSA + AES）
            # 生成随机AES密钥
            aes_key = secrets.token_bytes(32)
            
            # 用RSA加密AES密钥
            encrypted_aes_key = public_key.encrypt(
                aes_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # 用AES加密数据
            iv = secrets.token_bytes(12)
            cipher = Cipher(
                algorithms.AES(aes_key),
                modes.GCM(iv),
                backend=default_backend()
            )
            
            encryptor = cipher.encryptor()
            encrypted_data = encryptor.update(data) + encryptor.finalize()
            
            # 组合结果
            combined_data = encrypted_aes_key + iv + encryptor.tag + encrypted_data
            
            return EncryptionResult(
                encrypted_data=combined_data,
                key_id=key.id,
                algorithm=key.algorithm,
                metadata={'hybrid': True}
            )
        else:
            # 直接RSA加密
            encrypted_data = public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return EncryptionResult(
                encrypted_data=encrypted_data,
                key_id=key.id,
                algorithm=key.algorithm
            )
    
    def _decrypt_rsa(self, result: EncryptionResult, key: EncryptionKey) -> bytes:
        """RSA解密"""
        private_key = serialization.load_pem_private_key(
            key.key_data,
            password=None,
            backend=default_backend()
        )
        
        if result.metadata and result.metadata.get('hybrid'):
            # 混合解密
            data = result.encrypted_data
            key_size = private_key.key_size // 8
            
            # 提取组件
            encrypted_aes_key = data[:key_size]
            iv = data[key_size:key_size + 12]
            tag = data[key_size + 12:key_size + 12 + 16]
            encrypted_data = data[key_size + 12 + 16:]
            
            # 解密AES密钥
            aes_key = private_key.decrypt(
                encrypted_aes_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # 解密数据
            cipher = Cipher(
                algorithms.AES(aes_key),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
            
            decryptor = cipher.decryptor()
            return decryptor.update(encrypted_data) + decryptor.finalize()
        else:
            # 直接RSA解密
            return private_key.decrypt(
                result.encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )