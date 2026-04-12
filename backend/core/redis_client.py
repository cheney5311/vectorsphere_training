"""Redis客户端模块

提供Redis连接和操作功能，适配新架构需求。
"""

import logging
from typing import Optional, Any

import redis
from flask import Flask

logger = logging.getLogger(__name__)

# 全局Redis客户端实例
_redis_client: Optional[redis.Redis] = None


def init_redis(app: Flask) -> None:
    """初始化Redis客户端
    
    Args:
        app: Flask应用实例
    """
    global _redis_client
    
    try:
        redis_url = app.config.get('REDIS_URL', 'redis://localhost:6379/0')
        _redis_client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
            max_connections=100
        )
        
        # 测试连接
        _redis_client.ping()
        logger.info("Redis客户端初始化成功")
        
    except Exception as e:
        logger.error(f"Redis客户端初始化失败: {e}")
        # Redis连接失败时，自动降级到本地缓存，确保应用能正常启动
        logger.warning("Redis服务不可用，自动降级到本地缓存模式")
        _redis_client = MockRedisClient()


def get_redis_client() -> redis.Redis:
    """获取Redis客户端实例
    
    Returns:
        Redis客户端实例
    """
    if _redis_client is None:
        raise RuntimeError("Redis客户端未初始化")
    return _redis_client


class MockRedisClient:
    """模拟Redis客户端，用于开发环境"""
    
    def __init__(self):
        self._data = {}
        logger.info("使用模拟Redis客户端")
    
    def ping(self):
        """模拟ping操作"""
        return True
    
    def get(self, key: str) -> Optional[str]:
        """模拟get操作"""
        return self._data.get(key)
    
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """模拟set操作"""
        self._data[key] = str(value)
        return True
    
    def delete(self, *keys: str) -> int:
        """模拟delete操作"""
        count = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                count += 1
        return count
    
    def exists(self, key: str) -> bool:
        """模拟exists操作"""
        return key in self._data
    
    def expire(self, key: str, time: int) -> bool:
        """模拟expire操作"""
        return key in self._data
    
    def hget(self, name: str, key: str) -> Optional[str]:
        """模拟hget操作"""
        hash_data = self._data.get(name, {})
        if isinstance(hash_data, dict):
            return hash_data.get(key)
        return None
    
    def hset(self, name: str, key: str, value: Any) -> int:
        """模拟hset操作"""
        if name not in self._data:
            self._data[name] = {}
        if not isinstance(self._data[name], dict):
            self._data[name] = {}
        self._data[name][key] = str(value)
        return 1
    
    def hdel(self, name: str, *keys: str) -> int:
        """模拟hdel操作"""
        if name not in self._data or not isinstance(self._data[name], dict):
            return 0
        count = 0
        for key in keys:
            if key in self._data[name]:
                del self._data[name][key]
                count += 1
        return count
    
    def hgetall(self, name: str) -> dict:
        """模拟hgetall操作"""
        hash_data = self._data.get(name, {})
        if isinstance(hash_data, dict):
            return hash_data
        return {}
    
    def lpush(self, name: str, *values: Any) -> int:
        """模拟lpush操作"""
        if name not in self._data:
            self._data[name] = []
        if not isinstance(self._data[name], list):
            self._data[name] = []
        for value in values:
            self._data[name].insert(0, str(value))
        return len(self._data[name])
    
    def rpop(self, name: str) -> Optional[str]:
        """模拟rpop操作"""
        if name not in self._data or not isinstance(self._data[name], list):
            return None
        if self._data[name]:
            return self._data[name].pop()
        return None
    
    def llen(self, name: str) -> int:
        """模拟llen操作"""
        if name not in self._data or not isinstance(self._data[name], list):
            return 0
        return len(self._data[name])


class RedisManager:
    """Redis管理器"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        
    def get_cached_data(self, key: str, default: Any = None) -> Any:
        """获取缓存数据"""
        try:
            value = self.redis.get(key)
            return value if value is not None else default
        except Exception as e:
            logger.error(f"获取缓存数据失败: {e}")
            return default
    
    def set_cached_data(self, key: str, value: Any, expire: int = 3600) -> bool:
        """设置缓存数据"""
        try:
            return self.redis.set(key, value, ex=expire)
        except Exception as e:
            logger.error(f"设置缓存数据失败: {e}")
            return False
    
    def delete_cached_data(self, *keys: str) -> int:
        """删除缓存数据"""
        try:
            return self.redis.delete(*keys)
        except Exception as e:
            logger.error(f"删除缓存数据失败: {e}")
            return 0
    
    def increment_counter(self, key: str, amount: int = 1, expire: int = 3600) -> int:
        """增加计数器"""
        try:
            pipe = self.redis.pipeline()
            pipe.incrby(key, amount)
            pipe.expire(key, expire)
            result = pipe.execute()
            return result[0]
        except Exception as e:
            logger.error(f"增加计数器失败: {e}")
            return 0
    
    def add_to_set(self, key: str, *values: str) -> int:
        """添加到集合"""
        try:
            return self.redis.sadd(key, *values)
        except Exception as e:
            logger.error(f"添加到集合失败: {e}")
            return 0
    
    def is_member_of_set(self, key: str, value: str) -> bool:
        """检查是否为集合成员"""
        try:
            return self.redis.sismember(key, value)
        except Exception as e:
            logger.error(f"检查集合成员失败: {e}")
            return False


def get_redis_manager() -> RedisManager:
    """获取Redis管理器实例"""
    redis_client = get_redis_client()
    return RedisManager(redis_client)