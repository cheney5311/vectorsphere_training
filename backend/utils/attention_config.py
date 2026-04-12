"""注意力机制配置工具

提供注意力机制相关的配置和优化工具。
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class AttentionConfig:
    """注意力机制配置类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    def get_attention_type(self) -> str:
        """获取注意力类型"""
        return self.config.get('type', 'standard')
    
    def get_num_heads(self) -> int:
        """获取注意力头数"""
        return self.config.get('num_heads', 8)
    
    def get_head_dim(self) -> int:
        """获取每个头的维度"""
        return self.config.get('head_dim', 64)
    
    def get_dropout_rate(self) -> float:
        """获取dropout率"""
        return self.config.get('dropout', 0.1)
    
    def is_flash_attention_enabled(self) -> bool:
        """检查是否启用Flash Attention"""
        return self.config.get('flash_attention', False)
    
    def get_window_size(self) -> Optional[int]:
        """获取滑动窗口大小（用于局部注意力）"""
        return self.config.get('window_size')
    
    def get_sparse_pattern(self) -> Optional[str]:
        """获取稀疏注意力模式"""
        return self.config.get('sparse_pattern')
    
    def get_attention_scaling(self) -> float:
        """获取注意力缩放因子"""
        return self.config.get('scaling', 1.0)
    
    def get_position_encoding_type(self) -> str:
        """获取位置编码类型"""
        return self.config.get('position_encoding', 'absolute')
    
    def get_rope_theta(self) -> float:
        """获取RoPE编码的theta值"""
        return self.config.get('rope_theta', 10000.0)
    
    def get_alibi_slope(self) -> Optional[float]:
        """获取ALiBi斜率"""
        return self.config.get('alibi_slope')
    
    def get_attention_mask_type(self) -> str:
        """获取注意力掩码类型"""
        return self.config.get('mask_type', 'causal')
    
    def get_memory_efficient_config(self) -> Dict[str, Any]:
        """获取内存高效配置"""
        return {
            'enable_checkpointing': self.config.get('checkpointing', False),
            'enable_low_memory': self.config.get('low_memory', False),
            'gradient_checkpointing': self.config.get('gradient_checkpointing', False)
        }
    
    def validate_config(self) -> bool:
        """验证配置"""
        try:
            # 验证基本参数
            num_heads = self.get_num_heads()
            head_dim = self.get_head_dim()
            dropout = self.get_dropout_rate()
            
            if num_heads <= 0:
                raise ValueError("num_heads must be positive")
            
            if head_dim <= 0:
                raise ValueError("head_dim must be positive")
            
            if not (0.0 <= dropout <= 1.0):
                raise ValueError("dropout must be between 0 and 1")
            
            # 验证注意力类型
            attention_type = self.get_attention_type()
            valid_types = ['standard', 'multi_query', 'grouped_query', 'sparse', 'sliding_window']
            if attention_type not in valid_types:
                raise ValueError(f"Invalid attention type: {attention_type}")
            
            # 验证位置编码
            pos_encoding = self.get_position_encoding_type()
            valid_encodings = ['absolute', 'relative', 'rotary', 'alibi', 'none']
            if pos_encoding not in valid_encodings:
                raise ValueError(f"Invalid position encoding: {pos_encoding}")
            
            return True
            
        except Exception as e:
            logger.error(f"Attention config validation failed: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.config.copy()
    
    def update(self, updates: Dict[str, Any]) -> None:
        """更新配置"""
        self.config.update(updates)


class AttentionOptimizer:
    """注意力机制优化器"""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
    
    def optimize_for_sequence_length(self, seq_length: int) -> AttentionConfig:
        """根据序列长度优化配置"""
        optimized_config = self.config.to_dict()
        
        # 对于长序列，启用内存优化
        if seq_length > 1024:
            optimized_config['flash_attention'] = True
            optimized_config['checkpointing'] = True
            optimized_config['low_memory'] = True
        
        # 对于超长序列，使用滑动窗口注意力
        if seq_length > 4096:
            optimized_config['type'] = 'sliding_window'
            optimized_config['window_size'] = min(1024, seq_length // 4)
        
        return AttentionConfig(optimized_config)
    
    def optimize_for_batch_size(self, batch_size: int) -> AttentionConfig:
        """根据批处理大小优化配置"""
        optimized_config = self.config.to_dict()
        
        # 大批处理大小时启用内存优化
        if batch_size > 32:
            optimized_config['low_memory'] = True
            optimized_config['gradient_checkpointing'] = True
        
        return AttentionConfig(optimized_config)
    
    def get_recommended_config(self, model_size: str = 'base') -> AttentionConfig:
        """获取推荐配置"""
        base_config = {
            'base': {
                'num_heads': 8,
                'head_dim': 64,
                'dropout': 0.1,
                'flash_attention': True
            },
            'large': {
                'num_heads': 16,
                'head_dim': 64,
                'dropout': 0.1,
                'flash_attention': True,
                'checkpointing': True
            },
            'xl': {
                'num_heads': 32,
                'head_dim': 64,
                'dropout': 0.1,
                'flash_attention': True,
                'checkpointing': True,
                'low_memory': True,
                'gradient_checkpointing': True
            }
        }
        
        config = base_config.get(model_size, base_config['base'])
        return AttentionConfig(config)


def create_attention_config(config_dict: Optional[Dict[str, Any]] = None) -> AttentionConfig:
    """创建注意力配置实例
    
    Args:
        config_dict: 配置字典
        
    Returns:
        AttentionConfig实例
    """
    return AttentionConfig(config_dict)


def optimize_attention_config(
    base_config: AttentionConfig,
    seq_length: Optional[int] = None,
    batch_size: Optional[int] = None
) -> AttentionConfig:
    """优化注意力配置
    
    Args:
        base_config: 基础配置
        seq_length: 序列长度
        batch_size: 批处理大小
        
    Returns:
        优化后的配置
    """
    optimizer = AttentionOptimizer(base_config)
    
    if seq_length:
        base_config = optimizer.optimize_for_sequence_length(seq_length)
    
    if batch_size:
        base_config = optimizer.optimize_for_batch_size(batch_size)
    
    return base_config