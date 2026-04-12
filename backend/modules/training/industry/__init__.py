# -*- coding: utf-8 -*-
"""
行业模型模块

提供行业模型的统一抽象和实现：
- IndustryModel: 行业模型基类
- ManufacturingModel: 制造业模型
- FinanceModel: 金融模型
- HealthcareModel: 医疗模型
"""

from .industry_model import (
    IndustryModel,
    IndustryModelConfig,
    IndustryType,
    ModalityType,
    ModalityAdapter,
    TextAdapter,
    TableAdapter,
    ImageAdapter,
    TimeSeriesAdapter,
    ScenarioHead,
    ModalityFusion,
    ManufacturingModel,
    FinanceModel,
    HealthcareModel
)

__all__ = [
    'IndustryModel',
    'IndustryModelConfig',
    'IndustryType',
    'ModalityType',
    'ModalityAdapter',
    'TextAdapter',
    'TableAdapter',
    'ImageAdapter',
    'TimeSeriesAdapter',
    'ScenarioHead',
    'ModalityFusion',
    'ManufacturingModel',
    'FinanceModel',
    'HealthcareModel'
]


def create_industry_model(industry_type: str, **kwargs):
    """
    创建行业模型的工厂函数
    
    Args:
        industry_type: 行业类型
        **kwargs: 模型配置参数
    
    Returns:
        行业模型实例
    """
    model_map = {
        'manufacturing': ManufacturingModel,
        'finance': FinanceModel,
        'healthcare': HealthcareModel,
        'general': IndustryModel
    }
    
    if industry_type not in model_map:
        raise ValueError(f"Unknown industry type: {industry_type}")
    
    model_class = model_map[industry_type]
    
    if kwargs:
        config = IndustryModelConfig(**kwargs)
        return model_class(config)
    else:
        return model_class()

