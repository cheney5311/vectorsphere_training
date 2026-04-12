"""模型选择与配置服务

提供模型选择和配置相关的业务逻辑。
"""

import sys
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import random

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.core.exceptions import ValidationError, BusinessLogicError
from backend.schemas.enums import TrainingScenario, ModelType, ModelFramework
from backend.repositories.model_repository import ModelRepository

logger = logging.getLogger(__name__)


@dataclass
class ModelRecommendation:
    """模型推荐"""
    model_name: str
    framework: ModelFramework
    model_type: ModelType
    description: str
    confidence: float
    recommended_for: List[str]
    performance_metrics: Dict[str, float]


@dataclass
class ModelConfiguration:
    """模型配置"""
    model_name: str
    framework: ModelFramework
    model_type: ModelType
    hyperparameters: Dict[str, Any]
    training_config: Dict[str, Any]
    hardware_config: Dict[str, Any]


class ModelSelectionService:
    """模型选择与配置服务（支持租户隔离和持久化）"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化模型选择服务
        
        Args:
            use_memory_storage: 是否使用内存存储
        """
        self._use_memory_storage = use_memory_storage
        self._model_repository = ModelRepository()
        self._init_repositories()
        self._model_catalog = self._initialize_model_catalog()
    
    def _init_repositories(self):
        """初始化数据仓库"""
        try:
            from backend.repositories.model_selection_repository import (
                ModelRecommendationRepository,
                ModelConfigurationRepository,
                ModelCatalogRepository
            )
            self._recommendation_repository = ModelRecommendationRepository(
                use_memory_storage=self._use_memory_storage
            )
            self._configuration_repository = ModelConfigurationRepository(
                use_memory_storage=self._use_memory_storage
            )
            self._catalog_repository = ModelCatalogRepository(
                use_memory_storage=self._use_memory_storage
            )
            logger.info("Model selection repositories initialized successfully")
        except ImportError as e:
            logger.warning(f"Failed to import repositories: {e}, using memory storage")
            from backend.repositories.model_selection_repository import (
                ModelRecommendationRepository,
                ModelConfigurationRepository,
                ModelCatalogRepository
            )
            self._recommendation_repository = ModelRecommendationRepository(use_memory_storage=True)
            self._configuration_repository = ModelConfigurationRepository(use_memory_storage=True)
            self._catalog_repository = ModelCatalogRepository(use_memory_storage=True)
        
    def _initialize_model_catalog(self) -> Dict[str, Any]:
        """初始化模型目录
        
        Returns:
            模型目录
        """
        try:
            # 尝试从数据库加载模型目录
            catalog = self._load_catalog_from_database()
            if catalog:
                return catalog
        except Exception as e:
            logger.warning(f"从数据库加载模型目录失败: {e}")
        
        try:
            # 尝试从配置文件加载模型目录
            catalog = self._load_catalog_from_config()
            if catalog:
                return catalog
        except Exception as e:
            logger.warning(f"从配置文件加载模型目录失败: {e}")
        
        # 回退到预定义的模型目录
        return self._get_default_catalog()
    
    def _load_catalog_from_database(self) -> Optional[Dict[str, Any]]:
        """从数据库加载模型目录
        
        Returns:
            模型目录字典，如果加载失败返回 None
        """
        try:
            # 尝试从数据库获取模型目录
            # 这里可以根据实际的数据库结构来实现
            models = self._model_repository.get_all_models()
            if not models:
                return None
                
            catalog = {}
            for model in models:
                # 根据模型类型分组
                task_type = self._get_task_type_from_model_type(model.model_type)
                if task_type not in catalog:
                    catalog[task_type] = {}
                
                # 构建模型信息
                # 确保所有属性在会话内访问，避免延迟加载问题
                catalog[task_type][model.name] = {
                    "framework": model.framework,
                    "model_type": model.model_type,
                    "description": model.description or f"{model.name} 模型",
                    "performance": model.config.get('performance_metrics', {}),
                    "requirements": model.config.get('hardware_requirements', {})
                }
            
            logger.info(f"从数据库成功加载 {len(catalog)} 个任务类型的模型目录")
            return catalog
            
        except Exception as e:
            logger.error(f"从数据库加载模型目录时发生错误: {e}")
            return None
    
    def _load_catalog_from_config(self) -> Optional[Dict[str, Any]]:
        """从配置文件加载模型目录
        
        Returns:
            模型目录字典，如果加载失败返回 None
        """
        try:
            import json
            import yaml
            
            # 尝试加载配置文件
            config_paths = [
                os.path.join(os.path.dirname(__file__), '..', 'config', 'model_catalog.json'),
                os.path.join(os.path.dirname(__file__), '..', 'config', 'model_catalog.yaml'),
                os.path.join(os.path.dirname(__file__), '..', 'config', 'model_catalog.yml'),
                'config/model_catalog.json',
                'config/model_catalog.yaml'
            ]
            
            for config_path in config_paths:
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            if config_path.endswith('.json'):
                                catalog = json.load(f)
                            else:  # yaml/yml
                                catalog = yaml.safe_load(f)
                        
                        if catalog and isinstance(catalog, dict):
                            logger.info(f"从配置文件 {config_path} 成功加载模型目录")
                            return catalog
                            
                    except Exception as e:
                        logger.warning(f"解析配置文件 {config_path} 失败: {e}")
                        continue
            
            logger.warning("未找到有效的模型目录配置文件")
            return None
            
        except Exception as e:
            logger.error(f"从配置文件加载模型目录时发生错误: {e}")
            return None
    
    def _get_task_type_from_model_type(self, model_type: ModelType) -> str:
        """根据模型类型获取任务类型
        
        Args:
            model_type: 模型类型
            
        Returns:
            任务类型字符串
        """
        type_mapping = {
            ModelType.CLASSIFICATION: "text_classification",
            ModelType.GENERATION: "text_generation",
            ModelType.REGRESSION: "regression",
            ModelType.CLUSTERING: "clustering",
            ModelType.DETECTION: "object_detection",
            ModelType.SEGMENTATION: "image_segmentation"
        }
        return type_mapping.get(model_type, "other")
    
    def _get_default_catalog(self) -> Dict[str, Any]:
        """获取默认模型目录"""
        return {
            "text_classification": {
                "bert-base-uncased": {
                    "framework": ModelFramework.HUGGINGFACE,
                    "model_type": ModelType.CLASSIFICATION,
                    "description": "BERT基础模型，适用于文本分类任务",
                    "performance": {"accuracy": 0.87, "speed": 0.75},
                    "requirements": {"gpu_memory": "8GB", "cpu_cores": 4}
                },
                "roberta-base": {
                    "framework": ModelFramework.HUGGINGFACE,
                    "model_type": ModelType.CLASSIFICATION,
                    "description": "RoBERTa基础模型，适用于文本分类任务",
                    "performance": {"accuracy": 0.89, "speed": 0.70},
                    "requirements": {"gpu_memory": "8GB", "cpu_cores": 4}
                },
                "distilbert-base-uncased": {
                    "framework": ModelFramework.HUGGINGFACE,
                    "model_type": ModelType.CLASSIFICATION,
                    "description": "轻量级BERT模型，适用于资源受限环境",
                    "performance": {"accuracy": 0.85, "speed": 0.90},
                    "requirements": {"gpu_memory": "4GB", "cpu_cores": 2}
                }
            },
            "text_generation": {
                "gpt2": {
                    "framework": ModelFramework.HUGGINGFACE,
                    "model_type": ModelType.GENERATION,
                    "description": "GPT-2模型，适用于文本生成任务",
                    "performance": {"perplexity": 25.5, "speed": 0.65},
                    "requirements": {"gpu_memory": "12GB", "cpu_cores": 6}
                },
                "llama-2-7b": {
                    "framework": ModelFramework.HUGGINGFACE,
                    "model_type": ModelType.GENERATION,
                    "description": "Llama 2 7B模型，适用于高质量文本生成",
                    "performance": {"perplexity": 18.2, "speed": 0.55},
                    "requirements": {"gpu_memory": "16GB", "cpu_cores": 8}
                },
                "mistral-7b": {
                    "framework": ModelFramework.HUGGINGFACE,
                    "model_type": ModelType.GENERATION,
                    "description": "Mistral 7B模型，高效能文本生成",
                    "performance": {"perplexity": 15.8, "speed": 0.60},
                    "requirements": {"gpu_memory": "14GB", "cpu_cores": 6}
                }
            },
            "image_classification": {
                "resnet50": {
                    "framework": ModelFramework.PYTORCH,
                    "model_type": ModelType.CLASSIFICATION,
                    "description": "ResNet-50模型，适用于图像分类任务",
                    "performance": {"accuracy": 0.76, "speed": 0.85},
                    "requirements": {"gpu_memory": "6GB", "cpu_cores": 4}
                },
                "efficientnet-b0": {
                    "framework": ModelFramework.PYTORCH,
                    "model_type": ModelType.CLASSIFICATION,
                    "description": "EfficientNet-B0模型，轻量级图像分类",
                    "performance": {"accuracy": 0.77, "speed": 0.90},
                    "requirements": {"gpu_memory": "4GB", "cpu_cores": 2}
                },
                "vit-base": {
                    "framework": ModelFramework.PYTORCH,
                    "model_type": ModelType.CLASSIFICATION,
                    "description": "Vision Transformer基础模型",
                    "performance": {"accuracy": 0.82, "speed": 0.70},
                    "requirements": {"gpu_memory": "8GB", "cpu_cores": 4}
                }
            }
        }
        
    def recommend_models(
        self, 
        task_type: str, 
        requirements: Optional[Dict[str, Any]] = None,
        performance_requirements: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[ModelRecommendation]:
        """推荐模型（支持租户隔离和持久化）
        
        Args:
            task_type: 任务类型
            requirements: 硬件和环境要求
            performance_requirements: 性能要求
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            模型推荐列表
            
        Raises:
            ValidationError: 参数验证失败
        """
        import time
        start_time = time.time()
        recommendations = []
        
        try:
            # 尝试从智能推荐系统获取推荐
            recommendations = self._get_intelligent_recommendations(
                task_type, requirements, performance_requirements
            )
            if not recommendations:
                # 回退到基于目录的推荐
                recommendations = self._get_catalog_based_recommendations(
                    task_type, requirements, performance_requirements
                )
        except Exception as e:
            logger.warning(f"智能推荐失败: {e}")
            try:
                recommendations = self._get_catalog_based_recommendations(
                    task_type, requirements, performance_requirements
                )
            except Exception as e2:
                logger.error(f"目录推荐失败: {e2}")
                raise ValidationError(f"获取模型推荐失败: {e2}")
        
        # 记录推荐历史
        response_time_ms = (time.time() - start_time) * 1000
        if user_id and recommendations:
            try:
                self._save_recommendation_record(
                    task_type=task_type,
                    requirements=requirements,
                    performance_requirements=performance_requirements,
                    recommendations=recommendations,
                    response_time_ms=response_time_ms,
                    tenant_id=tenant_id,
                    user_id=user_id
                )
            except Exception as e:
                logger.warning(f"Failed to save recommendation record: {e}")
        
        # 更新模型推荐计数
        for rec in recommendations:
            try:
                self._catalog_repository.increment_usage(rec.model_name, 'recommendation')
            except Exception as e:
                logger.warning(f"Failed to increment recommendation count: {e}")
        
        return recommendations
    
    def _save_recommendation_record(
        self,
        task_type: str,
        requirements: Optional[Dict[str, Any]],
        performance_requirements: Optional[Dict[str, Any]],
        recommendations: List[ModelRecommendation],
        response_time_ms: float,
        tenant_id: Optional[str],
        user_id: str
    ):
        """保存推荐记录"""
        recommended_models = []
        for rec in recommendations:
            recommended_models.append({
                'model_name': rec.model_name,
                'framework': rec.framework.value,
                'model_type': rec.model_type.value,
                'confidence': rec.confidence,
                'description': rec.description
            })
        
        self._recommendation_repository.create({
            'tenant_id': tenant_id,
            'user_id': user_id,
            'task_type': task_type,
            'requirements': requirements,
            'performance_requirements': performance_requirements,
            'recommended_models': recommended_models,
            'top_recommendation': recommendations[0].model_name if recommendations else None,
            'top_confidence': recommendations[0].confidence if recommendations else None,
            'num_recommendations': len(recommendations),
            'response_time_ms': response_time_ms,
            'source': 'api'
        })
    
    def _get_catalog_based_recommendations(self, task_type: str, 
                                         requirements: Optional[Dict[str, Any]] = None,
                                         performance_requirements: Optional[Dict[str, Any]] = None) -> List[ModelRecommendation]:
        """基于目录的模型推荐"""
        try:
            # 验证任务类型
            if task_type not in self._model_catalog:
                raise ValidationError(f"不支持的任务类型: {task_type}")
                
            # 获取候选模型
            candidate_models = self._model_catalog[task_type]
            
            # 根据要求过滤模型
            filtered_models = self._filter_models(candidate_models, requirements, performance_requirements)
            
            # 生成推荐
            recommendations = []
            for model_name, model_info in filtered_models.items():
                recommendation = ModelRecommendation(
                    model_name=model_name,
                    framework=model_info["framework"],
                    model_type=model_info["model_type"],
                    description=model_info["description"],
                    confidence=self._calculate_confidence(model_info, requirements, performance_requirements),
                    recommended_for=[task_type],
                    performance_metrics=model_info["performance"]
                )
                recommendations.append(recommendation)
                
            # 按置信度排序
            recommendations.sort(key=lambda x: x.confidence, reverse=True)
            
            return recommendations
                
        except Exception as e:
            logger.error(f"模型推荐失败: {str(e)}")
            raise ValidationError(f"模型推荐失败: {str(e)}")
            
    def _filter_models(self, models: Dict[str, Any], 
                      requirements: Optional[Dict[str, Any]],
                      performance_requirements: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """过滤模型
        
        Args:
            models: 模型列表
            requirements: 硬件和环境要求
            performance_requirements: 性能要求
            
        Returns:
            过滤后的模型列表
        """
        if not requirements and not performance_requirements:
            return models
            
        filtered_models = {}
        
        for model_name, model_info in models.items():
            # 检查硬件要求
            if requirements and not self._check_hardware_requirements(model_info, requirements):
                continue
                
            # 检查性能要求
            if performance_requirements and not self._check_performance_requirements(model_info, performance_requirements):
                continue
                
            filtered_models[model_name] = model_info
            
        return filtered_models
        
    def _check_hardware_requirements(self, model_info: Dict[str, Any], 
                                   requirements: Dict[str, Any]) -> bool:
        """检查硬件要求
        
        Args:
            model_info: 模型信息
            requirements: 硬件要求
            
        Returns:
            是否满足要求
        """
        model_requirements = model_info.get("requirements", {})
        
        # 检查GPU内存
        if "gpu_memory" in requirements:
            required_memory = self._parse_memory(requirements["gpu_memory"])
            model_memory = self._parse_memory(model_requirements.get("gpu_memory", "0GB"))
            if model_memory < required_memory:
                return False
                
        # 检查CPU核心数
        if "cpu_cores" in requirements:
            required_cores = requirements["cpu_cores"]
            model_cores = model_requirements.get("cpu_cores", 0)
            if model_cores < required_cores:
                return False
                
        return True
        
    def _parse_memory(self, memory_str: str) -> int:
        """解析内存字符串
        
        Args:
            memory_str: 内存字符串 (如 "8GB")
            
        Returns:
            内存大小(GB)
        """
        if not memory_str:
            return 0
            
        if memory_str.endswith("GB"):
            return int(memory_str[:-2])
        elif memory_str.endswith("MB"):
            return int(memory_str[:-2]) // 1024
        else:
            return int(memory_str)
            
    def _check_performance_requirements(self, model_info: Dict[str, Any], 
                                      performance_requirements: Dict[str, Any]) -> bool:
        """检查性能要求
        
        Args:
            model_info: 模型信息
            performance_requirements: 性能要求
            
        Returns:
            是否满足要求
        """
        model_performance = model_info.get("performance", {})
        
        for metric, required_value in performance_requirements.items():
            if metric in model_performance:
                model_value = model_performance[metric]
                # 对于准确率等指标，要求模型值大于等于要求值
                if metric in ["accuracy", "speed"] and model_value < required_value:
                    return False
                # 对于困惑度等指标，要求模型值小于等于要求值
                elif metric in ["perplexity"] and model_value > required_value:
                    return False
                    
        return True
        
    def _calculate_confidence(self, model_info: Dict[str, Any], 
                            requirements: Optional[Dict[str, Any]],
                            performance_requirements: Optional[Dict[str, Any]]) -> float:
        """计算推荐置信度
        
        Args:
            model_info: 模型信息
            requirements: 硬件和环境要求
            performance_requirements: 性能要求
            
        Returns:
            置信度 (0-1)
        """
        confidence = 0.5  # 基础置信度
        
        # 根据性能指标调整置信度
        performance = model_info.get("performance", {})
        if "accuracy" in performance:
            confidence += performance["accuracy"] * 0.3
        if "speed" in performance:
            confidence += performance["speed"] * 0.2
            
        # 根据硬件匹配度调整置信度
        if requirements:
            model_requirements = model_info.get("requirements", {})
            matches = 0
            total = 0
            
            if "gpu_memory" in requirements:
                total += 1
                required_memory = self._parse_memory(requirements["gpu_memory"])
                model_memory = self._parse_memory(model_requirements.get("gpu_memory", "0GB"))
                if model_memory >= required_memory:
                    matches += 1
                    
            if "cpu_cores" in requirements:
                total += 1
                required_cores = requirements["cpu_cores"]
                model_cores = model_requirements.get("cpu_cores", 0)
                if model_cores >= required_cores:
                    matches += 1
                    
            if total > 0:
                confidence += (matches / total) * 0.3
                
        # 限制置信度在0-1之间
        return max(0.0, min(1.0, confidence))
        
    def get_model_configuration(
        self, 
        model_name: str, 
        task_type: str,
        dataset_info: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        save_configuration: bool = True
    ) -> ModelConfiguration:
        """获取模型配置（支持租户隔离和持久化）
        
        Args:
            model_name: 模型名称
            task_type: 任务类型
            dataset_info: 数据集信息
            tenant_id: 租户ID
            user_id: 用户ID
            save_configuration: 是否保存配置记录
            
        Returns:
            模型配置
            
        Raises:
            ValidationError: 参数验证失败
        """
        # 首先尝试获取已保存的默认配置
        if tenant_id:
            try:
                saved_config = self._configuration_repository.get_default_configuration(
                    model_name=model_name,
                    task_type=task_type,
                    tenant_id=tenant_id
                )
                if saved_config:
                    return ModelConfiguration(
                        model_name=saved_config['model_name'],
                        framework=ModelFramework(saved_config['framework']),
                        model_type=ModelType(saved_config['model_type']),
                        hyperparameters=saved_config.get('hyperparameters', {}),
                        training_config=saved_config.get('training_config', {}),
                        hardware_config=saved_config.get('hardware_config', {})
                    )
            except Exception as e:
                logger.warning(f"Failed to get saved configuration: {e}")
        
        config = None
        try:
            # 尝试从配置服务获取优化配置
            config = self._get_optimized_configuration(model_name, task_type, dataset_info)
            if not config:
                # 回退到基于目录的配置生成
                config = self._generate_catalog_based_configuration(model_name, task_type, dataset_info)
        except Exception as e:
            logger.warning(f"获取优化配置失败: {e}")
            try:
                config = self._generate_catalog_based_configuration(model_name, task_type, dataset_info)
            except Exception as e2:
                logger.error(f"生成配置失败: {e2}")
                raise ValidationError(f"获取模型配置失败: {e2}")
        
        # 保存配置记录
        if save_configuration and user_id and config:
            try:
                self._save_configuration_record(
                    config=config,
                    task_type=task_type,
                    dataset_info=dataset_info,
                    tenant_id=tenant_id,
                    user_id=user_id
                )
            except Exception as e:
                logger.warning(f"Failed to save configuration record: {e}")
        
        # 更新模型使用计数
        try:
            self._catalog_repository.increment_usage(model_name, 'usage')
        except Exception as e:
            logger.warning(f"Failed to increment usage count: {e}")
        
        return config
    
    def _save_configuration_record(
        self,
        config: ModelConfiguration,
        task_type: str,
        dataset_info: Optional[Dict[str, Any]],
        tenant_id: Optional[str],
        user_id: str
    ):
        """保存配置记录"""
        self._configuration_repository.create({
            'tenant_id': tenant_id,
            'user_id': user_id,
            'model_name': config.model_name,
            'task_type': task_type,
            'framework': config.framework.value,
            'model_type': config.model_type.value,
            'dataset_info': dataset_info,
            'hyperparameters': config.hyperparameters,
            'training_config': config.training_config,
            'hardware_config': config.hardware_config,
            'full_config': {
                'model_name': config.model_name,
                'framework': config.framework.value,
                'model_type': config.model_type.value,
                'hyperparameters': config.hyperparameters,
                'training_config': config.training_config,
                'hardware_config': config.hardware_config
            },
            'config_source': 'auto'
        })
    
    def _generate_catalog_based_configuration(self, model_name: str, 
                                            task_type: str,
                                            dataset_info: Optional[Dict[str, Any]] = None) -> ModelConfiguration:
        """基于目录生成模型配置"""
        # 验证模型是否存在
        if task_type not in self._model_catalog or model_name not in self._model_catalog[task_type]:
            raise ValidationError(f"模型不存在: {model_name} for task {task_type}")
            
        model_info = self._model_catalog[task_type][model_name]
        
        # 生成默认超参数
        hyperparameters = self._generate_default_hyperparameters(model_name, task_type, dataset_info)
        
        # 生成训练配置
        training_config = self._generate_training_config(model_name, task_type, dataset_info)
        
        # 生成硬件配置
        hardware_config = self._generate_hardware_config(model_info)
        
        # 创建模型配置
        config = ModelConfiguration(
            model_name=model_name,
            framework=model_info["framework"],
            model_type=model_info["model_type"],
            hyperparameters=hyperparameters,
            training_config=training_config,
            hardware_config=hardware_config
        )
        
        return config
            
    def _generate_default_hyperparameters(self, model_name: str, 
                                        task_type: str,
                                        dataset_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """生成默认超参数
        
        Args:
            model_name: 模型名称
            task_type: 任务类型
            dataset_info: 数据集信息
            
        Returns:
            超参数字典
        """
        # 基础超参数
        hyperparameters = {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 3,
            "warmup_steps": 500,
            "weight_decay": 0.01
        }
        
        # 根据模型类型调整
        if "gpt" in model_name.lower() or "llama" in model_name.lower():
            hyperparameters.update({
                "learning_rate": 1e-4,
                "batch_size": 8,
                "gradient_accumulation_steps": 2
            })
        elif "bert" in model_name.lower():
            hyperparameters.update({
                "learning_rate": 2e-5,
                "batch_size": 16
            })
        elif "resnet" in model_name.lower() or "efficientnet" in model_name.lower():
            hyperparameters.update({
                "learning_rate": 1e-3,
                "batch_size": 32,
                "num_epochs": 10
            })
            
        # 根据数据集信息调整
        if dataset_info:
            num_samples = dataset_info.get("num_samples", 1000)
            # 根据数据集大小调整训练轮数
            if num_samples < 1000:
                hyperparameters["num_epochs"] = 5
            elif num_samples > 100000:
                hyperparameters["num_epochs"] = 1
            
        return hyperparameters
        
    def _generate_training_config(self, model_name: str, 
                                 task_type: str,
                                 dataset_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """生成训练配置
        
        Args:
            model_name: 模型名称
            task_type: 任务类型
            dataset_info: 数据集信息
            
        Returns:
            训练配置字典
        """
        config = {
            "output_dir": f"./outputs/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "save_strategy": "epoch",
            "evaluation_strategy": "epoch",
            "logging_steps": 100,
            "save_total_limit": 3,
            "seed": 42
        }
        
        # 根据任务类型添加特定配置
        if task_type == "text_generation":
            config.update({
                "max_length": 512,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9
            })
        elif task_type == "text_classification":
            config.update({
                "max_length": 512,
                "pad_to_max_length": True
            })
            
        return config
        
    def _generate_hardware_config(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """生成硬件配置
        
        Args:
            model_info: 模型信息
            
        Returns:
            硬件配置字典
        """
        requirements = model_info.get("requirements", {})
        
        return {
            "gpu_memory": requirements.get("gpu_memory", "8GB"),
            "cpu_cores": requirements.get("cpu_cores", 4),
            "distributed_training": requirements.get("gpu_memory", "0GB").endswith("GB") and 
                                 self._parse_memory(requirements.get("gpu_memory", "0GB")) >= 16,
            "mixed_precision": True
        }
        
    def search_models(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索模型
        
        Args:
            query: 搜索查询
            limit: 限制数量
            
        Returns:
            搜索结果
        """
        results = []
        
        # 在所有模型中搜索
        for task_type, models in self._model_catalog.items():
            for model_name, model_info in models.items():
                # 检查模型名称或描述是否匹配查询
                if (query.lower() in model_name.lower() or 
                    query.lower() in model_info["description"].lower()):
                    result = {
                        "model_name": model_name,
                        "task_type": task_type,
                        "framework": model_info["framework"].value,
                        "model_type": model_info["model_type"].value,
                        "description": model_info["description"],
                        "performance": model_info["performance"]
                    }
                    results.append(result)
                    
                    if len(results) >= limit:
                        break
                        
            if len(results) >= limit:
                break
                
        return results
    
    # =========================================================================
    # 历史记录和统计查询方法
    # =========================================================================
    
    def get_recommendation_history(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """获取用户的推荐历史
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            task_type: 任务类型过滤
            limit: 返回数量限制
            
        Returns:
            推荐历史列表
        """
        try:
            recommendations, _ = self._recommendation_repository.list_by_user(
                user_id=user_id,
                tenant_id=tenant_id,
                task_type=task_type,
                limit=limit
            )
            return recommendations
        except Exception as e:
            logger.error(f"Failed to get recommendation history: {e}")
            return []
    
    def get_configuration_history(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        model_name: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """获取用户的配置历史
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            model_name: 模型名称过滤
            task_type: 任务类型过滤
            limit: 返回数量限制
            
        Returns:
            配置历史列表
        """
        try:
            configurations, _ = self._configuration_repository.list_by_user(
                user_id=user_id,
                tenant_id=tenant_id,
                model_name=model_name,
                task_type=task_type,
                limit=limit
            )
            return configurations
        except Exception as e:
            logger.error(f"Failed to get configuration history: {e}")
            return []
    
    def get_recommendation_statistics(
        self,
        tenant_id: str,
        user_id: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取推荐统计信息
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID（可选）
            task_type: 任务类型（可选）
            
        Returns:
            统计信息
        """
        try:
            return self._recommendation_repository.get_statistics(
                tenant_id=tenant_id,
                user_id=user_id,
                task_type=task_type
            )
        except Exception as e:
            logger.error(f"Failed to get recommendation statistics: {e}")
            return {}
    
    def submit_recommendation_feedback(
        self,
        recommendation_id: str,
        selected_model: Optional[str] = None,
        feedback_score: Optional[float] = None,
        feedback_comment: Optional[str] = None,
        is_helpful: Optional[bool] = None
    ) -> bool:
        """提交推荐反馈
        
        Args:
            recommendation_id: 推荐ID
            selected_model: 用户选择的模型
            feedback_score: 反馈评分 (1-5)
            feedback_comment: 反馈评论
            is_helpful: 推荐是否有帮助
            
        Returns:
            是否成功
        """
        try:
            update_data = {}
            if selected_model is not None:
                update_data['selected_model'] = selected_model
                # 增加选择计数
                self._catalog_repository.increment_usage(selected_model, 'selection')
            if feedback_score is not None:
                update_data['feedback_score'] = feedback_score
            if feedback_comment is not None:
                update_data['feedback_comment'] = feedback_comment
            if is_helpful is not None:
                update_data['is_helpful'] = is_helpful
            
            if not update_data:
                return False
            
            result = self._recommendation_repository.update(recommendation_id, update_data)
            return result is not None
        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}")
            return False
    
    def save_custom_configuration(
        self,
        model_name: str,
        task_type: str,
        hyperparameters: Dict[str, Any],
        training_config: Dict[str, Any],
        hardware_config: Dict[str, Any],
        tenant_id: str,
        user_id: str,
        is_default: bool = False,
        tags: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """保存自定义配置
        
        Args:
            model_name: 模型名称
            task_type: 任务类型
            hyperparameters: 超参数
            training_config: 训练配置
            hardware_config: 硬件配置
            tenant_id: 租户ID
            user_id: 用户ID
            is_default: 是否设为默认
            tags: 标签
            
        Returns:
            保存的配置记录
        """
        try:
            # 获取模型信息
            model_info = None
            if task_type in self._model_catalog and model_name in self._model_catalog[task_type]:
                model_info = self._model_catalog[task_type][model_name]
            
            framework = model_info['framework'].value if model_info else 'unknown'
            model_type = model_info['model_type'].value if model_info else 'unknown'
            
            configuration = self._configuration_repository.create({
                'tenant_id': tenant_id,
                'user_id': user_id,
                'model_name': model_name,
                'task_type': task_type,
                'framework': framework,
                'model_type': model_type,
                'hyperparameters': hyperparameters,
                'training_config': training_config,
                'hardware_config': hardware_config,
                'config_source': 'manual',
                'is_default': is_default,
                'tags': tags or []
            })
            
            return configuration
        except Exception as e:
            logger.error(f"Failed to save custom configuration: {e}")
            return None
    
    def delete_configuration(
        self,
        configuration_id: str,
        tenant_id: Optional[str] = None
    ) -> bool:
        """删除配置记录
        
        Args:
            configuration_id: 配置ID
            tenant_id: 租户ID（用于验证）
            
        Returns:
            是否成功
        """
        try:
            # 验证租户权限
            configuration = self._configuration_repository.get_by_id(configuration_id)
            if not configuration:
                return False
            
            if tenant_id and configuration.get('tenant_id') != tenant_id:
                logger.warning(f"Unauthorized delete attempt for configuration {configuration_id}")
                return False
            
            return self._configuration_repository.delete(configuration_id)
        except Exception as e:
            logger.error(f"Failed to delete configuration: {e}")
            return False
    
    def get_task_types(self, tenant_id: Optional[str] = None) -> List[str]:
        """获取支持的任务类型
        
        Args:
            tenant_id: 租户ID
            
        Returns:
            任务类型列表
        """
        try:
            # 从目录获取任务类型
            task_types = set(self._model_catalog.keys())
            
            # 从数据库获取自定义任务类型
            try:
                db_task_types = self._catalog_repository.get_task_types(tenant_id)
                task_types.update(db_task_types)
            except Exception as e:
                logger.warning(f"Failed to get task types from database: {e}")
            
            return sorted(list(task_types))
        except Exception as e:
            logger.error(f"Failed to get task types: {e}")
            return list(self._model_catalog.keys())
    
    def get_models_for_task(
        self,
        task_type: str,
        tenant_id: Optional[str] = None,
        framework: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """获取任务类型的可用模型列表
        
        Args:
            task_type: 任务类型
            tenant_id: 租户ID
            framework: 框架过滤
            limit: 返回数量限制
            
        Returns:
            模型列表
        """
        try:
            models = []
            
            # 从目录获取
            if task_type in self._model_catalog:
                for model_name, model_info in self._model_catalog[task_type].items():
                    if framework and model_info['framework'].value != framework:
                        continue
                    
                    models.append({
                        'model_name': model_name,
                        'task_type': task_type,
                        'framework': model_info['framework'].value,
                        'model_type': model_info['model_type'].value,
                        'description': model_info.get('description', ''),
                        'performance': model_info.get('performance', {}),
                        'requirements': model_info.get('requirements', {}),
                        'source': 'catalog'
                    })
            
            # 从数据库获取
            try:
                db_models, _ = self._catalog_repository.list_by_task_type(
                    task_type=task_type,
                    tenant_id=tenant_id,
                    framework=framework,
                    limit=limit - len(models)
                )
                for model in db_models:
                    model['source'] = 'database'
                    models.append(model)
            except Exception as e:
                logger.warning(f"Failed to get models from database: {e}")
            
            return models[:limit]
        except Exception as e:
            logger.error(f"Failed to get models for task: {e}")
            return []
    
    def add_model_to_catalog(
        self,
        model_name: str,
        task_type: str,
        framework: str,
        model_type: str,
        description: str,
        performance_metrics: Dict[str, float],
        hardware_requirements: Dict[str, Any],
        tenant_id: str,
        default_hyperparameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """添加模型到目录
        
        Args:
            model_name: 模型名称
            task_type: 任务类型
            framework: 框架
            model_type: 模型类型
            description: 描述
            performance_metrics: 性能指标
            hardware_requirements: 硬件要求
            tenant_id: 租户ID
            default_hyperparameters: 默认超参数
            tags: 标签
            
        Returns:
            创建的目录条目
        """
        try:
            entry = self._catalog_repository.create({
                'tenant_id': tenant_id,
                'model_name': model_name,
                'task_type': task_type,
                'framework': framework,
                'model_type': model_type,
                'description': description,
                'performance_metrics': performance_metrics,
                'hardware_requirements': hardware_requirements,
                'min_gpu_memory': hardware_requirements.get('gpu_memory'),
                'min_cpu_cores': hardware_requirements.get('cpu_cores'),
                'default_hyperparameters': default_hyperparameters or {},
                'tags': tags or [],
                'is_system': False,
                'is_public': False
            })
            
            return entry
        except Exception as e:
            logger.error(f"Failed to add model to catalog: {e}")
            return None


# 全局模型选择服务实例
_global_model_selection_service: Optional['ModelSelectionService'] = None


def get_model_selection_service(use_memory_storage: bool = False) -> 'ModelSelectionService':
    """获取全局模型选择服务实例
    
    Args:
        use_memory_storage: 是否使用内存存储
        
    Returns:
        ModelSelectionService: 模型选择服务实例
    """
    global _global_model_selection_service
    
    if _global_model_selection_service is None:
        _global_model_selection_service = ModelSelectionService(use_memory_storage=use_memory_storage)
        
    return _global_model_selection_service