# -*- coding: utf-8 -*-
"""
配置模板管理器
提供多种训练模式的配置模板，基于 config_examples.md 中的配置示例
"""

import json
from typing import Dict, Any, List
from enum import Enum


class TemplateType(Enum):
    """模板类型枚举"""
    BASIC_TEXT_GENERATION = "basic_text_generation"
    MOE_TRAINING = "moe_training"
    MULTIMODAL_TRAINING = "multimodal_training"
    DISTRIBUTED_TRAINING = "distributed_training"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    MODEL_COMPRESSION = "model_compression"
    HYPERPARAMETER_SEARCH = "hyperparameter_search"
    LR_FINDER = "lr_finder"
    DATABASE_TRAINING = "database_training"
    PRODUCTION_CONFIG = "production_config"


def get_template_description(template_name: str) -> str:
    """获取模板描述"""
    descriptions = {
        TemplateType.BASIC_TEXT_GENERATION.value: '基础文本生成模型训练',
        TemplateType.MOE_TRAINING.value: 'MoE (Mixture of Experts) 大模型训练',
        TemplateType.MULTIMODAL_TRAINING.value: '多模态训练（文本+图像）',
        TemplateType.DISTRIBUTED_TRAINING.value: '分布式训练配置',
        TemplateType.KNOWLEDGE_DISTILLATION.value: '知识蒸馏训练',
        TemplateType.MODEL_COMPRESSION.value: '模型压缩（量化+剪枝）',
        TemplateType.HYPERPARAMETER_SEARCH.value: '超参数搜索优化',
        TemplateType.LR_FINDER.value: '学习率查找器',
        TemplateType.DATABASE_TRAINING.value: '数据库驱动的训练',
        TemplateType.PRODUCTION_CONFIG.value: '生产环境配置'
    }
    return descriptions.get(template_name, '未知模板')


def _get_basic_text_generation_config() -> Dict[str, Any]:
    """基础文本生成配置"""
    return {
        "global": {
            "project_name": "text_generation_basic",
            "experiment_name": "gpt2_small_training",
            "seed": 42,
            "device": "auto",
            "mixed_precision": True,
            "gradient_checkpointing": False
        },
        "model": {
            "type": "gpt2",
            "name": "gpt2",
            "config": {
                "vocab_size": 50257,
                "n_positions": 1024,
                "n_embd": 768,
                "n_layer": 12,
                "n_head": 12,
                "activation_function": "gelu_new",
                "resid_pdrop": 0.1,
                "embd_pdrop": 0.1,
                "attn_pdrop": 0.1,
                "layer_norm_epsilon": 1e-5,
                "initializer_range": 0.02
            },
            "pretrained_path": None,
            "save_path": "./models/gpt2_basic"
        },
        "data": {
            "train_path": "./data/training/train.jsonl",
            "val_path": "./data/training/val.jsonl",
            "test_path": "./data/training/test.jsonl",
            "tokenizer": "gpt2",
            "max_length": 512,
            "preprocessing": {
                "lowercase": False,
                "remove_special_chars": False,
                "truncation": True,
                "padding": "max_length"
            }
        },
        "training": {
            "num_epochs": 3,
            "batch_size": 8,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "warmup_steps": 500,
            "max_grad_norm": 1.0,
            "optimizer": "adamw",
            "scheduler": "linear",
            "save_steps": 1000,
            "eval_steps": 500,
            "logging_steps": 100,
            "save_total_limit": 3
        },
        "monitoring": {
            "wandb": {
                "enabled": False,
                "project": "vectorsphere",
                "entity": None
            },
            "tensorboard": {
                "enabled": True,
                "log_dir": "./logs/tensorboard"
            },
            "metrics": ["loss", "perplexity", "bleu"]
        }
    }


class ConfigTemplateManager:
    """配置模板管理器"""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """加载所有配置模板"""
        return {
            TemplateType.BASIC_TEXT_GENERATION.value: _get_basic_text_generation_config(),
            TemplateType.MOE_TRAINING.value: self._get_moe_training_config(),
            TemplateType.MULTIMODAL_TRAINING.value: self._get_multimodal_training_config(),
            TemplateType.DISTRIBUTED_TRAINING.value: self._get_distributed_training_config(),
            TemplateType.KNOWLEDGE_DISTILLATION.value: self._get_knowledge_distillation_config(),
            TemplateType.MODEL_COMPRESSION.value: self._get_model_compression_config(),
            TemplateType.HYPERPARAMETER_SEARCH.value: self._get_hyperparameter_search_config(),
            TemplateType.LR_FINDER.value: self._get_lr_finder_config(),
            TemplateType.DATABASE_TRAINING.value: self._get_database_training_config(),
            TemplateType.PRODUCTION_CONFIG.value: self._get_production_config()
        }
    
    def get_template(self, template_name: str) -> Dict[str, Any]:
        """获取指定的配置模板"""
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        return self.templates[template_name].copy()
    
    def list_templates(self) -> List[str]:
        """列出所有可用的模板名称"""
        return list(self.templates.keys())

    def _get_moe_training_config(self) -> Dict[str, Any]:
        """MoE大模型训练配置"""
        return {
            "global": {
                "project_name": "moe_large_model",
                "experiment_name": "switch_transformer_8x",
                "seed": 42,
                "device": "auto",
                "mixed_precision": True,
                "gradient_checkpointing": True
            },
            "model": {
                "type": "switch_transformer",
                "name": "switch_base_8",
                "config": {
                    "vocab_size": 32128,
                    "d_model": 768,
                    "d_kv": 64,
                    "d_ff": 2048,
                    "num_layers": 12,
                    "num_heads": 12,
                    "num_experts": 8,
                    "expert_capacity": 64,
                    "router_z_loss_coef": 0.001,
                    "router_aux_loss_coef": 0.01,
                    "dropout_rate": 0.1
                },
                "pretrained_path": None,
                "save_path": "./models/switch_transformer_8x"
            },
            "data": {
                "train_path": "./data/training/large_corpus.jsonl",
                "val_path": "./data/training/val_large.jsonl",
                "tokenizer": "t5-base",
                "max_length": 512,
                "preprocessing": {
                    "lowercase": False,
                    "remove_special_chars": False,
                    "truncation": True,
                    "padding": "max_length"
                }
            },
            "training": {
                "num_epochs": 5,
                "batch_size": 4,
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "warmup_steps": 1000,
                "max_grad_norm": 1.0,
                "optimizer": "adamw",
                "scheduler": "cosine",
                "save_steps": 2000,
                "eval_steps": 1000,
                "logging_steps": 100,
                "gradient_accumulation_steps": 4
            },
            "monitoring": {
                "wandb": {
                    "enabled": True,
                    "project": "moe_training",
                    "entity": None
                },
                "metrics": ["loss", "router_z_loss", "router_aux_loss", "expert_utilization"]
            }
        }
    
    def _get_multimodal_training_config(self) -> Dict[str, Any]:
        """多模态训练配置"""
        return {
            "global": {
                "project_name": "multimodal_training",
                "experiment_name": "clip_style_model",
                "seed": 42,
                "device": "auto",
                "mixed_precision": True
            },
            "model": {
                "type": "clip",
                "name": "clip_vit_b32",
                "config": {
                    "vision_config": {
                        "image_size": 224,
                        "patch_size": 32,
                        "num_channels": 3,
                        "hidden_size": 768,
                        "num_hidden_layers": 12,
                        "num_attention_heads": 12,
                        "intermediate_size": 3072,
                        "dropout": 0.0,
                        "attention_dropout": 0.0
                    },
                    "text_config": {
                        "vocab_size": 49408,
                        "hidden_size": 512,
                        "intermediate_size": 2048,
                        "num_hidden_layers": 12,
                        "num_attention_heads": 8,
                        "max_position_embeddings": 77,
                        "dropout": 0.0,
                        "attention_dropout": 0.0
                    },
                    "projection_dim": 512,
                    "logit_scale_init_value": 2.6592
                },
                "save_path": "./models/clip_multimodal"
            },
            "data": {
                "train_path": "./data/training/image_text_pairs.jsonl",
                "val_path": "./data/training/val_image_text.jsonl",
                "image_preprocessing": {
                    "resize": [224, 224],
                    "normalize": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225]
                    },
                    "augmentation": {
                        "random_crop": True,
                        "horizontal_flip": True,
                        "color_jitter": 0.1
                    }
                },
                "text_preprocessing": {
                    "max_length": 77,
                    "truncation": True,
                    "padding": "max_length"
                }
            },
            "training": {
                "num_epochs": 10,
                "batch_size": 32,
                "learning_rate": 1e-4,
                "weight_decay": 0.2,
                "warmup_steps": 2000,
                "temperature": 0.07,
                "optimizer": "adamw",
                "scheduler": "cosine"
            }
        }
    
    def _get_distributed_training_config(self) -> Dict[str, Any]:
        """分布式训练配置"""
        return {
            "global": {
                "project_name": "distributed_training",
                "experiment_name": "multi_gpu_bert",
                "seed": 42,
                "device": "auto",
                "mixed_precision": True
            },
            "distributed": {
                "backend": "nccl",
                "init_method": "env://",
                "world_size": 4,
                "rank": 0,
                "local_rank": 0,
                "find_unused_parameters": False,
                "gradient_as_bucket_view": True
            },
            "model": {
                "type": "bert",
                "name": "bert-base-uncased",
                "config": {
                    "vocab_size": 30522,
                    "hidden_size": 768,
                    "num_hidden_layers": 12,
                    "num_attention_heads": 12,
                    "intermediate_size": 3072,
                    "max_position_embeddings": 512,
                    "type_vocab_size": 2,
                    "hidden_dropout_prob": 0.1,
                    "attention_probs_dropout_prob": 0.1
                },
                "save_path": "./models/bert_distributed"
            },
            "training": {
                "num_epochs": 3,
                "batch_size": 16,
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "warmup_steps": 1000,
                "max_grad_norm": 1.0,
                "gradient_accumulation_steps": 2,
                "dataloader_num_workers": 4,
                "pin_memory": True
            }
        }
    
    def _get_knowledge_distillation_config(self) -> Dict[str, Any]:
        """知识蒸馏配置"""
        return {
            "global": {
                "project_name": "knowledge_distillation",
                "experiment_name": "bert_to_distilbert",
                "seed": 42,
                "device": "auto"
            },
            "teacher_model": {
                "type": "bert",
                "name": "bert-base-uncased",
                "path": "./models/teacher_bert",
                "freeze": True
            },
            "student_model": {
                "type": "distilbert",
                "name": "distilbert-base-uncased",
                "config": {
                    "vocab_size": 30522,
                    "max_position_embeddings": 512,
                    "sinusoidal_pos_embds": False,
                    "n_layers": 6,
                    "n_heads": 12,
                    "dim": 768,
                    "hidden_dim": 3072,
                    "dropout": 0.1,
                    "attention_dropout": 0.1
                },
                "save_path": "./models/student_distilbert"
            },
            "distillation": {
                "temperature": 4.0,
                "alpha": 0.7,
                "beta": 0.3,
                "loss_type": "kl_div",
                "layer_mapping": "linear"
            },
            "training": {
                "num_epochs": 5,
                "batch_size": 32,
                "learning_rate": 5e-5,
                "weight_decay": 0.01,
                "warmup_steps": 1000
            }
        }
    
    def _get_model_compression_config(self) -> Dict[str, Any]:
        """模型压缩配置"""
        return {
            "global": {
                "project_name": "model_compression",
                "experiment_name": "bert_quantization_pruning",
                "seed": 42,
                "device": "auto"
            },
            "model": {
                "type": "bert",
                "name": "bert-base-uncased",
                "pretrained_path": "./models/bert_base",
                "save_path": "./models/bert_compressed"
            },
            "compression": {
                "quantization": {
                    "enabled": True,
                    "method": "dynamic",
                    "dtype": "qint8",
                    "qconfig": "fbgemm"
                },
                "pruning": {
                    "enabled": True,
                    "method": "magnitude",
                    "sparsity": 0.5,
                    "structured": False,
                    "global_pruning": True
                },
                "knowledge_distillation": {
                    "enabled": True,
                    "teacher_path": "./models/bert_base",
                    "temperature": 4.0,
                    "alpha": 0.7
                }
            },
            "training": {
                "num_epochs": 3,
                "batch_size": 32,
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "fine_tuning_steps": 1000
            }
        }
    
    def _get_hyperparameter_search_config(self) -> Dict[str, Any]:
        """超参数搜索配置"""
        return {
            "global": {
                "project_name": "hyperparameter_search",
                "experiment_name": "bert_hp_optimization",
                "seed": 42,
                "device": "auto"
            },
            "search": {
                "method": "optuna",
                "direction": "maximize",
                "metric": "eval_accuracy",
                "n_trials": 50,
                "timeout": 3600,
                "sampler": "TPE",
                "pruner": "MedianPruner"
            },
            "search_space": {
                "learning_rate": {
                    "type": "loguniform",
                    "low": 1e-6,
                    "high": 1e-3
                },
                "batch_size": {
                    "type": "categorical",
                    "choices": [8, 16, 32, 64]
                },
                "weight_decay": {
                    "type": "uniform",
                    "low": 0.0,
                    "high": 0.3
                },
                "warmup_steps": {
                    "type": "int",
                    "low": 100,
                    "high": 2000
                },
                "dropout": {
                    "type": "uniform",
                    "low": 0.0,
                    "high": 0.5
                }
            },
            "base_config": {
                "model": {
                    "type": "bert",
                    "name": "bert-base-uncased"
                },
                "training": {
                    "num_epochs": 3,
                    "max_grad_norm": 1.0
                }
            }
        }
    
    def _get_lr_finder_config(self) -> Dict[str, Any]:
        """学习率查找配置"""
        return {
            "global": {
                "project_name": "lr_finder",
                "experiment_name": "bert_lr_range_test",
                "seed": 42,
                "device": "auto"
            },
            "model": {
                "type": "bert",
                "name": "bert-base-uncased",
                "config": {
                    "vocab_size": 30522,
                    "hidden_size": 768,
                    "num_hidden_layers": 12,
                    "num_attention_heads": 12,
                    "intermediate_size": 3072,
                    "max_position_embeddings": 512,
                    "type_vocab_size": 2,
                    "hidden_dropout_prob": 0.1,
                    "attention_probs_dropout_prob": 0.1
                },
                "save_path": "./models/bert_lr_finder"
            },
            "lr_finder": {
                "start_lr": 1e-7,
                "end_lr": 1e-1,
                "num_steps": 100,
                "step_mode": "exp",
                "smooth_f": 0.05,
                "diverge_th": 5
            },
            "data": {
                "train_path": "./data/training/train_small.jsonl",
                "tokenizer": "bert-base-uncased",
                "max_length": 128,
                "preprocessing": {
                    "truncation": True,
                    "padding": "max_length"
                }
            },
            "training": {
                "batch_size": 32,
                "optimizer": "adamw",
                "weight_decay": 0.01
            }
        }
    
    def _get_database_training_config(self) -> Dict[str, Any]:
        """数据库训练配置"""
        return {
            "global": {
                "project_name": "database_training",
                "experiment_name": "sql_knowledge_model",
                "seed": 42,
                "device": "auto",
                "mixed_precision": True
            },
            "model": {
                "type": "t5",
                "name": "t5-base",
                "config": {
                    "vocab_size": 32128,
                    "d_model": 768,
                    "d_kv": 64,
                    "d_ff": 3072,
                    "num_layers": 12,
                    "num_heads": 12,
                    "relative_attention_num_buckets": 32,
                    "dropout_rate": 0.1,
                    "layer_norm_epsilon": 1e-6,
                    "initializer_factor": 1.0,
                    "feed_forward_proj": "relu"
                },
                "save_path": "./models/t5_database"
            },
            "data": {
                "type": "database",
                "database": {
                    "connection_string": "postgresql://user:password@localhost:5432/training_db",
                    "query_templates": [
                        "SELECT * FROM {table} WHERE {condition}",
                        "INSERT INTO {table} ({columns}) VALUES ({values})",
                        "UPDATE {table} SET {updates} WHERE {condition}",
                        "DELETE FROM {table} WHERE {condition}"
                    ],
                    "schema_path": "./data/training/database_schema.json",
                    "sample_queries_path": "./data/training/sample_queries.sql"
                },
                "preprocessing": {
                    "max_source_length": 512,
                    "max_target_length": 256,
                    "truncation": True,
                    "padding": "max_length"
                }
            },
            "training": {
                "num_epochs": 10,
                "batch_size": 16,
                "learning_rate": 3e-4,
                "weight_decay": 0.01,
                "warmup_steps": 1000,
                "max_grad_norm": 1.0,
                "optimizer": "adamw",
                "scheduler": "linear",
                "save_steps": 1000,
                "eval_steps": 500,
                "logging_steps": 100
            },
            "monitoring": {
                "wandb": {
                    "enabled": True,
                    "project": "database_training",
                    "entity": None
                },
                "metrics": ["loss", "bleu", "exact_match", "sql_accuracy"]
            }
        }
    
    def _get_production_config(self) -> Dict[str, Any]:
        """生产环境配置"""
        return {
            "global": {
                "project_name": "production_deployment",
                "experiment_name": "production_model_v1",
                "seed": 42,
                "device": "auto",
                "mixed_precision": True,
                "gradient_checkpointing": True
            },
            "model": {
                "type": "gpt2",
                "name": "gpt2-medium",
                "config": {
                    "vocab_size": 50257,
                    "n_positions": 1024,
                    "n_embd": 1024,
                    "n_layer": 24,
                    "n_head": 16,
                    "activation_function": "gelu_new",
                    "resid_pdrop": 0.1,
                    "embd_pdrop": 0.1,
                    "attn_pdrop": 0.1,
                    "layer_norm_epsilon": 1e-5,
                    "initializer_range": 0.02
                },
                "pretrained_path": "./models/pretrained/gpt2-medium",
                "save_path": "./models/production/gpt2_production"
            },
            "data": {
                "train_path": "./data/training/production/train_large.jsonl",
                "val_path": "./data/training/production/val_large.jsonl",
                "test_path": "./data/training/production/test_large.jsonl",
                "tokenizer": "gpt2",
                "max_length": 1024,
                "preprocessing": {
                    "lowercase": False,
                    "remove_special_chars": False,
                    "truncation": True,
                    "padding": "max_length",
                    "data_validation": True,
                    "quality_filtering": True
                }
            },
            "training": {
                "num_epochs": 50,
                "batch_size": 32,
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "warmup_steps": 2000,
                "max_grad_norm": 1.0,
                "optimizer": "adamw",
                "scheduler": "cosine",
                "save_steps": 1000,
                "eval_steps": 500,
                "logging_steps": 100,
                "save_total_limit": 5,
                "gradient_accumulation_steps": 2,
                "early_stopping": {
                    "enabled": True,
                    "patience": 5,
                    "metric": "eval_loss",
                    "mode": "min"
                }
            },
            "monitoring": {
                "wandb": {
                    "enabled": True,
                    "project": "production_training",
                    "entity": None,
                    "tags": ["production", "gpt2-medium"]
                },
                "tensorboard": {
                    "enabled": True,
                    "log_dir": "./logs/production/tensorboard"
                },
                "metrics": ["loss", "perplexity", "bleu", "rouge"],
                "performance": {
                    "monitor_gpu": True,
                    "monitor_memory": True,
                    "profile_training": False
                }
            },
            "inference": {
                "service": {
                    "host": "0.0.0.0",
                    "port": 8080,
                    "workers": 4,
                    "timeout": 30
                },
                "model_loading": {
                    "device_map": "auto",
                    "torch_dtype": "float16",
                    "low_cpu_mem_usage": True,
                    "load_in_8bit": False
                },
                "generation": {
                    "max_new_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "do_sample": True,
                    "repetition_penalty": 1.1,
                    "length_penalty": 1.0
                }
            },
            "security": {
                "privacy": {
                    "differential_privacy": False,
                    "noise_multiplier": 1.0,
                    "max_grad_norm": 1.0
                },
                "model_security": {
                    "validate_inputs": True,
                    "sanitize_outputs": True,
                    "max_input_length": 2048,
                    "content_filtering": True
                },
                "data_security": {
                    "encrypt_checkpoints": False,
                    "secure_logging": True,
                    "audit_trail": True
                }
            },
            "deployment": {
                "docker": {
                    "enabled": True,
                    "base_image": "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
                    "requirements_file": "requirements.txt"
                },
                "kubernetes": {
                    "enabled": False,
                    "namespace": "vectorsphere",
                    "replicas": 3,
                    "resources": {
                        "requests": {
                            "cpu": "2",
                            "memory": "8Gi",
                            "nvidia.com/gpu": "1"
                        },
                        "limits": {
                            "cpu": "4",
                            "memory": "16Gi",
                            "nvidia.com/gpu": "1"
                        }
                    }
                }
            }
        }