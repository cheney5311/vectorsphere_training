"""数据库训练器（合并自 training.database.database_trainer）

支持从数据库加载数据进行训练。
"""

import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from backend.core.exceptions import BusinessLogicError
from backend.modules.monitoring.training_monitor import get_training_monitor

logger = logging.getLogger(__name__)


class DatabaseTrainingConfig:
    """数据库训练配置类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 项目信息
        self.project_name = config.get('project_name', 'Database Training')
        self.experiment_name = config.get('experiment_name', 'Experiment 1')
        
        # 数据库配置
        self.db_config = config.get('database', {})
        self.host = self.db_config.get('host', 'localhost')
        self.port = self.db_config.get('port', 5432)
        self.database = self.db_config.get('database', '')
        self.username = self.db_config.get('username', '')
        self.password = self.db_config.get('password', '')
        self.pool_size = self.db_config.get('pool_size', 5)
        
        # 数据配置
        self.data_config = config.get('data', {})
        self.train_table = self.data_config.get('train_table', '')
        self.val_table = self.data_config.get('val_table', '')
        self.text_column = self.data_config.get('text_column', '')
        self.label_column = self.data_config.get('label_column', '')
        self.custom_query = self.data_config.get('custom_query', '')
        
        # 模型配置
        self.model_config = config.get('model', {})
        self.model_type = self.model_config.get('type', 'bert-base-uncased')
        self.model_name = self.model_config.get('name', 'bert-base-uncased')
        self.max_length = self.model_config.get('max_length', 512)
        
        # 训练配置
        self.training_config = config.get('training', {})
        self.num_epochs = self.training_config.get('epochs', 3)
        self.batch_size = self.training_config.get('batch_size', 16)
        self.learning_rate = self.training_config.get('learning_rate', 2e-5)
        self.warmup_steps = self.training_config.get('warmup_steps', 500)
        self.weight_decay = self.training_config.get('weight_decay', 0.01)
        self.fp16 = self.training_config.get('fp16', False)
        
        # 数据处理配置
        self.processing_config = config.get('processing', {})
        self.enable_cache = self.processing_config.get('enable_cache', True)
        self.shuffle_data = self.processing_config.get('shuffle_data', True)
        self.text_preprocessing = self.processing_config.get('text_preprocessing', True)
        self.auto_split = self.processing_config.get('auto_split', False)
        self.val_split_ratio = self.processing_config.get('val_split_ratio', 0.2)
        self.cache_size = self.processing_config.get('cache_size', 1000)
        self.num_workers = self.processing_config.get('num_workers', 4)
        
        # 输出配置
        self.output_dir = config.get('output_dir', './models/database_training')
        self.save_steps = config.get('save_steps', 500)
        self.eval_steps = config.get('eval_steps', 500)
        self.logging_steps = config.get('logging_steps', 100)


class DatabaseTrainer:
    """数据库训练器"""
    
    def __init__(self, config: DatabaseTrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.training_data = None
        self.validation_data = None
        
        # 训练监控器
        self.monitor = get_training_monitor(self.config.output_dir)
        
    def load_data_from_database(self) -> bool:
        """从数据库加载数据（此处为示例，使用模拟数据）"""
        try:
            logger.info("创建模拟训练数据...")
            
            # 创建训练数据
            train_data = []
            for i in range(1000):  # 1000个训练样本
                train_data.append({
                    'id': i,
                    'text': f'这是第{i}个训练样本的文本内容，用于数据库训练演示。',
                    'label': np.random.randint(0, 2)  # 二分类标签
                })
            
            train_df = pd.DataFrame(train_data)
            
            # 创建验证数据
            val_data = []
            for i in range(200):  # 200个验证样本
                val_data.append({
                    'id': i,
                    'text': f'这是第{i}个验证样本的文本内容，用于数据库训练演示。',
                    'label': np.random.randint(0, 2)  # 二分类标签
                })
            
            val_df = pd.DataFrame(val_data)
            
            # 数据预处理
            train_df = self._preprocess_data(train_df)
            val_df = self._preprocess_data(val_df)
            
            # 编码标签
            if 'label' in train_df.columns:
                self.label_encoder = LabelEncoder()
                train_df['labels'] = self.label_encoder.fit_transform(train_df['label'])
                if 'label' in val_df.columns:
                    val_df['labels'] = self.label_encoder.transform(val_df['label'])
            
            self.training_data = train_df
            self.validation_data = val_df
            
            logger.info(f"加载了 {len(train_df)} 个训练样本")
            logger.info(f"加载了 {len(val_df)} 个验证样本")
            
            return True
            
        except Exception as e:
            logger.error(f"从数据库加载数据失败: {e}")
            return False
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        if not self.config.text_preprocessing:
            return df
        
        # 清理文本数据
        if 'text' in df.columns:
            df = df.dropna(subset=['text'])
            df['text'] = df['text'].astype(str).str.strip()
            df = df[df['text'] != '']
        
        return df
    
    def prepare_model_and_tokenizer(self) -> bool:
        """准备模型和分词器（示例用简化实现）"""
        try:
            # 简单的分词器
            class MockTokenizer:
                def __init__(self):
                    self.pad_token_id = 0
                def __call__(self, texts, truncation=True, padding=True, max_length=512, return_tensors='pt'):
                    import torch
                    batch_size = len(texts) if isinstance(texts, list) else 1
                    input_ids = torch.randint(0, 1000, (batch_size, max_length))
                    attention_mask = torch.ones_like(input_ids)
                    return {'input_ids': input_ids, 'attention_mask': attention_mask}
            self.tokenizer = MockTokenizer()
            
            # 简单的模型
            class MockModel(torch.nn.Module):
                def __init__(self, num_labels=2):
                    super().__init__()
                    self.classifier = torch.nn.Linear(512, num_labels)
                def forward(self, input_ids, attention_mask=None, labels=None):
                    import torch.nn.functional as F
                    logits = self.classifier(torch.randn(input_ids.shape[0], 512))
                    if labels is not None:
                        loss = F.cross_entropy(logits, labels.squeeze())
                        return {'loss': loss, 'logits': logits}
                    return {'logits': logits}
            
            num_labels = len(self.label_encoder.classes_) if self.label_encoder else 2
            self.model = MockModel(num_labels)
            
            logger.info(f"创建模拟模型，标签数量: {num_labels}")
            return True
            
        except Exception as e:
            logger.error(f"模型准备错误: {e}")
            return False
    
    def create_datasets(self) -> Tuple[Optional[torch.utils.data.Dataset], Optional[torch.utils.data.Dataset]]:
        """创建训练和验证数据集"""
        try:
            class SimpleDataset(torch.utils.data.Dataset):
                def __init__(self, dataframe, tokenizer, text_column='text', label_column='labels', max_length=512):
                    self.dataframe = dataframe
                    self.tokenizer = tokenizer
                    self.text_column = text_column
                    self.label_column = label_column
                    self.max_length = max_length
                def __len__(self):
                    return len(self.dataframe)
                def __getitem__(self, idx):
                    row = self.dataframe.iloc[idx]
                    text = row[self.text_column]
                    labels = row[self.label_column] if self.label_column in row else 0
                    encoding = self.tokenizer([text], max_length=self.max_length, padding='max_length', truncation=True)
                    return {
                        'input_ids': encoding['input_ids'][0],
                        'attention_mask': encoding['attention_mask'][0],
                        'labels': torch.tensor(labels, dtype=torch.long)
                    }
            train_dataset = SimpleDataset(self.training_data, self.tokenizer) if self.training_data is not None else None
            val_dataset = SimpleDataset(self.validation_data, self.tokenizer) if self.validation_data is not None else None
            return train_dataset, val_dataset
        except Exception as e:
            logger.error(f"数据集创建错误: {e}")
            return None, None
    
    def train(self) -> Dict[str, Any]:
        """开始训练"""
        try:
            start_time = time.time()
            self.monitor.log_training_start(self.config.config)
            
            if not self.load_data_from_database():
                return {'success': False, 'error': '从数据库加载数据失败'}
            if not self.prepare_model_and_tokenizer():
                return {'success': False, 'error': '准备模型和分词器失败'}
            
            train_dataset, val_dataset = self.create_datasets()
            if train_dataset is None:
                return {'success': False, 'error': '创建训练数据集失败'}
            
            train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False) if val_dataset else None
            
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
            
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            for epoch in range(self.config.num_epochs):
                self.monitor.log_epoch_start(epoch, self.config.num_epochs)
                for batch_idx, batch in enumerate(train_dataloader):
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    labels = batch['labels']
                    outputs = self.model(input_ids, attention_mask, labels)
                    loss = outputs['loss']
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    num_batches += 1
                    if batch_idx % self.config.logging_steps == 0:
                        self.monitor.log_step(epoch * len(train_dataloader) + batch_idx, {'loss': loss.item(), 'learning_rate': self.config.learning_rate})
                avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
                self.monitor.log_epoch_end(epoch, {'loss': avg_loss})
                logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - 平均损失: {avg_loss:.4f}")
            
            output_path = Path(self.config.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), output_path / 'model.pth')
            
            end_time = time.time()
            training_time = end_time - start_time
            final_metrics = {
                'final_loss': total_loss / num_batches if num_batches > 0 else 0,
                'training_time': training_time
            }
            self.monitor.log_training_end(training_time, final_metrics)
            return {
                'success': True,
                'training_time': training_time,
                'train_samples': len(self.training_data) if self.training_data is not None else 0,
                'val_samples': len(self.validation_data) if self.validation_data is not None else 0,
                'model_path': str(output_path / 'model.pth'),
                'final_loss': total_loss / num_batches if num_batches > 0 else 0,
                'config': self.config.config
            }
        except Exception as e:
            logger.error(f"训练错误: {e}")
            return {'success': False, 'error': str(e)}


def create_database_trainer(config: Dict[str, Any]) -> DatabaseTrainer:
    """创建数据库训练器的便捷函数"""
    try:
        training_config = DatabaseTrainingConfig(config)
        return DatabaseTrainer(training_config)
    except Exception as e:
        logger.error(f"创建数据库训练器失败: {e}")
        raise BusinessLogicError(f"创建数据库训练器失败: {e}")


def launch_database_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """启动数据库训练的便捷函数"""
    try:
        trainer = create_database_trainer(config)
        return trainer.train()
    except Exception as e:
        logger.error(f"启动数据库训练失败: {e}")
        return {'success': False, 'error': str(e)}