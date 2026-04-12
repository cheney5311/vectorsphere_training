import logging
import platform
import sys
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .datasets import TextDataset, SFTDataset, DPODataset

logger = logging.getLogger(__name__)

MAX_LENGTH_DEFAULT = 512


def _create_mock_tokenizer_and_model():
    """回退使用的Mock分词器与模型（无外部依赖即可运行）"""
    class MockTokenizer:
        def __init__(self):
            self.pad_token = '<pad>'
            self.eos_token = '<eos>'

        def __call__(self, texts, truncation=True, padding=True, max_length=512, return_tensors='pt'):
            import torch
            batch_size = len(texts) if isinstance(texts, list) else 1
            input_ids = torch.randint(0, 1000, (batch_size, max_length))
            attention_mask = torch.ones_like(input_ids)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

        def encode(self, text, **kwargs):
            return [1, 2, 3, 4, 5]

        def decode(self, tokens, **kwargs):
            return "mock-decode"

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 768)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True),
                num_layers=2
            )
            self.lm_head = nn.Linear(768, 1000)

        def forward(self, input_ids, attention_mask=None, labels=None):
            x = self.embedding(input_ids)
            x = self.transformer(x)
            logits = self.lm_head(x)
            
            from types import SimpleNamespace
            result = SimpleNamespace(logits=logits)
            
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                result.loss = loss
                
            return result

    return MockTokenizer(), MockModel()


def setup_model_and_tokenizer(base_model: str, device: torch.device):
    """优先加载真实HuggingFace模型，否则回退到Mock"""
    # 如果是测试/Mock模型，直接回退，避免在 Windows 上触发句柄复制错误
    base_lower = (base_model or "").lower()
    if ("mock" in base_lower) or ("test" in base_lower):
        logger.info(f"检测到测试/Mock模型标识 '{base_model}'，直接使用内置Mock模型")
        tokenizer, model = _create_mock_tokenizer_and_model()
        model = model.to(device)
        return tokenizer, model

    # 导入模型下载配置
    from backend.utils.model_download_config import setup_model_download_environment, get_model_download_config
    
    # 设置模型下载环境
    setup_model_download_environment()
    config = get_model_download_config()
    
    tokenizer = None
    model = None
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 使用配置的缓存目录和超时设置
        download_kwargs = {
            'cache_dir': config['cache_dir'],
            'local_files_only': False,
            'force_download': False
        }
        
        logger.info(f"使用国内镜像下载模型: {base_model}, 镜像源: {config['hf_endpoint']}")
        
        tokenizer = AutoTokenizer.from_pretrained(base_model, **download_kwargs)
        model = AutoModelForCausalLM.from_pretrained(base_model, **download_kwargs)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = model.to(device)
        logger.info(f"已加载真实模型与分词器: {base_model}")
        return tokenizer, model
    except Exception as e:
        logger.warning(f"加载真实模型失败，使用Mock模型: {e}")
        tokenizer, model = _create_mock_tokenizer_and_model()
        model = model.to(device)
        return tokenizer, model


def _mock_stage_data(stage: str) -> List[Dict]:
    """内置少量样本，避免外部数据依赖"""
    if stage == 'pretrain':
        texts = [
            f"预训练样本 {i}：这是用于语言建模的通用文本。" for i in range(100)
        ]
        return [{'text': t} for t in texts]
    if stage == 'finetune':
        data = []
        for i in range(100):
            data.append({
                'instruction': f'回答问题{i}',
                'input': f'输入内容{i}',
                'output': f'这是输出内容{i}（SFT）'
            })
        return data
    if stage == 'preference':
        data = []
        for i in range(100):
            data.append({
                'prompt': f'请生成主题{i}的内容',
                'chosen': f'高质量回答{i}',
                'rejected': f'低质量回答{i}'
            })
        return data
    return []


import json
import glob

def load_dataset_from_path(path: str, stage: str) -> List[Dict]:
    """从指定路径加载数据集（支持 .jsonl, .json, .txt）"""
    data = []
    if not path or not Path(path).exists():
        logger.warning(f"数据集路径不存在: {path}")
        return []
    
    path_obj = Path(path)
    files = []
    
    if path_obj.is_file():
        files = [path_obj]
    else:
        # 递归查找文件
        files.extend(path_obj.glob('**/*.jsonl'))
        files.extend(path_obj.glob('**/*.json'))
        files.extend(path_obj.glob('**/*.txt'))
    
    logger.info(f"找到 {len(files)} 个数据文件: {[str(f) for f in files]}")
    
    for file_path in files:
        try:
            if file_path.suffix == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
            elif file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        data.extend(content)
                    else:
                        data.append(content)
            elif file_path.suffix == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    # 预训练阶段按行或段落读取
                    if stage == 'pretrain':
                        text = f.read()
                        # 简单切分，实际可能需要更复杂的切分逻辑
                        chunks = [t.strip() for t in text.split('\n\n') if t.strip()]
                        for chunk in chunks:
                            data.append({'text': chunk})
                    else:
                         # 其他阶段TXT可能不适用，除非是特定格式
                        for line in f:
                            if line.strip():
                                data.append({'text': line.strip()})
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            
    logger.info(f"共加载 {len(data)} 条数据")
    return data

def build_dataloaders(stage: str,
                      tokenizer,
                      batch_size: int,
                      dataset_path: Optional[str] = None,
                      max_length: int = MAX_LENGTH_DEFAULT,
                      num_workers: int = 2) -> Tuple[DataLoader, Optional[DataLoader]]:
    """构建指定阶段的训练与评估 DataLoader"""
    
    data = []
    if dataset_path:
        data = load_dataset_from_path(dataset_path, stage)
    
    logger.info(f"Building dataloaders for stage {stage} with num_workers={num_workers}")

    if not data:
        logger.warning(f"未从路径 {dataset_path} 加载到数据，回退到 Mock 数据")
        data = _mock_stage_data(stage)
        
    train_size = int(0.9 * len(data))
    train_split = data[:train_size]
    eval_split = data[train_size:] if len(data) > train_size else None

    if stage == 'pretrain':
        texts = [item.get('text', str(item)) for item in train_split]
        train_ds = TextDataset(texts, tokenizer, max_length)
        eval_ds = None
        if eval_split:
            eval_texts = [item.get('text', str(item)) for item in eval_split]
            eval_ds = TextDataset(eval_texts, tokenizer, max_length)
    elif stage == 'finetune':
        train_ds = SFTDataset(train_split, tokenizer, max_length)
        eval_ds = SFTDataset(eval_split, tokenizer, max_length) if eval_split else None
    elif stage == 'preference':
        train_ds = DPODataset(train_split, tokenizer, max_length)
        eval_ds = DPODataset(eval_split, tokenizer, max_length) if eval_split else None
    else:
        raise ValueError(f"未知阶段: {stage}")
    # Windows 环境下强制关闭多进程以避免句柄复制错误
    if sys.platform == 'win32':
        num_workers = 0
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False
    ) if eval_ds else None
    return train_loader, eval_loader
