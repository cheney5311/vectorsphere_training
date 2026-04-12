from typing import List, Dict
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """纯文本数据集（用于预训练）"""
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }


class SFTDataset(Dataset):
    """监督微调数据集（instruction-tuning 或对话格式）"""
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if 'instruction' in item and 'input' in item and 'output' in item:
            text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
        elif 'prompt' in item and 'response' in item:
            text = f"Human: {item['prompt']}\n\nAssistant: {item['response']}"
        else:
            text = item.get('text', str(item))

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }


class DPODataset(Dataset):
    """DPO 偏好数据集（prompt + chosen/rejected）"""
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']

        chosen_text = f"{prompt}\n{chosen}"
        rejected_text = f"{prompt}\n{rejected}"

        chosen_encoding = self.tokenizer(
            chosen_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        rejected_encoding = self.tokenizer(
            rejected_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'chosen_input_ids': chosen_encoding['input_ids'].flatten(),
            'chosen_attention_mask': chosen_encoding['attention_mask'].flatten(),
            'rejected_input_ids': rejected_encoding['input_ids'].flatten(),
            'rejected_attention_mask': rejected_encoding['attention_mask'].flatten(),
        }