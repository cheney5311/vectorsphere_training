"""训练监控器（合并自 training.monitoring.training_monitor）

提供训练过程的监控和日志记录功能。
"""

import json
import logging
import time
import psutil
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from backend.core.exceptions import BusinessLogicError

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "training.log"
        self.metrics_file = self.output_dir / "metrics.json"
        self.system_metrics_file = self.output_dir / "system_metrics.json"
        
        # 训练历史记录
        self.training_history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'step': [],
            'epoch': [],
            'system_metrics': []
        }
        
        # 设置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志记录"""
        # 创建文件处理器
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # 添加处理器到根日志记录器
        logging.getLogger().addHandler(file_handler)
    
    def log_training_start(self, config: Dict[str, Any]):
        """记录训练开始"""
        logger.info("开始训练")
        logger.info(f"模型: {config.get('model_name', 'Unknown')}")
        logger.info(f"训练类型: {config.get('training_type', 'Unknown')}")
        logger.info(f"输出目录: {self.output_dir}")
        
        # 记录配置信息
        config_file = self.output_dir / "training_config.json"
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"保存配置文件失败: {e}")
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录epoch开始"""
        logger.info(f"开始Epoch {epoch + 1}/{total_epochs}")
    
    def log_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """记录epoch结束"""
        logger.info(f"Epoch {epoch + 1}完成 - 损失: {metrics.get('loss', 0.0):.4f}")
        
        # 记录指标
        self.training_history['epoch'].append(epoch)
        if 'loss' in metrics:
            self.training_history['train_loss'].append(metrics['loss'])
        
        # 记录系统指标
        system_metrics = self._collect_system_metrics()
        self.training_history['system_metrics'].append(system_metrics)
    
    def log_step(self, step: int, metrics: Dict[str, float]):
        """记录训练步骤"""
        if step % 100 == 0:  # 每100步记录一次
            logger.info(f"步骤 {step} - 损失: {metrics.get('loss', 0.0):.4f}")
            
            # 记录指标
            self.training_history['step'].append(step)
            if 'loss' in metrics:
                self.training_history['train_loss'].append(metrics['loss'])
            if 'learning_rate' in metrics:
                self.training_history['learning_rate'].append(metrics['learning_rate'])
    
    def log_evaluation(self, metrics: Dict[str, float]):
        """记录评估结果"""
        logger.info("评估完成:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # 记录评估指标
        if 'eval_loss' in metrics:
            self.training_history['eval_loss'].append(metrics['eval_loss'])
    
    def log_training_end(self, total_time: float, final_metrics: Dict[str, Any]):
        """记录训练结束"""
        logger.info(f"训练完成，总耗时: {total_time:.2f}秒")
        logger.info("最终指标:")
        for key, value in final_metrics.items():
            logger.info(f"  {key}: {value}")
        
        # 保存训练历史
        self._save_training_history()
        
        # 生成报告
        self._generate_training_report(total_time, final_metrics)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 由于服务环境不保证 GPU/torch，可暂不直接依赖 torch
            gpu_metrics = {}
            # 可扩展：通过后端 GPU 监控服务聚合 GPU 指标
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'gpu_metrics': gpu_metrics
            }
        except Exception as e:
            logger.warning(f"收集系统指标失败: {e}")
            return {}
    
    def _save_training_history(self):
        """保存训练历史"""
        try:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2, ensure_ascii=False)
            logger.info(f"训练历史已保存到: {self.metrics_file}")
        except Exception as e:
            logger.warning(f"保存训练历史失败: {e}")
    
    def _generate_training_report(self, total_time: float, final_metrics: Dict[str, Any]):
        """生成训练报告"""
        try:
            report = {
                'training_summary': {
                    'total_time': total_time,
                    'final_metrics': final_metrics,
                    'timestamp': datetime.now().isoformat()
                },
                'training_history': self.training_history
            }
            
            report_file = self.output_dir / "training_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"训练报告已保存到: {report_file}")
        except Exception as e:
            logger.warning(f"生成训练报告失败: {e}")


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
    
    def update(self, step: int, metrics: Optional[Dict[str, Any]] = None):
        """更新进度"""
        self.current_step = step
        
        # 每隔一段时间记录进度
        current_time = time.time()
        if current_time - self.last_log_time > 30:  # 每30秒记录一次
            progress = (self.current_step / self.total_steps) * 100
            elapsed_time = current_time - self.start_time
            eta = (elapsed_time / self.current_step) * (self.total_steps - self.current_step) if self.current_step > 0 else 0
            
            logger.info(f"进度: {progress:.1f}% ({self.current_step}/{self.total_steps})")
            logger.info(f"已用时间: {elapsed_time:.1f}秒, 预计剩余时间: {eta:.1f}秒")
            
            if metrics:
                logger.info("当前指标:")
                for key, value in metrics.items():
                    logger.info(f"  {key}: {value}")
            
            self.last_log_time = current_time


def get_training_monitor(output_dir: str) -> TrainingMonitor:
    """获取训练监控器实例"""
    return TrainingMonitor(output_dir)


def create_progress_tracker(total_steps: int) -> ProgressTracker:
    """创建进度跟踪器"""
    return ProgressTracker(total_steps)