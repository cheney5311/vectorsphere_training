"""异步处理器

提供异步任务处理和队列管理功能，支持优先级队列和任务超时控制。
"""

import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from queue import PriorityQueue, Empty
from typing import Any, Callable, Dict, Optional

from backend.modules.performance.performance_errors import AsyncProcessorError

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    URGENT = 0


@dataclass
class AsyncTask:
    """异步任务"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    func: Optional[Callable[..., Any]] = None
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    callback: Optional[Callable[[Any], None]] = None
    error_callback: Optional[Callable[[Exception], None]] = None
    timeout: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None
    status: str = 'pending'  # pending, running, completed, failed, timeout

    def __lt__(self, other):
        """用于优先级队列排序

        增强的比较方法，处理所有可能的None值和边界情况
        """
        # 类型检查
        if not isinstance(other, AsyncTask):
            return NotImplemented

        # 如果两个对象是同一个实例，返回False
        if self is other:
            return False

        # 处理 None 优先级的情况，将其视为最低优先级
        try:
            if self.priority is None:
                self_priority = TaskPriority.LOW.value
            elif hasattr(self.priority, 'value'):
                self_priority = self.priority.value
            else:
                # 如果priority不是枚举类型，尝试转换为数值
                self_priority = int(self.priority) if self.priority is not None else TaskPriority.LOW.value
        except (AttributeError, TypeError, ValueError):
            self_priority = TaskPriority.LOW.value

        try:
            if other.priority is None:
                other_priority = TaskPriority.LOW.value
            elif hasattr(other.priority, 'value'):
                other_priority = other.priority.value
            else:
                # 如果priority不是枚举类型，尝试转换为数值
                other_priority = int(other.priority) if other.priority is not None else TaskPriority.LOW.value
        except (AttributeError, TypeError, ValueError):
            other_priority = TaskPriority.LOW.value

        # 如果优先级相同，按创建时间排序（先创建的先执行）
        if self_priority == other_priority:
            # 确保创建时间不为None，并处理可能的类型错误
            try:
                self_time = float(self.created_at) if self.created_at is not None else 0.0
            except (TypeError, ValueError):
                self_time = 0.0

            try:
                other_time = float(other.created_at) if other.created_at is not None else 0.0
            except (TypeError, ValueError):
                other_time = 0.0

            # 安全的时间比较
            try:
                # 如果时间也相同，使用ID进行稳定排序
                if abs(self_time - other_time) < 1e-9:  # 浮点数比较容差
                    try:
                        return str(self.id) < str(other.id)
                    except (AttributeError, TypeError):
                        return False

                return self_time < other_time
            except (TypeError, ValueError):
                # 如果时间比较失败，使用ID比较
                try:
                    return str(self.id) < str(other.id)
                except (AttributeError, TypeError):
                    return False

        return self_priority < other_priority


# 特殊的停止任务，用于优雅关闭工作线程
class StopTask:
    """停止任务标记，用于通知工作线程停止"""

    def __init__(self):
        self.id = "STOP_TASK"
        self.priority = TaskPriority.URGENT.value  # 最高优先级确保立即处理
        self.created_at = time.time()

    def __lt__(self, other):
        """停止任务具有最高优先级"""
        if isinstance(other, StopTask):
            return False  # 停止任务之间不分先后
        return True  # 停止任务优先于所有其他任务


class AsyncProcessor:
    """异步处理器"""

    def __init__(self, max_workers: int = 5, queue_size: int = 500):
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.task_queue = PriorityQueue(maxsize=queue_size)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="AsyncProcessor")
        self.pending_tasks: Dict[str, AsyncTask] = {}  # 跟踪队列中等待的任务
        self.running_tasks: Dict[str, AsyncTask] = {}
        self.completed_tasks: Dict[str, AsyncTask] = {}
        self.failed_tasks: Dict[str, AsyncTask] = {}
        self._lock = threading.RLock()
        self._stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'timeout_tasks': 0,
            'avg_execution_time': 0.0,
            'queue_size': 0,
            'active_workers': 0
        }
        self._execution_times = []
        self._running = True
        self._worker_threads = []

        # 启动工作线程
        self._start_workers()

    def init_app(self, app):
        """初始化Flask应用集成

        Args:
            app: Flask应用实例
        """
        try:
            # 将async_processor实例绑定到app上
            app.async_processor = self

            # 从应用配置更新处理器设置
            if hasattr(app, 'config'):
                # 更新最大工作线程数
                max_workers = app.config.get('ASYNC_PROCESSOR_MAX_WORKERS')
                if max_workers and isinstance(max_workers, int) and max_workers > 0:
                    logger.info(f"AsyncProcessor max_workers updated from config: {max_workers}")

                # 更新队列大小
                queue_size = app.config.get('ASYNC_PROCESSOR_QUEUE_SIZE')
                if queue_size and isinstance(queue_size, int) and queue_size > 0:
                    logger.info(f"AsyncProcessor queue_size updated from config: {queue_size}")

            # 注册应用关闭时的清理函数
            @app.teardown_appcontext
            def cleanup_async_processor(error):
                """应用上下文清理时执行任务清理"""
                try:
                    self.cleanup_completed_tasks()
                except Exception as e:
                    logger.warning(f"AsyncProcessor cleanup warning: {e}")

            # 在应用关闭时执行完整关闭
            def shutdown_async_processor():
                """应用关闭时执行AsyncProcessor关闭"""
                try:
                    self.shutdown()
                except Exception as e:
                    logger.error(f"AsyncProcessor shutdown error: {e}")

            # 注册关闭处理器
            import atexit
            atexit.register(shutdown_async_processor)

            logger.info("AsyncProcessor initialized with Flask app successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AsyncProcessor with Flask app: {e}")
            raise

    def _start_workers(self):
        """启动工作线程"""
        for i in range(self.max_workers):
            worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"AsyncProcessor-Worker-{i}",
                daemon=True
            )
            worker_thread.start()
            self._worker_threads.append(worker_thread)

        logger.info(f"AsyncProcessor started with {self.max_workers} workers")

    def _worker_loop(self):
        """工作线程循环（改进版，快速响应停止信号）"""
        while self._running:
            try:
                # 获取任务（较短超时，快速响应停止信号）
                task = self.task_queue.get(timeout=0.5)

                if isinstance(task, StopTask):  # 停止信号
                    logger.debug("工作线程收到停止信号")
                    break

                # 再次检查运行状态
                if not self._running:
                    # 如果已停止，将任务放回队列
                    try:
                        self.task_queue.put_nowait(task)
                    except:
                        pass
                    break

                self._execute_task(task)

            except Empty:
                # 队列为空时检查运行状态
                if not self._running:
                    break
                continue
            except Exception as e:
                logger.error(f"Worker loop error: {str(e)}")
                # 出错时也检查运行状态
                if not self._running:
                    break

    def _execute_task(self, task: AsyncTask):
        """执行任务"""
        task.started_at = time.time()
        task.status = 'running'

        with self._lock:
            # 从待处理任务中移除
            if task.id in self.pending_tasks:
                del self.pending_tasks[task.id]
            self.running_tasks[task.id] = task
            self._stats['active_workers'] += 1

        try:
            # 设置超时
            if task.timeout:
                future = self.executor.submit(task.func, *task.args, **task.kwargs)
                task.result = future.result(timeout=task.timeout)
            else:
                task.result = task.func(*task.args, **task.kwargs)

            task.status = 'completed'
            task.completed_at = time.time()

            # 执行回调
            if task.callback:
                try:
                    task.callback(task.result)
                except Exception as callback_error:
                    logger.error(f"Task callback error: {str(callback_error)}")

            # 更新统计
            execution_time = task.completed_at - task.started_at
            with self._lock:
                self.completed_tasks[task.id] = task
                self._stats['completed_tasks'] += 1
                self._execution_times.append(execution_time)

                # 保持最近1000次执行时间
                if len(self._execution_times) > 1000:
                    self._execution_times = self._execution_times[-1000:]

                self._stats['avg_execution_time'] = sum(self._execution_times) / len(self._execution_times)

            logger.debug(f"Task {task.id} completed in {execution_time:.3f}s")

        except (asyncio.TimeoutError, TimeoutError) as timeout_error:
            task.status = 'timeout'
            task.error = timeout_error
            task.completed_at = time.time()

            # 执行错误回调
            if task.error_callback:
                try:
                    task.error_callback(timeout_error)
                except Exception as callback_error:
                    logger.error(f"Task timeout callback error: {str(callback_error)}")

            with self._lock:
                self.failed_tasks[task.id] = task
                self._stats['timeout_tasks'] += 1

            logger.warning(f"Task {task.id} timed out after {task.timeout}s")

        except Exception as e:
            task.status = 'failed'
            task.error = e
            task.completed_at = time.time()

            # 执行错误回调
            if task.error_callback:
                try:
                    task.error_callback(e)
                except Exception as callback_error:
                    logger.error(f"Task error callback error: {str(callback_error)}")

            with self._lock:
                self.failed_tasks[task.id] = task
                self._stats['failed_tasks'] += 1

            logger.error(f"Task {task.id} failed: {str(e)}")

        finally:
            with self._lock:
                if task.id in self.running_tasks:
                    del self.running_tasks[task.id]
                self._stats['active_workers'] = max(0, self._stats['active_workers'] - 1)
                self._stats['queue_size'] = self.task_queue.qsize()

    def submit_task(
        self,
        func: Callable[..., Any],
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        callback: Optional[Callable[[Any], None]] = None,
        error_callback: Optional[Callable[[Exception], None]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> str:
        """提交任务"""
        if not self._running:
            raise AsyncProcessorError("AsyncProcessor is not running")

        if func is None:
            raise AsyncProcessorError("Task function cannot be None")

        try:
            # 确保优先级有效
            if priority is None:
                priority = TaskPriority.NORMAL
            elif not isinstance(priority, TaskPriority):
                logger.warning(f"Invalid priority type: {type(priority)}, using NORMAL")
                priority = TaskPriority.NORMAL

            task = AsyncTask(
                func=func,
                args=args,
                kwargs=kwargs,
                priority=priority,
                callback=callback,
                error_callback=error_callback,
                timeout=timeout
            )

            self.task_queue.put(task, timeout=1.0)

            with self._lock:
                # 添加到待处理任务跟踪字典
                self.pending_tasks[task.id] = task
                self._stats['total_tasks'] += 1
                self._stats['queue_size'] = self.task_queue.qsize()

            logger.debug(f"Task {task.id} submitted with priority {priority.name}")

            return task.id

        except Exception as e:
            logger.error(f"Failed to submit task: {str(e)}")
            raise AsyncProcessorError(f"Failed to submit task: {str(e)}")

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态信息字典，如果任务不存在则返回None
        """
        with self._lock:
            current_time = time.time()
            
            # 检查待处理的任务（队列中等待执行）
            if task_id in self.pending_tasks:
                task = self.pending_tasks[task_id]
                wait_time = current_time - task.created_at
                # 估算队列位置（基于优先级）
                queue_position = self._estimate_queue_position(task)
                return {
                    'id': task.id,
                    'status': task.status,
                    'created_at': task.created_at,
                    'started_at': None,
                    'completed_at': None,
                    'priority': task.priority.name if hasattr(task.priority, 'name') else str(task.priority),
                    'wait_time': wait_time,
                    'queue_position': queue_position,
                    'execution_time': None,
                    'result': None,
                    'error': None
                }
            
            # 检查运行中的任务
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                execution_time = current_time - task.started_at if task.started_at else None
                wait_time = task.started_at - task.created_at if task.started_at else None
                return {
                    'id': task.id,
                    'status': task.status,
                    'created_at': task.created_at,
                    'started_at': task.started_at,
                    'completed_at': None,
                    'priority': task.priority.name if hasattr(task.priority, 'name') else str(task.priority),
                    'wait_time': wait_time,
                    'queue_position': None,
                    'execution_time': execution_time,
                    'result': None,
                    'error': None
                }

            # 检查已完成的任务
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                execution_time = task.completed_at - task.started_at if task.started_at and task.completed_at else None
                wait_time = task.started_at - task.created_at if task.started_at else None
                return {
                    'id': task.id,
                    'status': task.status,
                    'created_at': task.created_at,
                    'started_at': task.started_at,
                    'completed_at': task.completed_at,
                    'priority': task.priority.name if hasattr(task.priority, 'name') else str(task.priority),
                    'wait_time': wait_time,
                    'queue_position': None,
                    'execution_time': execution_time,
                    'result': task.result,
                    'error': None
                }

            # 检查失败的任务
            if task_id in self.failed_tasks:
                task = self.failed_tasks[task_id]
                execution_time = task.completed_at - task.started_at if task.started_at and task.completed_at else None
                wait_time = task.started_at - task.created_at if task.started_at else None
                return {
                    'id': task.id,
                    'status': task.status,
                    'created_at': task.created_at,
                    'started_at': task.started_at,
                    'completed_at': task.completed_at,
                    'priority': task.priority.name if hasattr(task.priority, 'name') else str(task.priority),
                    'wait_time': wait_time,
                    'queue_position': None,
                    'execution_time': execution_time,
                    'result': None,
                    'error': str(task.error) if task.error else None
                }

            return None
    
    def _estimate_queue_position(self, task: AsyncTask) -> int:
        """估算任务在队列中的位置
        
        基于优先级和创建时间估算任务位置
        
        Args:
            task: 要估算位置的任务
            
        Returns:
            估算的队列位置（从1开始）
        """
        position = 1
        task_priority = task.priority.value if hasattr(task.priority, 'value') else TaskPriority.LOW.value
        
        for pending_task in self.pending_tasks.values():
            if pending_task.id == task.id:
                continue
            pending_priority = pending_task.priority.value if hasattr(pending_task.priority, 'value') else TaskPriority.LOW.value
            # 优先级值越小优先级越高
            if pending_priority < task_priority:
                position += 1
            elif pending_priority == task_priority and pending_task.created_at < task.created_at:
                position += 1
        
        return position

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._lock:
            if task_id in self.running_tasks:
                # 运行中的任务无法取消
                return False

            # 从队列中移除任务（这个操作比较复杂，简化处理）
            logger.warning(f"Task {task_id} cancellation requested but not implemented for queued tasks")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取处理器统计信息"""
        with self._lock:
            stats = self._stats.copy()
            stats.update({
                'pending_tasks_count': len(self.pending_tasks),
                'running_tasks': len(self.running_tasks),
                'completed_tasks_count': len(self.completed_tasks),
                'failed_tasks_count': len(self.failed_tasks),
                'worker_threads': len(self._worker_threads),
                'is_running': self._running
            })

            return stats

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            stats = self.get_stats()

            # 检查工作线程状态
            alive_workers = sum(1 for thread in self._worker_threads if thread.is_alive())

            if alive_workers < self.max_workers * 0.5:
                return {
                    'status': 'unhealthy',
                    'message': f'Too few worker threads alive: {alive_workers}/{self.max_workers}',
                    'stats': stats
                }

            # 检查队列大小
            if stats['queue_size'] > self.queue_size * 0.9:
                return {
                    'status': 'degraded',
                    'message': f'Queue nearly full: {stats["queue_size"]}/{self.queue_size}',
                    'stats': stats
                }

            # 检查失败率
            total_tasks = stats['total_tasks']
            if total_tasks > 0:
                failure_rate = (stats['failed_tasks'] + stats['timeout_tasks']) / total_tasks
                if failure_rate > 0.1:  # 10%失败率
                    return {
                        'status': 'degraded',
                        'message': f'High failure rate: {failure_rate:.2%}',
                        'stats': stats
                    }

            return {
                'status': 'healthy',
                'message': 'AsyncProcessor is working properly',
                'stats': stats
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Health check failed: {str(e)}'
            }

    def cleanup_completed_tasks(self, max_age: float = 3600):
        """清理已完成的任务"""
        try:
            current_time = time.time()
            cleanup_count = 0

            with self._lock:
                # 清理已完成的任务
                expired_completed = [
                    task_id for task_id, task in self.completed_tasks.items()
                    if task.completed_at is not None and current_time - task.completed_at > max_age
                ]

                for task_id in expired_completed:
                    del self.completed_tasks[task_id]
                    cleanup_count += 1

                # 清理失败的任务
                expired_failed = [
                    task_id for task_id, task in self.failed_tasks.items()
                    if task.completed_at and current_time - task.completed_at > max_age
                ]

                for task_id in expired_failed:
                    del self.failed_tasks[task_id]
                    cleanup_count += 1

            if cleanup_count > 0:
                logger.info(f"Cleaned up {cleanup_count} expired tasks")

            return cleanup_count

        except Exception as e:
            logger.error(f"Task cleanup error: {str(e)}")
            return 0

    def get_cleanup_stats(self) -> Dict[str, Any]:
        """获取清理统计信息"""
        with self._lock:
            current_time = time.time()

            # 计算各种任务的年龄分布
            pending_ages = [
                current_time - task.created_at
                for task in self.pending_tasks.values()
            ]
            
            completed_ages = [
                current_time - task.completed_at
                for task in self.completed_tasks.values()
                if task.completed_at
            ]

            failed_ages = [
                current_time - task.completed_at
                for task in self.failed_tasks.values()
                if task.completed_at
            ]

            return {
                'pending_tasks_count': len(self.pending_tasks),
                'completed_tasks_count': len(self.completed_tasks),
                'failed_tasks_count': len(self.failed_tasks),
                'running_tasks_count': len(self.running_tasks),
                'oldest_pending_task_age': max(pending_ages) if pending_ages else 0.0,
                'oldest_completed_task_age': max(completed_ages) if completed_ages else 0.0,
                'oldest_failed_task_age': max(failed_ages) if failed_ages else 0.0,
                'avg_pending_task_age': sum(pending_ages) / len(pending_ages) if pending_ages else 0.0,
                'avg_completed_task_age': sum(completed_ages) / len(completed_ages) if completed_ages else 0.0,
                'avg_failed_task_age': sum(failed_ages) / len(failed_ages) if failed_ages else 0.0,
                'memory_usage_estimate': {
                    'pending_tasks_mb': len(self.pending_tasks) * 0.001,  # 估算每个任务1KB
                    'completed_tasks_mb': len(self.completed_tasks) * 0.001,
                    'failed_tasks_mb': len(self.failed_tasks) * 0.001,
                    'running_tasks_mb': len(self.running_tasks) * 0.001
                }
            }

    def shutdown(self, timeout: float = 30.0):
        """关闭处理器（改进版，防止卡死）"""
        if not self._running:
            logger.debug("AsyncProcessor已经关闭")
            return

        start_time = time.time()
        max_shutdown_time = min(timeout, 15.0)  # 限制最大关闭时间为15秒
        
        try:
            logger.info(f"正在关闭AsyncProcessor (超时: {max_shutdown_time}秒)...")

            # 设置停止标志（立即生效）
            self._running = False

            # 快速发送停止信号，不等待队列
            stop_signals_sent = 0
            for i in range(self.max_workers):
                try:
                    # 使用非阻塞方式发送停止信号
                    self.task_queue.put_nowait(StopTask())
                    stop_signals_sent += 1
                except:
                    # 队列满时跳过，工作线程会因为_running=False而停止
                    pass
            
            logger.debug(f"发送了 {stop_signals_sent} 个停止信号")

            # 快速等待工作线程结束
            alive_threads = []
            thread_timeout = max(1.0, max_shutdown_time * 0.6)  # 60%时间用于等待线程
            
            for i, thread in enumerate(self._worker_threads):
                if thread.is_alive():
                    remaining_time = max(0.1, thread_timeout - (time.time() - start_time))
                    if remaining_time <= 0:
                        alive_threads.append(i)
                        continue
                        
                    try:
                        thread.join(timeout=remaining_time)
                        if thread.is_alive():
                            alive_threads.append(i)
                        else:
                            logger.debug(f"工作线程{i}已停止")
                    except Exception as e:
                        logger.debug(f"等待工作线程{i}停止时出错: {e}")
                        alive_threads.append(i)

            if alive_threads:
                logger.warning(f"有{len(alive_threads)}个工作线程未能在{thread_timeout:.1f}秒内停止，将强制关闭")

            # 立即关闭线程池，不等待
            try:
                if hasattr(self, 'executor') and self.executor:
                    self.executor.shutdown(wait=False)
                    logger.debug("线程池已关闭")
            except Exception as e:
                logger.debug(f"关闭线程池时出现警告: {e}")

            # 快速清理资源
            try:
                with self._lock:
                    self._worker_threads.clear()
                    # 清理所有任务，避免内存泄漏
                    self.pending_tasks.clear()
                    self.running_tasks.clear()
                logger.debug("AsyncProcessor资源清理完成")
            except Exception as e:
                logger.debug(f"清理AsyncProcessor资源时出错: {e}")

            elapsed = time.time() - start_time
            logger.info(f"AsyncProcessor关闭完成 (耗时: {elapsed:.2f}秒)")

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"AsyncProcessor关闭时发生错误: {e} (耗时: {elapsed:.2f}秒)")
            # 确保即使出错也设置停止标志
            self._running = False
            
            # 强制清理
            try:
                if hasattr(self, 'executor') and self.executor:
                    self.executor.shutdown(wait=False)
                with self._lock:
                    self._worker_threads.clear()
                    self.pending_tasks.clear()
                    self.running_tasks.clear()
            except:
                pass

    def _notify_training_tasks_to_stop(self, wait_timeout: float = 20.0):
        """通知训练现场停止并等待收尾工作完成"""
        try:
            # 尝试导入训练管理器
            try:
                # 注意：这里需要根据实际模块结构调整导入路径
                # from modules.training.core.task_manager import get_training_task_manager
                logger.info("训练管理器导入功能需要根据实际模块结构调整")
            except ImportError:
                logger.warning("无法导入训练管理器，跳过训练任务停止通知")
            except Exception as e:
                logger.error(f"通知训练任务停止时出现错误: {e}")

        except Exception as e:
            logger.error(f"训练任务停止通知流程出现严重错误: {e}")


# 全局异步处理器实例
_global_async_processor: Optional[AsyncProcessor] = None


def get_async_processor() -> AsyncProcessor:
    """获取全局异步处理器实例

    Returns:
        AsyncProcessor: 异步处理器实例
    """
    global _global_async_processor
    if _global_async_processor is None:
        _global_async_processor = AsyncProcessor()
    return _global_async_processor


def create_async_processor(max_workers: int = 5, queue_size: int = 500) -> AsyncProcessor:
    """创建异步处理器实例

    Args:
        max_workers: 最大工作线程数
        queue_size: 任务队列大小

    Returns:
        AsyncProcessor: 异步处理器实例
    """
    return AsyncProcessor(max_workers, queue_size)