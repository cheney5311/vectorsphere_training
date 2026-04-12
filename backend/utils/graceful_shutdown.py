"""优雅关闭管理器

负责管理应用的优雅关闭流程，确保所有服务在应用退出时能够正确关闭。
"""

import logging
import signal
import atexit
import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class GracefulShutdownManager:
    """优雅关闭管理器"""
    
    def __init__(self, default_timeout: float = 30.0, force_exit_timeout: float = 35.0):
        """初始化优雅关闭管理器
        
        Args:
            default_timeout: 默认关闭超时时间（秒）
            force_exit_timeout: 强制退出超时时间（秒）
        """
        self.shutdown_handlers: List[tuple[Callable, str, float]] = []
        self.is_shutting_down = False
        self.default_timeout = default_timeout
        self.force_exit_timeout = force_exit_timeout
        self._shutdown_lock = threading.Lock()
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        try:
            # 保存原始信号处理器
            original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
            original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
            
            logger.info(f"成功设置信号处理器:")
            logger.info(f"  SIGTERM (15): 原处理器 {original_sigterm} -> 新处理器 {self._signal_handler}")
            logger.info(f"  SIGINT (2): 原处理器 {original_sigint} -> 新处理器 {self._signal_handler}")
            
            # 验证信号处理器是否正确设置
            current_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
            current_sigint = signal.signal(signal.SIGINT, self._signal_handler)
            
            if current_sigterm != self._signal_handler:
                logger.error(f"SIGTERM 信号处理器设置失败: 期望 {self._signal_handler}, 实际 {current_sigterm}")
            if current_sigint != self._signal_handler:
                logger.error(f"SIGINT 信号处理器设置失败: 期望 {self._signal_handler}, 实际 {current_sigint}")
                
        except Exception as e:
            logger.error(f"设置信号处理器失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        signal_names = {
            signal.SIGTERM: "SIGTERM",
            signal.SIGINT: "SIGINT"
        }
        signal_name = signal_names.get(signum, f"Unknown({signum})")
        logger.info(f"接收到信号 {signal_name} ({signum})，开始优雅关闭...")
        
        # 防止信号处理器被重复调用
        if self.is_shutting_down:
            logger.warning(f"已在关闭过程中，忽略信号 {signal_name}")
            return
            
        self.shutdown_now()
    
    def register_handler(self, handler: Callable, name: str = "Unknown", timeout: Optional[float] = None):
        """注册关闭处理器
        
        Args:
            handler: 关闭处理函数
            name: 处理器名称
            timeout: 处理器超时时间（秒），None表示使用默认超时
        """
        handler_timeout = timeout if timeout is not None else self.default_timeout
        self.shutdown_handlers.append((handler, name, handler_timeout))
        logger.debug(f"已注册关闭处理器: {name} (超时: {handler_timeout}秒)")
    
    def _execute_handler_with_timeout(self, handler: Callable, name: str, timeout: float) -> bool:
        """在超时限制内执行关闭处理器
        
        Args:
            handler: 关闭处理函数
            name: 处理器名称
            timeout: 超时时间（秒）
            
        Returns:
            bool: 是否成功执行
        """
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(handler)
                future.result(timeout=timeout)
                return True
        except FutureTimeoutError:
            logger.error(f"关闭处理器 {name} 执行超时 ({timeout}秒)")
            return False
        except Exception as e:
            logger.error(f"关闭处理器 {name} 执行失败: {e}")
            return False

    def shutdown_now(self):
        """立即执行关闭流程"""
        with self._shutdown_lock:
            if self.is_shutting_down:
                return
            
            self.is_shutting_down = True
        
        logger.info("开始执行优雅关闭流程...")
        start_time = time.time()
        
        # 启动强制退出定时器
        force_exit_timer = threading.Timer(self.force_exit_timeout, self._force_exit)
        force_exit_timer.start()
        
        try:
            # 按相反顺序执行关闭处理器
            for handler, name, timeout in reversed(self.shutdown_handlers):
                handler_start = time.time()
                logger.debug(f"执行关闭处理器: {name} (超时: {timeout}秒)")
                
                success = self._execute_handler_with_timeout(handler, name, timeout)
                
                elapsed = time.time() - handler_start
                if success:
                    logger.debug(f"关闭处理器执行完成: {name} (耗时: {elapsed:.2f}秒)")
                else:
                    logger.warning(f"关闭处理器执行失败: {name} (耗时: {elapsed:.2f}秒)")
            
            total_elapsed = time.time() - start_time
            logger.info(f"优雅关闭流程执行完成 (总耗时: {total_elapsed:.2f}秒)")
            
            # 取消强制退出定时器
            force_exit_timer.cancel()
            
            # 优雅关闭完成后，主动退出进程
            logger.info("优雅关闭完成，正在退出进程...")
            os._exit(0)
            
        except Exception as e:
            logger.error(f"优雅关闭流程执行异常: {e}")
            # 取消强制退出定时器
            force_exit_timer.cancel()
            # 异常情况下也要退出进程
            logger.error("优雅关闭异常，强制退出进程...")
            os._exit(1)
    
    def _force_exit(self):
        """强制退出进程"""
        logger.critical(f"优雅关闭超时 ({self.force_exit_timeout}秒)，强制退出进程")
        try:
            os._exit(1)
        except Exception as e:
            logger.error(f"强制退出失败: {e}")
            # 最后的手段
            os.kill(os.getpid(), signal.SIGKILL)


# 全局关闭管理器实例
_shutdown_manager: Optional[GracefulShutdownManager] = None


def init_graceful_shutdown(register_atexit: bool = False) -> GracefulShutdownManager:
    """初始化优雅关闭管理器
    
    Args:
        register_atexit: 是否注册到atexit（通常不需要，因为信号处理器已经足够）
    
    Returns:
        GracefulShutdownManager: 关闭管理器实例
    """
    global _shutdown_manager
    if _shutdown_manager is None:
        _shutdown_manager = GracefulShutdownManager()
        # 只在明确要求时才注册到atexit
        if register_atexit:
            atexit.register(_shutdown_manager.shutdown_now)
            logger.info("已注册 atexit 处理器")
    return _shutdown_manager


def get_shutdown_manager() -> Optional[GracefulShutdownManager]:
    """获取全局关闭管理器实例
    
    Returns:
        GracefulShutdownManager: 关闭管理器实例
    """
    return _shutdown_manager


def register_shutdown_handler(handler: Callable, name: str = "Unknown", timeout: Optional[float] = None):
    """注册关闭处理器
    
    Args:
        handler: 关闭处理函数
        name: 处理器名称
        timeout: 处理器超时时间（秒），None表示使用默认超时
    """
    manager = get_shutdown_manager()
    if manager:
        manager.register_handler(handler, name, timeout)
    else:
        logger.warning(f"关闭管理器未初始化，无法注册处理器: {name}")