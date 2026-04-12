#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端口管理工具模块
提供端口检测、进程查找和清理功能，支持跨平台操作
"""

import socket
import subprocess
import platform
import psutil
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class ProcessInfo:
    """进程信息数据类"""
    pid: int
    name: str
    cmdline: List[str]
    create_time: float
    connections: List[Dict[str, Any]]


class PortManager:
    """端口管理器 - 增强版本，支持进程清理"""

    def __init__(self):
        self.system = platform.system().lower()
        logger.info(f"初始化端口管理器，检测到系统: {self.system}")

    def check_port_available(self, port: int, host: str = 'localhost') -> bool:
        """检查端口是否可用
        
        Args:
            port: 端口号
            host: 主机地址，默认localhost
            
        Returns:
            bool: True表示端口可用，False表示被占用
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                return result != 0  # 0表示连接成功，即端口被占用
        except Exception as e:
            logger.warning(f"检查端口 {port} 时发生错误: {e}")
            return False

    def find_process_using_port(self, port: int) -> Optional[ProcessInfo]:
        """查找占用指定端口的进程
        
        Args:
            port: 端口号
            
        Returns:
            ProcessInfo: 进程信息，如果没有找到返回None
        """
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                try:
                    # 跳过系统空闲进程和无效进程
                    if proc.info['pid'] <= 0:
                        continue

                    # 跳过系统关键进程
                    proc_name = proc.info['name'].lower() if proc.info['name'] else ''
                    if 'system idle process' in proc_name or proc_name in ['system', 'idle']:
                        continue

                    connections = proc.connections(kind='inet')
                    for conn in connections:
                        if conn.laddr.port == port:
                            return ProcessInfo(
                                pid=proc.info['pid'],
                                name=proc.info['name'],
                                cmdline=proc.info['cmdline'] or [],
                                create_time=proc.info['create_time'],
                                connections=[{
                                    'local_addr': f"{conn.laddr.ip}:{conn.laddr.port}",
                                    'remote_addr': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                                    'status': conn.status
                                } for conn in connections]
                            )
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as e:
            logger.error(f"查找占用端口 {port} 的进程时发生错误: {e}")

        return None

    def kill_process_on_port(self, port: int, force: bool = False, timeout: int = 5) -> bool:
        """终止占用指定端口的进程
        
        Args:
            port: 端口号
            force: 是否强制终止（使用SIGKILL）
            timeout: 等待进程正常退出的超时时间（秒）
            
        Returns:
            bool: True表示成功终止，False表示失败
        """
        process_info = self.find_process_using_port(port)
        if not process_info:
            logger.info(f"端口 {port} 没有被任何进程占用")
            return True

        try:
            proc = psutil.Process(process_info.pid)
            proc_name = process_info.name

            logger.info(f"尝试终止占用端口 {port} 的进程: {proc_name} (PID: {process_info.pid})")

            # 检查是否是系统关键进程
            if self._is_system_process(process_info):
                logger.warning(f"拒绝终止系统进程: {proc_name} (PID: {process_info.pid})")
                return False

            if force:
                # 强制终止
                proc.kill()
                logger.info(f"强制终止进程 {proc_name} (PID: {process_info.pid})")
            else:
                # 优雅终止
                proc.terminate()
                logger.info(f"发送终止信号给进程 {proc_name} (PID: {process_info.pid})")

                # 等待进程退出
                try:
                    proc.wait(timeout=timeout)
                except psutil.TimeoutExpired:
                    logger.warning(f"进程 {proc_name} 在 {timeout} 秒内未退出，强制终止")
                    proc.kill()

            # 验证进程是否已终止
            time.sleep(0.5)
            if not proc.is_running():
                logger.info(f"成功终止进程 {proc_name} (PID: {process_info.pid})")
                return True
            else:
                logger.error(f"进程 {proc_name} (PID: {process_info.pid}) 仍在运行")
                return False

        except psutil.NoSuchProcess:
            logger.info(f"进程 (PID: {process_info.pid}) 已不存在")
            return True
        except psutil.AccessDenied:
            logger.error(f"没有权限终止进程 (PID: {process_info.pid})")
            return False
        except Exception as e:
            logger.error(f"终止进程时发生错误: {e}")
            return False

    def _is_system_process(self, process_info: ProcessInfo) -> bool:
        """检查是否是系统关键进程
        
        Args:
            process_info: 进程信息
            
        Returns:
            bool: True表示是系统进程，不应该被终止
        """
        # 系统关键进程名称列表
        system_processes = {
            'windows': ['system', 'csrss.exe', 'winlogon.exe', 'services.exe', 'lsass.exe', 'svchost.exe'],
            'linux': ['init', 'kthreadd', 'systemd', 'kernel', 'migration'],
            'darwin': ['kernel_task', 'launchd', 'kextd']
        }

        current_system = self.system
        if current_system == 'windows':
            current_system = 'windows'
        elif current_system == 'darwin':
            current_system = 'darwin'
        else:
            current_system = 'linux'

        critical_names = system_processes.get(current_system, [])
        process_name = process_info.name.lower()

        # 检查进程名
        if any(critical in process_name for critical in critical_names):
            return True

        # 检查PID（通常PID < 100的是系统进程）
        if process_info.pid < 100:
            return True

        return False

    def get_available_port(self, start_port: int = 5000, max_attempts: int = 100) -> Optional[int]:
        """获取可用端口
        
        Args:
            start_port: 起始端口号
            max_attempts: 最大尝试次数
            
        Returns:
            int: 可用的端口号，如果没有找到返回None
        """
        for i in range(max_attempts):
            port = start_port + i
            if port > 65535:  # 端口号上限
                break

            if self.check_port_available(port):
                logger.info(f"找到可用端口: {port}")
                return port

        logger.error(f"在 {start_port}-{start_port + max_attempts} 范围内未找到可用端口")
        return None

    def get_port_info(self, port: int) -> Dict[str, Any]:
        """获取端口详细信息
        
        Args:
            port: 端口号
            
        Returns:
            dict: 端口信息字典
        """
        info = {
            'port': port,
            'available': self.check_port_available(port),
            'process': None
        }

        if not info['available']:
            process_info = self.find_process_using_port(port)
            if process_info:
                info['process'] = {
                    'pid': process_info.pid,
                    'name': process_info.name,
                    'cmdline': ' '.join(process_info.cmdline),
                    'create_time': time.strftime('%Y-%m-%d %H:%M:%S',
                                                 time.localtime(process_info.create_time)),
                    'connections': process_info.connections
                }

        return info


# 便捷函数
def check_port_available(port: int, host: str = 'localhost') -> bool:
    """检查端口是否可用的便捷函数"""
    manager = PortManager()
    return manager.check_port_available(port, host)


def find_process_using_port(port: int) -> Optional[ProcessInfo]:
    """查找占用端口进程的便捷函数"""
    manager = PortManager()
    return manager.find_process_using_port(port)


def kill_process_on_port(port: int, force: bool = False) -> bool:
    """终止占用端口进程的便捷函数"""
    manager = PortManager()
    return manager.kill_process_on_port(port, force)


def get_available_port(start_port: int = 5000, max_attempts: int = 100) -> Optional[int]:
    """获取可用端口的便捷函数"""
    manager = PortManager()
    return manager.get_available_port(start_port, max_attempts)


def get_port_info(port: int) -> Dict[str, Any]:
    """获取端口信息的便捷函数"""
    manager = PortManager()
    return manager.get_port_info(port)


def check_and_cleanup_port(port: int, host: str = '0.0.0.0') -> Dict[str, Any]:
    """检查端口并清理占用进程的便捷函数
    
    Args:
        port: 端口号
        host: 主机地址
        
    Returns:
        dict: 包含清理结果的字典
            - success: bool, 是否成功
            - message: str, 结果消息
            - cleaned_processes: int, 清理的进程数量
            - process_info: dict, 被清理的进程信息（如果有）
    """
    manager = PortManager()
    result = {
        'success': False,
        'message': '',
        'cleaned_processes': 0,
        'process_info': None
    }

    try:
        # 检查端口是否可用
        if manager.check_port_available(port, host):
            result['success'] = True
            result['message'] = f'端口 {port} 可用'
            return result

        # 查找占用进程
        process_info = manager.find_process_using_port(port)
        if not process_info:
            result['success'] = True
            result['message'] = f'端口 {port} 未被占用'
            return result

        # 记录进程信息
        result['process_info'] = {
            'pid': process_info.pid,
            'name': process_info.name,
            'cmdline': ' '.join(process_info.cmdline)
        }

        # 尝试清理进程
        if manager.kill_process_on_port(port, force=False):
            result['success'] = True
            result['cleaned_processes'] = 1
            result['message'] = f'成功清理占用端口 {port} 的进程: {process_info.name} (PID: {process_info.pid})'
        else:
            result['success'] = False
            result['message'] = f'无法清理占用端口 {port} 的进程: {process_info.name} (PID: {process_info.pid})'

    except Exception as e:
        result['success'] = False
        result['message'] = f'检查和清理端口 {port} 时发生错误: {str(e)}'
        logger.error(f'端口清理错误: {e}')

    return result


if __name__ == "__main__":
    # 测试代码
    import sys

    if len(sys.argv) > 1:
        test_port = int(sys.argv[1])
    else:
        test_port = 5000

    print(f"测试端口: {test_port}")

    manager = PortManager()

    # 检查端口可用性
    available = manager.check_port_available(test_port)
    print(f"端口 {test_port} 可用: {available}")

    if not available:
        # 查找占用进程
        process_info = manager.find_process_using_port(test_port)
        if process_info:
            print(f"占用进程: {process_info.name} (PID: {process_info.pid})")
            print(f"命令行: {' '.join(process_info.cmdline)}")

    # 获取端口详细信息
    info = manager.get_port_info(test_port)
    print(f"端口信息: {info}")

    # 查找可用端口
    available_port = manager.get_available_port(test_port)
    print(f"可用端口: {available_port}")