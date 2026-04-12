# -*- coding: utf-8 -*-
"""
Scheduler模块异常处理
"""

class SchedulerError(Exception):
    """调度器基础异常"""


class TaskNotFoundError(SchedulerError):
    """任务未找到异常"""


class TaskAlreadyScheduledError(SchedulerError):
    """任务已调度异常"""


class TaskExecutionError(SchedulerError):
    """任务执行异常"""


class InvalidScheduleTimeError(SchedulerError):
    """无效调度时间异常"""


class SchedulerNotRunningError(SchedulerError):
    """调度器未运行异常"""