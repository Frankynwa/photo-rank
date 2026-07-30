"""统一日志配置 — 替代所有 print() 调用

用法：
    from core.logger import setup_logging, get_logger
    setup_logging()          # 默认 INFO 级别，输出到控制台
    logger = get_logger(__name__)

环境变量：
    LOG_LEVEL   — 日志级别（DEBUG / INFO / WARNING / ERROR），默认 INFO
    LOG_FILE    — 可选，日志文件路径（设置后同时输出到文件）
"""
import logging
import os
import sys


def setup_logging(level: str | None = None, log_file: str | None = None):
    """配置根日志系统

    Args:
        level:    日志级别，优先使用参数，其次读取环境变量 LOG_LEVEL，默认 INFO
        log_file: 日志文件路径，优先使用参数，其次读取环境变量 LOG_FILE
    """
    # 解析日志级别
    level_name = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    level_value = getattr(logging, level_name, logging.INFO)

    # 日志格式：时间 | 级别 | 模块 | 消息
    fmt = "[%(asctime)s] %(levelname)-8s %(name)-24s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # 获取根 logger
    root = logging.getLogger()
    root.setLevel(level_value)

    # 避免重复添加 handler（热重载场景）
    if not root.handlers:
        # 控制台 handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(console_handler)

    # 文件 handler（可选）
    resolved_log_file = log_file or os.environ.get("LOG_FILE")
    if resolved_log_file:
        # 检查是否已有同路径的 FileHandler
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == os.path.abspath(resolved_log_file)
            for h in root.handlers
        )
        if not has_file_handler:
            log_dir = os.path.dirname(resolved_log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(resolved_log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            root.addHandler(file_handler)

    # 抑制第三方库噪音
    for noisy in ("httpx", "httpcore", "urllib3", "PIL", "mediapipe"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的 logger"""
    return logging.getLogger(name)
