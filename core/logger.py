"""统一日志配置 — 替代所有 print() 调用"""
import logging
import sys


def setup_logging(level: str = "INFO"):
    """配置根日志系统"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)-8s %(name)-24s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # 抑制第三方库噪音
    for noisy in ("httpx", "httpcore", "urllib3", "PIL", "mediapipe"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
