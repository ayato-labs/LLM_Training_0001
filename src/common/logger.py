import datetime
import json
import sys
import traceback
from functools import wraps
from pathlib import Path
from typing import Any

from loguru import logger

# ログディレクトリがなければ作成
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# ロガー設定のリセット
logger.remove()

# コンソール出力（人間可読） - リアルタイム出力のため同期実行
logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    level="INFO",
    enqueue=False,  # 同期実行でリアルタイム出力
    backtrace=True,
    diagnose=True,
)


# 構造化JSONファイル出力（機械解析・トレーサビリティ用）
def json_serializer(record: dict[str, Any]) -> str:
    """追加フィールド付きカスタムJSONシリアライザ。"""
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
        "module": record["module"],
        "process_id": record["process"].id,
        "thread_id": record["thread"].id,
        "thread_name": record["thread"].name,
    }
    # 追加フィールドがあればマージ
    if "extra" in record:
        log_entry.update(record["extra"])
    # 例外情報があれば追加
    if record["exception"]:
        log_entry["exception"] = {
            "type": record["exception"].type.__name__ if record["exception"].type else None,
            "value": str(record["exception"].value) if record["exception"].value else None,
            "traceback": "".join(traceback.format_tb(record["exception"].traceback))
            if record["exception"].traceback
            else None,
        }
    return json.dumps(log_entry, ensure_ascii=False)


# 構造化JSONファイル出力（機械解析・トレーサビリティ用）
logger.add(
    "logs/app.json",
    serialize=json_serializer,
    level="DEBUG",
    rotation="10 MB",
    retention="30 days",
    enqueue=True,
    compression="gz",
)

# タイムスタンプ付きログファイル（実行ごとに1つ） - 人間可読
_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logger.add(
    f"logs/run_{_run_timestamp}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="50 MB",
    retention="7 days",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

# エラー専用ログファイル
logger.add(
    f"logs/errors_{_run_timestamp}.log",
    format=(
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
        "{name}:{function}:{line} - {message}\n{exception}"
    ),
    level="ERROR",
    rotation="10 MB",
    retention="30 days",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)


def log_exceptions(func):
    """完全なトレースバック付きで例外をログ出力するデコレータ。"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Unhandled exception in {func.__name__}: {e}")
            raise

    return wrapper


def log_function_call(log_args: bool = False, log_result: bool = False, level: str = "DEBUG"):
    """引数・戻り値のオプション付きで関数のエントリ/イグジットをログ出力するデコレータ。"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.log(
                level,
                f"Entering {func.__name__}",
                extra={"args": args if log_args else None, "kwargs": kwargs if log_args else None},
            )
            try:
                result = func(*args, **kwargs)
                logger.log(
                    level,
                    f"Exiting {func.__name__}",
                    extra={"result": result if log_result else None},
                )
                return result
            except Exception as e:
                logger.exception(f"Exception in {func.__name__}: {e}")
                raise

        return wrapper

    return decorator


class LogContext:
    """ログに文脈情報を追加するコンテキストマネージャ。"""

    def __init__(self, **context):
        self.context = context
        self._context_id = None

    def __enter__(self):
        self._context_id = logger.contextualize(**self.context).__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._context_id is not None:
            logger.contextualize(**self.context).__exit__(exc_type, None, None)


def get_logger(name: str | None = None):
    """オプションの名前バインディング付きでロガーインスタンスを取得。"""
    return logger.bind(module=name) if name else logger
