from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    """Structured JSON formatter for machine-readable logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if isinstance(context, dict) and context:
            payload["context"] = context
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


class ContextLoggerAdapter(logging.LoggerAdapter[logging.Logger]):
    """Logger adapter that keeps structured context attached to log records."""

    def process(self, msg: object, kwargs: dict[str, Any]) -> tuple[object, dict[str, Any]]:
        extra = kwargs.setdefault("extra", {})
        incoming = extra.get("context")
        merged_context = dict(self.extra)
        if isinstance(incoming, dict):
            merged_context.update(incoming)
        extra["context"] = merged_context
        return msg, kwargs


def with_context(logger: logging.Logger, **context: Any) -> ContextLoggerAdapter:
    """Create a structured logger adapter with stable contextual fields."""
    return ContextLoggerAdapter(logger, context)


def _build_formatter(log_format: str) -> logging.Formatter:
    if log_format == "json":
        return JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
    return logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_logging(
    level: str = "INFO",
    log_dir: str | None = None,
    log_to_console: bool = True,
    log_format: str = "json",
    max_size_mb: int = 100,
    retention_count: int = 5,
) -> None:
    """Set up VNS logging with structured output and rotation."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = _build_formatter(log_format)
    handlers: list[logging.Handler] = []

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path / "vns.log",
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=max(retention_count, 1),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(level=numeric_level, handlers=handlers, force=True)
    with_context(
        logging.getLogger("vns"),
        log_format=log_format,
        log_dir=log_dir,
        retention_count=retention_count,
    ).info("VNS logging initialized")
