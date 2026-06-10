import logging
import os
from pathlib import Path
from typing import Optional

def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_to_console: bool = True
) -> None:
    """Set up loggers for VNS."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = []
    
    # Formatter for log messages
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
        
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / "vns.log")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
        
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True
    )
    
    logger = logging.getLogger("vns")
    logger.info("VNS Logging initialized at level %s", level)
