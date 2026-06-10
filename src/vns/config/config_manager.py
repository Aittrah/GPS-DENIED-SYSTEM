import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigManager:
    """Manages VNS configurations, loading from YAML files with defaults."""

    def __init__(self, config_dict: Dict[str, Any] = None) -> None:
        self._config = config_dict or {}

    @classmethod
    def load(cls, filepath: str) -> "ConfigManager":
        """Load configuration from a YAML file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {filepath}")
        
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        
        return cls(data)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get value from nested configuration using dot notation.
        Example: config.get("camera.width", 640)
        """
        keys = key_path.split(".")
        val: Any = self._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    @property
    def data(self) -> Dict[str, Any]:
        """Returns the raw configuration dictionary."""
        return self._config
