from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TypeAlias, TypeVar, overload

from pydantic import ValidationError
import yaml

from .models import VnsConfig

ConfigScalar: TypeAlias = str | int | float | bool | None
ConfigValue: TypeAlias = ConfigScalar | list["ConfigValue"] | dict[str, "ConfigValue"]
ConfigDict: TypeAlias = dict[str, ConfigValue]

_T = TypeVar("_T")


class ConfigError(Exception):
    """Base exception for configuration loading and validation errors."""


class InvalidConfigError(ConfigError):
    """Raised when configuration content is syntactically or structurally invalid."""


class ConfigManager:
    """Load, validate, and query VNS YAML configuration."""

    def __init__(self, config_dict: Mapping[str, object] | None = None) -> None:
        raw_config = {} if config_dict is None else config_dict
        try:
            self._model = VnsConfig.model_validate(raw_config)
        except ValidationError as exc:
            raise InvalidConfigError(f"Configuration validation failed: {exc}") from exc
        self._config = self._normalize_mapping(
            self._model.model_dump(mode="python"),
            context="config",
        )

    @classmethod
    def load(cls, filepath: str | Path) -> "ConfigManager":
        """Load configuration from a YAML file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"Configuration path is not a file: {path}")

        try:
            with path.open("r", encoding="utf-8") as handle:
                raw_data = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise InvalidConfigError(
                f"Failed to parse configuration YAML at {path}: {exc}"
            ) from exc
        except OSError as exc:
            raise ConfigError(
                f"Failed to read configuration file at {path}: {exc}"
            ) from exc

        if raw_data is None:
            return cls({})
        if not isinstance(raw_data, Mapping):
            raise InvalidConfigError(
                "Configuration top-level YAML document must be a mapping."
            )

        return cls(raw_data)

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> "ConfigManager":
        """Backward-compatible alias for loading a YAML configuration file."""
        return cls.load(filepath)

    @staticmethod
    def _normalize_mapping(
        mapping: Mapping[str, object],
        *,
        context: str,
    ) -> ConfigDict:
        normalized: ConfigDict = {}
        for key, value in mapping.items():
            if not isinstance(key, str):
                raise InvalidConfigError(
                    f"{context} contains a non-string key: {key!r}."
                )
            child_context = f"{context}.{key}"
            normalized[key] = ConfigManager._normalize_value(
                value,
                context=child_context,
            )
        return normalized

    @staticmethod
    def _normalize_value(value: object, *, context: str) -> ConfigValue:
        if isinstance(value, Mapping):
            return ConfigManager._normalize_mapping(value, context=context)
        if isinstance(value, list):
            return [
                ConfigManager._normalize_value(item, context=f"{context}[{index}]")
                for index, item in enumerate(value)
            ]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        raise InvalidConfigError(
            f"{context} contains unsupported value type {type(value).__name__!r}."
        )

    @staticmethod
    def _split_key_path(key_path: str) -> list[str]:
        if not key_path:
            raise ValueError("key_path must not be empty.")

        parts = key_path.split(".")
        if any(not part for part in parts):
            raise ValueError(
                "key_path must use dot notation without empty path segments."
            )
        return parts

    @overload
    def get(self, key_path: str) -> ConfigValue | None: ...

    @overload
    def get(self, key_path: str, default: _T) -> ConfigValue | _T: ...

    def get(
        self,
        key_path: str,
        default: _T | None = None,
    ) -> ConfigValue | _T | None:
        """
        Get a value from nested configuration using dot notation.
        Example: ``config.get("camera.width", 640)``.
        """
        current: ConfigValue = self._config
        for key in self._split_key_path(key_path):
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return deepcopy(current)

    @property
    def data(self) -> ConfigDict:
        """Return a defensive copy of the full configuration mapping."""
        return deepcopy(self._config)

    @property
    def model(self) -> VnsConfig:
        """Return the validated Pydantic configuration model."""
        return self._model.model_copy(deep=True)
