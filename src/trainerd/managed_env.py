"""Persistent job environment for LAN-mode tasks.

Values are written once by a local operator and are never returned by the CLI
or API. Repository manifests opt in to individual names with ``required_env``.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MAX_ENV_VARS = 128
_MAX_VALUE_BYTES = 16_384
_SCHEMA_VERSION = 1


class ManagedEnvError(ValueError):
    """Managed environment state is invalid or incomplete."""


def managed_env_path(state_dir: Path) -> Path:
    return state_dir.expanduser().resolve() / "job-env.json"


def validate_env_name(name: str) -> str:
    if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
        raise ManagedEnvError(
            "Environment variable name must contain only letters, numbers, and underscores"
        )
    return name


def load_managed_env(state_dir: Path) -> dict[str, str]:
    path = managed_env_path(state_dir)
    if not path.exists():
        return {}
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManagedEnvError(f"Could not read managed job environment: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("version") != _SCHEMA_VERSION:
        raise ManagedEnvError("Managed job environment has an unsupported schema")
    values = raw.get("env")
    if not isinstance(values, dict) or len(values) > _MAX_ENV_VARS:
        raise ManagedEnvError("Managed job environment is not a bounded mapping")
    result: dict[str, str] = {}
    for name, value in values.items():
        validate_env_name(name)
        if (
            not isinstance(value, str)
            or "\x00" in value
            or len(value.encode("utf-8")) > _MAX_VALUE_BYTES
        ):
            raise ManagedEnvError(f"Managed environment value is invalid: {name}")
        result[name] = value
    return result


def set_managed_env(state_dir: Path, name: str, value: str) -> None:
    name = validate_env_name(name)
    if not isinstance(value, str) or "\x00" in value:
        raise ManagedEnvError("Environment value must be a string without NUL bytes")
    if len(value.encode("utf-8")) > _MAX_VALUE_BYTES:
        raise ManagedEnvError(
            f"Environment value exceeds the {_MAX_VALUE_BYTES}-byte limit"
        )
    values = load_managed_env(state_dir)
    if name not in values and len(values) >= _MAX_ENV_VARS:
        raise ManagedEnvError(f"At most {_MAX_ENV_VARS} variables may be stored")
    values[name] = value
    _write_managed_env(state_dir, values)


def unset_managed_env(state_dir: Path, name: str) -> bool:
    name = validate_env_name(name)
    values = load_managed_env(state_dir)
    existed = name in values
    values.pop(name, None)
    _write_managed_env(state_dir, values)
    return existed


def _write_managed_env(state_dir: Path, values: dict[str, str]) -> None:
    state_dir = state_dir.expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    target = managed_env_path(state_dir)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=state_dir,
        prefix=".job-env-",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                {"version": _SCHEMA_VERSION, "env": values},
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        os.replace(temporary, target)
        try:
            target.chmod(0o600)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)
