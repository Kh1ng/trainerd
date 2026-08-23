"""User-owned non-secret client connection profiles."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from .contracts import is_safe_identifier

_VERSION = 1


class ProfileError(ValueError):
    """A client profile file or requested profile is invalid."""


def profiles_path() -> Path:
    """Return the user-owned profile path, with an environment override for tools."""
    configured = os.environ.get("TRAINERD_PROFILES_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt" and os.environ.get("APPDATA"):
        root = Path(os.environ["APPDATA"])
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "trainerd" / "profiles.json"


def load_profiles() -> dict[str, dict[str, str]]:
    """Load validated connection defaults without accepting secrets or commands."""
    path = profiles_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"Could not read client profiles: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("version") != _VERSION:
        raise ProfileError("Client profiles have an unsupported schema")
    entries = raw.get("profiles")
    if not isinstance(entries, dict) or len(entries) > 64:
        raise ProfileError("Client profiles must be a mapping of at most 64 entries")
    profiles: dict[str, dict[str, str]] = {}
    for name, entry in entries.items():
        if not is_safe_identifier(name):
            raise ProfileError(f"Invalid profile name: {name!r}")
        if not isinstance(entry, dict) or set(entry) != {"server_url", "project"}:
            raise ProfileError(
                f"Profile {name!r} may contain only server_url and project"
            )
        profiles[name] = _validated_profile(entry["server_url"], entry["project"])
    return profiles


def set_profile(name: str, server_url: str, project: str) -> None:
    """Create or replace one non-secret connection profile."""
    if not is_safe_identifier(name):
        raise ProfileError("Profile name must be a safe identifier")
    profiles = load_profiles()
    if name not in profiles and len(profiles) >= 64:
        raise ProfileError("At most 64 client profiles may be stored")
    profiles[name] = _validated_profile(server_url, project)
    _write_profiles(profiles)


def remove_profile(name: str) -> bool:
    """Remove a profile and return whether it existed."""
    if not is_safe_identifier(name):
        raise ProfileError("Profile name must be a safe identifier")
    profiles = load_profiles()
    existed = name in profiles
    profiles.pop(name, None)
    _write_profiles(profiles)
    return existed


def _validated_profile(server_url: object, project: object) -> dict[str, str]:
    if not isinstance(server_url, str):
        raise ProfileError("Profile server_url must be a string")
    server_url = server_url.strip().rstrip("/")
    if any(character.isspace() or ord(character) < 32 for character in server_url):
        raise ProfileError("Profile server_url must not contain whitespace")
    parsed = urlsplit(server_url)
    try:
        parsed_port = parsed.port
    except ValueError as exc:
        raise ProfileError("Profile server_url contains an invalid port") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (parsed_port is not None and not 1 <= parsed_port <= 65535)
    ):
        raise ProfileError(
            "Profile server_url must be an HTTP(S) URL without credentials, query, or fragment"
        )
    if not is_safe_identifier(project):
        raise ProfileError("Profile project must be a safe identifier")
    return {"server_url": server_url, "project": project}


def _write_profiles(profiles: dict[str, dict[str, str]]) -> None:
    path = profiles_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=".profiles-", suffix=".tmp", text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as profile_file:
            json.dump(
                {"version": _VERSION, "profiles": profiles},
                profile_file,
                indent=2,
                sort_keys=True,
            )
            profile_file.write("\n")
            profile_file.flush()
            os.fsync(profile_file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
