"""trainerd API contracts: job payload + artifact manifest schemas.

trainerd is domain-neutral compute. Everything domain-specific arrives as
opaque strings substituted into command templates from training_config.yaml.
These schemas freeze the boundary so clients and the server can validate
independently of each other's release cadence.
"""
from __future__ import annotations

from typing import Any

JOB_PAYLOAD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # Optional startup-allowlisted project identifier. Clients never send
        # config paths or commands; omission preserves single-project clients.
        "project": {"type": "string"},
        # Explicitly insecure LAN mode resolves these to a daemon-owned checkout
        # and a repository-owned `.trainerd.yaml` task.
        "repo": {"type": "string"},
        "repo_url": {"type": "string"},
        "task": {"type": "string"},
        # Opaque run label. "cv_v42" means nothing to trainerd; it is only
        # substituted into {version} slots in step command templates.
        "version": {"type": "string"},
        "steps": {"type": "array", "items": {"type": "string"}},
        "branch": {"type": "string"},
        # Opaque template substitutions — domain vocabulary lives client-side.
        "markets": {"type": "string"},
        "extra_args": {"type": "string"},
        "force": {"type": "boolean"},
        "triggered_by": {"type": "string"},
    },
    "additionalProperties": False,
}

ARTIFACT_MANIFEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["run_label", "produced_at", "artifacts"],
    "properties": {
        "run_label": {"type": "string"},
        "job_id": {"type": "string"},
        "produced_at": {"type": "string"},
        # Optional domain-owned metadata. trainerd stores/transports it without
        # interpreting sport, market, target, or promotion policy fields.
        "metadata": {"type": "object"},
        "artifacts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string"},
                    "sha256": {"type": "string"},
                    "bytes": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}


def validate_payload(payload: dict[str, Any], schema: dict[str, Any] = JOB_PAYLOAD_SCHEMA) -> list[str]:
    """Minimal dependency-free schema check. Returns problems (empty = valid)."""
    problems: list[str] = []
    if not isinstance(payload, dict):
        return ["payload must be an object"]
    props = schema.get("properties", {})
    if not schema.get("additionalProperties", True):
        for key in payload:
            if key not in props:
                problems.append(f"unknown field: {key}")
    for key in schema.get("required", []):
        if key not in payload:
            problems.append(f"missing required field: {key}")
    _types = {"string": str, "boolean": bool, "integer": int, "array": list, "object": dict}
    for key, spec in props.items():
        if key in payload and spec.get("type") in _types:
            if not isinstance(payload[key], _types[spec["type"]]):
                problems.append(f"{key}: expected {spec['type']}")
            elif spec.get("type") == "array":
                item_spec = spec.get("items", {})
                for i, item in enumerate(payload[key]):
                    sub = validate_payload(item, item_spec) if item_spec.get("type") == "object" else []
                    if item_spec.get("type") == "string" and not isinstance(item, str):
                        sub = [f"expected string at index {i}"]
                    problems.extend(f"{key}[{i}]: {p}" for p in sub)
    return problems
