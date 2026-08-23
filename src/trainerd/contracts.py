"""trainerd API contracts: job payload + artifact manifest schemas.

trainerd is domain-neutral compute. Everything domain-specific arrives as
opaque strings substituted into command templates from training_config.yaml.
These schemas freeze the boundary so clients and the server can validate
independently of each other's release cadence.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr


class JobRequest(BaseModel):
    """Fields an HTTP client may submit; commands and paths stay server-owned."""

    model_config = ConfigDict(extra="forbid")

    project: StrictStr | None = None
    repo: StrictStr | None = None
    repo_url: StrictStr | None = None
    task: StrictStr | None = None
    version: StrictStr | None = None
    steps: list[StrictStr] | None = None
    branch: StrictStr | None = None
    markets: StrictStr | None = None
    extra_args: StrictStr | None = None
    force: StrictBool = False
    triggered_by: StrictStr = "api"


JOB_PAYLOAD_SCHEMA: dict[str, Any] = JobRequest.model_json_schema()
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def is_safe_identifier(value: object) -> bool:
    """Return whether value uses the shared bounded identifier syntax."""
    return isinstance(value, str) and _SAFE_IDENTIFIER.fullmatch(value) is not None

ARTIFACT_MANIFEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["run_label", "job_id", "produced_at", "artifacts"],
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
                "required": ["path", "sha256", "bytes"],
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
        if key in payload and payload[key] is None and any(
            choice.get("type") == "null" for choice in spec.get("anyOf", [])
        ):
            continue
        value_schema = spec
        if spec.get("type") is None:
            non_null = [
                choice
                for choice in spec.get("anyOf", [])
                if choice.get("type") != "null"
            ]
            value_schema = non_null[0] if len(non_null) == 1 else spec
        expected_type = value_schema.get("type")
        if key in payload and expected_type in _types:
            if not isinstance(payload[key], _types[expected_type]):
                problems.append(f"{key}: expected {expected_type}")
            elif expected_type == "array":
                item_spec = value_schema.get("items", {})
                for i, item in enumerate(payload[key]):
                    sub = validate_payload(item, item_spec) if item_spec.get("type") == "object" else []
                    if item_spec.get("type") == "string" and not isinstance(item, str):
                        sub = [f"expected string at index {i}"]
                    problems.extend(f"{key}[{i}]: {p}" for p in sub)
    return problems
