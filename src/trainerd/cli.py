"""CLI for the reusable training orchestration helper."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from . import __version__
from .managed_env import (
    ManagedEnvError,
    load_managed_env,
    set_managed_env,
    unset_managed_env,
)
from .profiles import ProfileError, load_profiles, remove_profile, set_profile
from .storage import JobStatus


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _request_json(
    method: str,
    url: str,
    api_key: str | None,
    payload: dict[str, Any] | None = None,
) -> Any:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=_headers(api_key), method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _request_text(url: str, api_key: str | None, *, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers=_headers(api_key), method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _compose_extra_args(args: argparse.Namespace) -> str:
    extra_args: list[str] = []
    if getattr(args, "extra_args", None):
        extra_args.append(str(args.extra_args).strip())
    if getattr(args, "training_preset", None):
        preset = str(args.training_preset).replace('"', '\\"')
        extra_args.append(f'--training-preset "{preset}"')
    if getattr(args, "shuffle_labels", False):
        extra_args.append("--shuffle-labels")
    if getattr(args, "point_in_time_strict", False):
        extra_args.append("--point-in-time-strict")
    if getattr(args, "event_group_split", False):
        extra_args.append("--event-group-split")
    if getattr(args, "dedupe", False):
        extra_args.append("--dedupe")
    return " ".join(extra_args)


def _cmd_serve(args: argparse.Namespace) -> int:
    from .server import main as serve_main

    serve_main(
        host=args.host or ("0.0.0.0" if args.lan else "127.0.0.1"),
        port=args.port,
        projects_config=args.projects_config,
        config=args.config,
        lan=args.lan,
        state_dir=args.state_dir,
        max_concurrent_jobs=args.max_concurrent_jobs,
        cpu_concurrency=args.cpu_concurrency,
        gpu_capacity=args.gpu_capacity,
        allowed_repos=args.allow_repo,
        lan_config=getattr(args, "lan_config", None),
    )
    return 0


def _cmd_policy_hash(args: argparse.Namespace) -> int:
    from .lan import LanConfigError, load_lan_server_config
    from .runtime import DaemonRuntime

    try:
        repositories = load_lan_server_config(Path(args.lan_config))
    except LanConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    runtime = DaemonRuntime()
    runtime.lan_repositories = repositories
    print(runtime.lan_policy_hash())
    return 0


def _env_state_dir(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    from .lan import default_state_dir

    return default_state_dir().expanduser().resolve()


def _cmd_env_set(args: argparse.Namespace) -> int:
    sources = sum(
        (
            args.value is not None,
            args.from_env is not None,
            bool(args.stdin),
        )
    )
    if sources != 1:
        raise ManagedEnvError(
            "Choose exactly one value source: --value, --from-env, or --stdin"
        )
    if args.from_env is not None:
        if args.from_env not in os.environ:
            raise ManagedEnvError(
                f"Source environment variable is not set: {args.from_env}"
            )
        value = os.environ[args.from_env]
    elif args.stdin:
        value = sys.stdin.read()
        if value.endswith("\n"):
            value = value[:-1]
        if value.endswith("\r"):
            value = value[:-1]
    else:
        value = args.value
    set_managed_env(_env_state_dir(args.state_dir), args.name, value)
    print(f"{args.name} configured")
    return 0


def _cmd_env_unset(args: argparse.Namespace) -> int:
    existed = unset_managed_env(_env_state_dir(args.state_dir), args.name)
    print(f"{args.name} {'removed' if existed else 'was not configured'}")
    return 0


def _cmd_env_list(args: argparse.Namespace) -> int:
    values = load_managed_env(_env_state_dir(args.state_dir))
    for name in sorted(values):
        print(name)
    return 0


def _cmd_submit(args: argparse.Namespace) -> int:
    server_url = args.server_url
    project = args.project
    if args.profile:
        profile = load_profiles().get(args.profile)
        if profile is None:
            raise ProfileError(f"Unknown client profile: {args.profile}")
        server_url = server_url or profile["server_url"]
        project = project or profile["project"]
    if not server_url:
        raise ProfileError("submit requires --server-url or --profile")
    payload: dict[str, Any] = {"triggered_by": args.triggered_by}
    if project:
        payload["project"] = project
    if args.repo:
        payload["repo"] = args.repo
    if args.task:
        payload["task"] = args.task
    if args.version and str(args.version).strip().lower() != "auto":
        payload["version"] = args.version
    if args.steps:
        payload["steps"] = [s.strip() for s in args.steps.split(",") if s.strip()]
    if args.branch:
        payload["branch"] = args.branch
    if args.markets:
        payload["markets"] = args.markets
    extra_args = _compose_extra_args(args)
    if extra_args:
        payload["extra_args"] = extra_args

    result = _request_json("POST", f"{server_url.rstrip('/')}/api/jobs", args.api_key, payload)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not args.wait:
        return 0
    watch_args = argparse.Namespace(
        server_url=server_url,
        job_id=result["job_id"],
        api_key=args.api_key,
        poll_seconds=args.poll_seconds,
        logs=args.logs,
        log_chars=args.log_chars,
        log_tail_lines=args.log_tail_lines,
        log_timeout_seconds=args.log_timeout_seconds,
    )
    return _cmd_watch(watch_args)


def _cmd_mcp(args: argparse.Namespace) -> int:
    server_url = args.server_url or os.environ.get("TRAINERD_SERVER_URL")
    if not server_url:
        raise ProfileError("mcp requires --server-url or TRAINERD_SERVER_URL")
    from .mcp import run

    run(server_url, os.environ.get("TRAINERD_API_KEY", ""))
    return 0


def _cmd_profile_set(args: argparse.Namespace) -> int:
    set_profile(args.name, args.server_url, args.project)
    print(f"{args.name} configured")
    return 0


def _cmd_profile_list(_args: argparse.Namespace) -> int:
    for name, profile in sorted(load_profiles().items()):
        print(f"{name}\t{profile['server_url']}\t{profile['project']}")
    return 0


def _cmd_profile_remove(args: argparse.Namespace) -> int:
    existed = remove_profile(args.name)
    print(f"{args.name} {'removed' if existed else 'was not configured'}")
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    base = args.server_url.rstrip("/")
    job_url = f"{base}/api/jobs/{args.job_id}"
    log_url = f"{base}/api/jobs/{args.job_id}/logs?tail={args.log_tail_lines}"
    last_log = ""

    while True:
        job = _request_json("GET", job_url, args.api_key)
        print(json.dumps(job, indent=2, sort_keys=True), flush=True)
        if args.logs:
            try:
                log_text = _request_text(log_url, args.api_key, timeout=args.log_timeout_seconds)
            except (TimeoutError, socket.timeout):
                log_text = last_log
            if log_text != last_log:
                print("--- logs ---", flush=True)
                print(log_text[-args.log_chars :].rstrip(), flush=True)
                last_log = log_text
        status = job.get("status")
        if JobStatus.is_terminal(status):
            return 0 if status != "failed" else 1
        time.sleep(args.poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trainerd")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the training orchestration API server")
    config_group = serve.add_mutually_exclusive_group()
    config_group.add_argument(
        "--projects-config",
        help="Path to a startup allowlist for multi-project registry mode.",
    )
    config_group.add_argument(
        "--config",
        help="Path to one project config for legacy single-project mode.",
    )
    config_group.add_argument(
        "--lan",
        action="store_true",
        help="Insecure zero-config LAN mode: clone HTTP Git repos and run repo-owned tasks.",
    )
    serve.add_argument(
        "--host",
        help="Listen address (default: 0.0.0.0 with --lan; otherwise 127.0.0.1).",
    )
    serve.add_argument("--port", type=int, help="Override the configured listen port.")
    serve.add_argument(
        "--state-dir",
        help="Managed LAN checkouts/jobs directory (only valid with --lan).",
    )
    serve.add_argument(
        "--max-concurrent-jobs",
        type=int,
        help="Daemon-wide LAN concurrency (only valid with --lan; default: 1).",
    )
    serve.add_argument(
        "--cpu-concurrency",
        type=int,
        help="CPU stage concurrency across all projects (default: 1).",
    )
    serve.add_argument(
        "--gpu-capacity",
        type=int,
        help="GPU capacity units shared across all projects (default: 1).",
    )
    serve.add_argument(
        "--allow-repo",
        action="append",
        help=(
            "Anonymous HTTP(S) Git repository allowed in LAN mode; repeat for "
            "additional repositories. Requires TRAINERD_API_KEY."
        ),
    )
    serve.add_argument(
        "--lan-config",
        help="Persistent server-owned LAN repository and task configuration.",
    )
    serve.set_defaults(func=_cmd_serve)

    policy_hash = sub.add_parser(
        "policy-hash",
        help="Print the complete policy hash for a LAN config.",
    )
    policy_hash.add_argument("--lan-config", required=True)
    policy_hash.set_defaults(func=_cmd_policy_hash)

    env = sub.add_parser(
        "env",
        help="Manage persistent environment variables injected into opted-in LAN tasks.",
    )
    env.add_argument(
        "--state-dir",
        help="Managed LAN state directory (defaults to the same path as serve --lan).",
    )
    env_sub = env.add_subparsers(dest="env_command", required=True)
    env_set = env_sub.add_parser("set", help="Store or replace one managed variable.")
    env_set.add_argument("name")
    value_source = env_set.add_mutually_exclusive_group(required=True)
    value_source.add_argument("--value")
    value_source.add_argument("--from-env")
    value_source.add_argument("--stdin", action="store_true")
    env_set.set_defaults(func=_cmd_env_set)
    env_unset = env_sub.add_parser("unset", help="Remove one managed variable.")
    env_unset.add_argument("name")
    env_unset.set_defaults(func=_cmd_env_unset)
    env_list = env_sub.add_parser(
        "list",
        help="List configured names without revealing values.",
    )
    env_list.set_defaults(func=_cmd_env_list)

    submit = sub.add_parser("submit", help="Submit a training job to a running trainerd server")
    submit.add_argument("--server-url")
    submit.add_argument("--profile", help="Use saved server URL and project defaults.")
    submit.add_argument("--api-key", default=os.environ.get("TRAINERD_API_KEY"))
    submit.add_argument(
        "--project",
        help="Startup-allowlisted project id (required in registry mode).",
    )
    submit.add_argument(
        "--repo",
        help="Allowlisted HTTP(S) Git repository URL (required in LAN mode).",
    )
    submit.add_argument(
        "--task",
        help="Repository-owned task name (required in LAN mode).",
    )
    submit.add_argument("--version", help="Optional. Omit or pass 'auto' to use the server's next vN.")
    submit.add_argument("--steps", help="Comma-separated subset of step ids to run")
    submit.add_argument("--branch", help="Override the git branch for the pull step")
    submit.add_argument("--markets", help="Opaque string substituted into command templates")
    submit.add_argument("--extra-args", help="Opaque command suffix appended to the job payload")
    submit.add_argument(
        "--training-preset",
        "--preset",
        dest="training_preset",
        help="Compatibility passthrough: append --training-preset to extra_args.",
    )
    submit.add_argument("--shuffle-labels", action="store_true", help="Compatibility passthrough for extra_args")
    submit.add_argument("--point-in-time-strict", action="store_true", help="Compatibility passthrough for extra_args")
    submit.add_argument("--event-group-split", action="store_true", help="Compatibility passthrough for extra_args")
    submit.add_argument("--dedupe", action="store_true", help="Compatibility passthrough for extra_args")
    submit.add_argument("--triggered-by", default="cli")
    submit.add_argument("--wait", action="store_true", help="Wait for the submitted job to finish")
    submit.add_argument("--poll-seconds", type=int, default=15)
    submit.add_argument("--logs", action="store_true")
    submit.add_argument("--log-chars", type=int, default=4000)
    submit.add_argument("--log-tail-lines", type=int, default=200)
    submit.add_argument("--log-timeout-seconds", type=int, default=10)
    submit.set_defaults(func=_cmd_submit)

    mcp = sub.add_parser("mcp", help="Expose a trainerd server as MCP tools over stdio")
    mcp.add_argument("--server-url")
    mcp.set_defaults(func=_cmd_mcp)

    profile = sub.add_parser("profile", help="Manage non-secret client connection profiles")
    profile_sub = profile.add_subparsers(dest="profile_command", required=True)
    profile_set = profile_sub.add_parser("set", help="Create or replace a profile")
    profile_set.add_argument("name")
    profile_set.add_argument("--server-url", required=True)
    profile_set.add_argument("--project", required=True)
    profile_set.set_defaults(func=_cmd_profile_set)
    profile_list = profile_sub.add_parser("list", help="List configured profiles")
    profile_list.set_defaults(func=_cmd_profile_list)
    profile_remove = profile_sub.add_parser("remove", help="Remove a profile")
    profile_remove.add_argument("name")
    profile_remove.set_defaults(func=_cmd_profile_remove)

    watch = sub.add_parser("watch", help="Poll job status from a running trainerd server")
    watch.add_argument("--server-url", required=True)
    watch.add_argument("--job-id", required=True)
    watch.add_argument("--api-key", default=os.environ.get("TRAINERD_API_KEY"))
    watch.add_argument("--poll-seconds", type=int, default=15)
    watch.add_argument("--logs", action="store_true")
    watch.add_argument("--log-chars", type=int, default=4000)
    watch.add_argument("--log-tail-lines", type=int, default=200)
    watch.add_argument("--log-timeout-seconds", type=int, default=10)
    watch.set_defaults(func=_cmd_watch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(detail or str(exc), file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ManagedEnvError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ProfileError as exc:
        print(str(exc), file=sys.stderr)
        return 2
