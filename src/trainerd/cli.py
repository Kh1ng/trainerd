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
from typing import Any

from . import __version__

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
        host=args.host,
        port=args.port,
        projects_config=args.projects_config,
        config=args.config,
    )
    return 0


def _cmd_submit(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {"triggered_by": args.triggered_by}
    if args.project:
        payload["project"] = args.project
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

    result = _request_json("POST", f"{args.server_url.rstrip('/')}/api/jobs", args.api_key, payload)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not args.wait:
        return 0
    watch_args = argparse.Namespace(
        server_url=args.server_url,
        job_id=result["job_id"],
        api_key=args.api_key,
        poll_seconds=args.poll_seconds,
        logs=args.logs,
        log_chars=args.log_chars,
        log_tail_lines=args.log_tail_lines,
        log_timeout_seconds=args.log_timeout_seconds,
    )
    return _cmd_watch(watch_args)


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
        if status in {"failed", "promoted", "validated", "completed"}:
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
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, help="Override the configured listen port.")
    serve.set_defaults(func=_cmd_serve)

    submit = sub.add_parser("submit", help="Submit a training job to a running trainerd server")
    submit.add_argument("--server-url", required=True)
    submit.add_argument("--api-key", default=os.environ.get("TRAINERD_API_KEY"))
    submit.add_argument(
        "--project",
        help="Startup-allowlisted project id (required in registry mode).",
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
