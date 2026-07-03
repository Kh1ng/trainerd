"""Job runner for the training server.

Executes training steps as subprocesses, captures logs, runs validation,
and optionally promotes model artifacts back to the git repo.
"""
from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import signal
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .config import TrainingConfig, load_config
from .storage import JobStatus, JobStore

log = logging.getLogger(__name__)


class JobRunner:
    def __init__(
        self,
        store: JobStore,
        config: TrainingConfig,
        *,
        config_path: Path | None = None,
    ) -> None:
        self._store = store
        self._config = config
        self._config_path = config_path
        # Track running subprocesses for cancellation: job_id -> asyncio.subprocess.Process | None
        self._running_procs: dict[str, asyncio.subprocess.Process | None] = {}

    def update_config(self, config: TrainingConfig, *, config_path: Path | None = None) -> None:
        self._config = config
        if config_path is not None:
            self._config_path = config_path

    async def cancel_job(self, job_id: str) -> bool:
        """Kill the running subprocess tree for a job, if any."""
        proc = self._running_procs.get(job_id)
        return await _terminate_proc_tree(proc)

    def _load_current_config(self) -> TrainingConfig:
        if self._config_path is None:
            return self._config
        self._config = load_config(self._config_path)
        return self._config

    async def run_job(self, job_id: str) -> None:
        job = self._store.get_job(job_id)
        if not job:
            return
        # Don't start if the job was already cancelled/marked failed
        if job["status"] in (JobStatus.FAILED, JobStatus.PROMOTED, JobStatus.COMPLETED):
            return
        config = self._load_current_config()
        log_path = config.log_dir / f"{job_id}.log"
        requested_steps = set(job.get("steps") or [])
        configured_steps = {s.id: s for s in config.steps}
        step_ids = [s.id for s in config.steps if s.id in requested_steps] if requested_steps else [s.id for s in config.steps]
        version = job.get("version", "unknown")
        branch = job.get("branch") or config.repo.branch
        markets = job.get("markets") or ""
        extra_args = job.get("extra_args") or ""

        with open(log_path, "w", encoding="utf-8") as logfile:

            def _log(msg: str) -> None:
                ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                line = f"[{ts}] {msg}\n"
                logfile.write(line)
                logfile.flush()
                log.info("[job:%s] %s", job_id, msg)

            if requested_steps and not step_ids:
                msg = f"None of the requested steps exist in active config: {sorted(requested_steps)}"
                _log(f"SKIPPED (no-op): {msg}")
                self._store.set_completed(job_id)
                return

            _log(f"=== Job {job_id} starting. version={version} steps={step_ids} ===")

            for step_id in step_ids:
                config = self._load_current_config()
                configured_steps = {s.id: s for s in config.steps}
                step = configured_steps.get(step_id)
                if step is None:
                    self._store.set_failed(job_id, f"Step '{step_id}' missing from current config")
                    _log(f"FAILED: step '{step_id}' no longer exists in {self._config_path or 'active config'}")
                    return
                self._store.set_running(job_id, step.id)
                _log(f"--- Step: {step.name} ---")
                repo_path = str(config.repo.local_path)
                work_dir = str(config.work_dir)
                cmd = _resolve_cmd(
                    step.cmd,
                    version=version,
                    repo_path=repo_path,
                    work_dir=work_dir,
                    branch=branch,
                    markets=markets,
                    extra_args=extra_args,
                )
                resolved_env = {
                    k: _resolve_template(v, version=version, repo_path=repo_path, work_dir=work_dir, extra_args=extra_args)
                    for k, v in (step.env or {}).items()
                } if step.env else None
                resolved_env = _with_repo_pythonpath(repo_path, resolved_env)
                resolved_cwd = _resolve_template(step.cwd or "", version=version, repo_path=repo_path, work_dir=work_dir, extra_args=extra_args) or None
                _log(f"cmd: {cmd}")
                ok = await _run_cmd(cmd, cwd=resolved_cwd, logfile=logfile, env=resolved_env, timeout=step.timeout_seconds,
                                    proc_store=self._running_procs, job_id=job_id)
                self._running_procs.pop(job_id, None)
                if not ok:
                    self._store.set_failed(job_id, f"Step '{step.id}' failed — see job log")
                    _log(f"FAILED at step: {step.id}")
                    return

            should_validate_and_promote = "train" in step_ids
            if not should_validate_and_promote:
                _log("=== All requested steps completed. Skipping validation/promotion (no train step requested). ===")
                self._store.set_completed(job_id)
                return

            _log("=== All steps completed. Running validation... ===")
            self._store.set_completed(job_id)

            config = self._load_current_config()
            if config.validation:
                ok, result = await _validate(config.validation, version=version, logfile=logfile)
                if ok:
                    self._store.set_validated(job_id, result)
                    _log(f"Validation PASSED: {result}")
                else:
                    _log(f"Validation FAILED — models not promoted. Result: {result}")
                    self._store.set_failed(job_id, f"Validation failed: {json.dumps(result)[:500]}")
                    return

            if config.promotion and (config.validation is None or self._store.get_job(job_id, "status") == JobStatus.VALIDATED):
                await self.promote_job(job_id, logfile=logfile, version=version)

            _log(f"=== Job {job_id} DONE. status={self._store.get_job(job_id, 'status')} ===")

    async def promote_job(self, job_id: str, logfile=None, version: str | None = None) -> None:
        if not self._config.promotion:
            return
        if version is None:
            version = self._store.get_job(job_id, "version") or "unknown"
        prom = self._config.promotion
        source_dir = Path(_resolve_cmd(prom.source_dir, version=version))
        repo_path = Path(self._config.repo.local_path)
        dest_dir = repo_path / _resolve_cmd(prom.repo_subdir, version=version)

        def _log(msg: str) -> None:
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            line = f"[{ts}] {msg}\n"
            if logfile:
                logfile.write(line)
                logfile.flush()
            log.info("[job:%s promote] %s", job_id, msg)

        _log(f"Promoting {source_dir} → {dest_dir}")
        if not source_dir.exists():
            _log(f"ERROR: source_dir does not exist: {source_dir}")
            self._store.set_failed(job_id, f"Promotion failed: source_dir missing: {source_dir}")
            return

        branch = prom.branch
        commit_msg = prom.commit_message.replace("{version}", version).replace(
            "{timestamp}", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        )

        git_env = {
            **os.environ,
            "GIT_AUTHOR_NAME": os.environ.get("GIT_AUTHOR_NAME", "trainerd"),
            "GIT_AUTHOR_EMAIL": os.environ.get("GIT_AUTHOR_EMAIL", "trainerd@local"),
            "GIT_COMMITTER_NAME": os.environ.get("GIT_COMMITTER_NAME", os.environ.get("GIT_AUTHOR_NAME", "trainerd")),
            "GIT_COMMITTER_EMAIL": os.environ.get("GIT_COMMITTER_EMAIL", os.environ.get("GIT_AUTHOR_EMAIL", "trainerd@local")),
        }

        # Pull before copying artifacts so there are no unstaged changes during rebase
        pre_cmds = [
            ["git", "-C", str(repo_path), "checkout", branch],
            ["git", "-C", str(repo_path), "pull", "--rebase", "origin", branch],
        ]
        for cmd in pre_cmds:
            _log(f"git: {' '.join(cmd[2:])}")
            result = subprocess.run(cmd, capture_output=True, text=True, env=git_env)
            _log(result.stdout.strip())
            if result.returncode != 0:
                _log(f"ERROR: {result.stderr.strip()}")
                self._store.set_failed(job_id, f"Git promotion failed at: {cmd}")
                return

        dest_dir.mkdir(parents=True, exist_ok=True)
        _copy_excluding(source_dir, dest_dir, excludes=prom.excludes, log_fn=_log)

        git_cmds = [
            ["git", "-C", str(repo_path), "add", str(dest_dir)],
            ["git", "-C", str(repo_path), "commit", "-m", commit_msg],
        ]
        if prom.push:
            git_cmds.append(["git", "-C", str(repo_path), "push", "origin", branch])

        for cmd in git_cmds:
            _log(f"git: {' '.join(cmd[2:])}")
            result = subprocess.run(cmd, capture_output=True, text=True, env=git_env)
            _log(result.stdout.strip())
            if result.returncode != 0:
                _log(f"ERROR: {result.stderr.strip()}")
                self._store.set_failed(job_id, f"Git promotion failed at: {cmd}")
                return

        ref_out = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
        self._store.set_promoted(job_id, ref_out)
        _log(f"Promoted. git ref: {ref_out}")


def _resolve_template(
    template: str,
    version: str = "",
    repo_path: str = "",
    work_dir: str = "",
    branch: str = "",
    markets: str = "",
    extra_args: str = "",
) -> str:
    return (
        template
        .replace("{version}", version)
        .replace("{repo_path}", repo_path)
        .replace("{work_dir}", work_dir)
        .replace("{markets_flag}", f"--markets {markets}" if markets else "")
        .replace("{extra_args}", extra_args)
    )


def _resolve_cmd(
    template: str,
    version: str = "",
    repo_path: str = "",
    work_dir: str = "",
    branch: str = "",
    markets: str = "",
    extra_args: str = "",
) -> str:
    return _resolve_template(
        template,
        version=version,
        repo_path=repo_path,
        work_dir=work_dir,
        branch=branch,
        markets=markets,
        extra_args=extra_args,
    )


def _with_repo_pythonpath(repo_path: str | None, env: dict[str, str] | None) -> dict[str, str]:
    out = dict(env or {})
    if not repo_path:
        return out
    existing = out.get("PYTHONPATH") or os.environ.get("PYTHONPATH") or ""
    sep = os.pathsep
    parts = [repo_path, *[p for p in existing.split(sep) if p]]
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part in seen:
            continue
        seen.add(part)
        deduped.append(part)
    out["PYTHONPATH"] = sep.join(deduped)
    return out


async def _run_cmd(
    cmd: str,
    cwd: str | None,
    logfile,
    env: dict[str, str] | None,
    timeout: int,
    *,
    proc_store: dict[str, asyncio.subprocess.Process | None] | None = None,
    job_id: str | None = None,
) -> bool:
    full_env = {**os.environ, **(env or {})}
    try:
        proc_kwargs: dict[str, object] = {}
        if os.name == "nt":
            proc_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            proc_kwargs["start_new_session"] = True
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=full_env,
            **proc_kwargs,
        )
        if proc_store is not None and job_id is not None:
            proc_store[job_id] = proc
        try:
            async with asyncio.timeout(timeout):
                async for line in proc.stdout:  # type: ignore[union-attr]
                    decoded = line.decode("utf-8", errors="replace").rstrip()
                    logfile.write(decoded + "\n")
                    logfile.flush()
        except asyncio.TimeoutError:
            await _terminate_proc_tree(proc)
            logfile.write(f"\n[TIMEOUT after {timeout}s]\n")
            logfile.flush()
            return False
        await proc.wait()
        return proc.returncode == 0
    except Exception as exc:
        logfile.write(f"\n[EXCEPTION: {exc}]\n")
        logfile.flush()
        return False


async def _terminate_proc_tree(proc: asyncio.subprocess.Process | None) -> bool:
    if proc is None or proc.returncode is not None:
        return False
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            if os.name == "nt":
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()
        return True
    except (ProcessLookupError, PermissionError):
        return False


async def _validate(val_cfg, version: str, logfile) -> tuple[bool, dict]:
    cmd = _resolve_cmd(val_cfg.cmd, version=version)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_path = tmp.name

    full_cmd = f"{cmd} --output-json {out_path}" if val_cfg.output_is_json else cmd
    full_env = {**os.environ, **_with_repo_pythonpath(str(getattr(val_cfg, "cwd", "") or ""), getattr(val_cfg, "env", None) or {})}
    try:
        proc = await asyncio.create_subprocess_shell(
            full_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=val_cfg.cwd,
            env=full_env,
        )
        async for line in proc.stdout:  # type: ignore[union-attr]
            logfile.write(line.decode("utf-8", errors="replace"))
        logfile.flush()
        await proc.wait()
        returncode = proc.returncode

        result: dict = {"returncode": returncode}
        if val_cfg.output_is_json and Path(out_path).exists():
            try:
                result.update(json.loads(Path(out_path).read_text()))
            except Exception:
                pass
        Path(out_path).unlink(missing_ok=True)

        if val_cfg.output_is_json:
            passed = str(result.get(val_cfg.success_key, "")).lower() == val_cfg.success_value.lower()
        else:
            passed = returncode == 0

        return passed, result
    except Exception as exc:
        return False, {"error": str(exc)}


def _copy_excluding(src: Path, dst: Path, excludes: list[str], log_fn) -> None:
    for item in src.rglob("*"):
        if not item.is_file():
            continue
        name = item.name
        if any(fnmatch.fnmatch(name, pattern) for pattern in excludes):
            log_fn(f"  skip {item.relative_to(src)}")
            continue
        rel = item.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
        log_fn(f"  copy {rel}")
