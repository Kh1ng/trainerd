from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path

import yaml

import trainerd.runner as runner_module
from trainerd.config import RepoConfig, StepConfig, TrainingConfig
from trainerd.lan import load_lan_task, normalize_repo_url, repo_key
from trainerd.runner import JobRunner, StageQueuePool
from trainerd.storage import JobStore


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_lan_manifest(repo: Path, suffix: str = "") -> None:
    label = f"-{suffix}" if suffix else ""
    (repo / ".trainerd.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "tasks": {
                    "train": {
                        "max_concurrent_jobs": 2,
                        "steps": [
                            {
                                "id": "prepare",
                                "cmd": f"prepare{label} --work-dir {{work_dir}}",
                                "cwd": ".",
                                "queue": "cpu",
                            },
                            {
                                "id": "train",
                                "cmd": f"train{label} --work-dir {{work_dir}}",
                                "cwd": ".",
                                "queue": "gpu",
                            },
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _load_lan_config(state_dir: Path, repo: Path, repo_url: str) -> TrainingConfig:
    key = repo_key(repo_url)
    work_dir = state_dir / "work" / key / "train"
    log_dir = state_dir / "jobs" / f"lan-{key}-train"
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return load_lan_task(
        repo / ".trainerd.yaml",
        task="train",
        project=f"lan-{key}-train",
        repo_url=repo_url,
        repo_path=repo,
        branch=_git(repo, "branch", "--show-current"),
        work_dir=work_dir,
        log_dir=log_dir,
    )


def test_cpu_preparation_overlaps_gpu_without_concurrent_gpu_stages(
    tmp_path: Path, monkeypatch
) -> None:
    config = TrainingConfig(
        project="test",
        repo=RepoConfig("", "main", str(tmp_path)),
        work_dir=tmp_path / "work",
        steps=[
            StepConfig("prepare", "Prepare", "prepare", queue="cpu"),
            StepConfig("train", "Train", "train", queue="gpu"),
        ],
        validation=None,
        promotion=None,
        api_key="",
        server_port=7860,
        log_dir=tmp_path / "logs",
        max_concurrent_jobs=2,
    )
    config.log_dir.mkdir()
    store = JobStore(tmp_path / "jobs.db")
    stage_queues = {"prepare": "cpu", "train": "gpu"}
    store.create_job("job-a", ["prepare", "train"], "v1", stage_queues=stage_queues)
    store.create_job("job-b", ["prepare", "train"], "v2", stage_queues=stage_queues)
    intervals: dict[tuple[str, str], tuple[float, float]] = {}
    artifact_dirs: dict[tuple[str, str], str] = {}
    pool = StageQueuePool(cpu=1, gpu=1)
    occupancy: list[dict[str, dict[str, int | float]]] = []
    gpu_running = 0
    max_gpu_running = 0
    first_gpu_started = asyncio.Event()
    second_cpu_started = asyncio.Event()

    async def fake_run_cmd(cmd, cwd, logfile, env, timeout, **kwargs):
        nonlocal gpu_running, max_gpu_running
        job_id = env["TRAINERD_JOB_ID"]
        artifact_dirs[(job_id, cmd)] = env["TRAINERD_ARTIFACT_DIR"]
        occupancy.append(pool.snapshot())
        start = time.perf_counter()
        if cmd == "train":
            gpu_running += 1
            max_gpu_running = max(max_gpu_running, gpu_running)
        if (job_id, cmd) == ("job-a", "train"):
            first_gpu_started.set()
            await asyncio.wait_for(second_cpu_started.wait(), timeout=5)
        if (job_id, cmd) == ("job-b", "prepare"):
            second_cpu_started.set()
        await asyncio.sleep(0.01)
        if cmd == "train":
            gpu_running -= 1
        intervals[(job_id, cmd)] = (start, time.perf_counter())
        return True

    monkeypatch.setattr(runner_module, "_run_cmd", fake_run_cmd)
    runner = JobRunner(store, config, queues=pool)

    async def run_jobs() -> None:
        first = asyncio.create_task(runner.run_job("job-a"))
        await asyncio.wait_for(first_gpu_started.wait(), timeout=5)
        await asyncio.gather(first, runner.run_job("job-b"))

    asyncio.run(run_jobs())

    assert max_gpu_running == 1
    assert any(sample["cpu"]["occupancy"] == 1 for sample in occupancy)
    assert any(sample["gpu"]["occupancy"] == 1 for sample in occupancy)
    assert pool.snapshot()["gpu"]["occupancy"] == 0
    assert second_cpu_started.is_set()
    assert any(
        cpu_start < gpu_end and gpu_start < cpu_end
        for cpu_job in ("job-a", "job-b")
        for gpu_job in ("job-a", "job-b")
        if cpu_job != gpu_job
        for cpu_start, cpu_end in [intervals[(cpu_job, "prepare")]]
        for gpu_start, gpu_end in [intervals[(gpu_job, "train")]]
    )
    for job_id in ("job-a", "job-b"):
        assert artifact_dirs[(job_id, "prepare")] == artifact_dirs[(job_id, "train")]
        job = store.get_job(job_id)
        assert job["status"] == "completed"
        assert job["stages"]["prepare"]["status"] == "completed"
        assert job["stages"]["train"]["status"] == "completed"
        assert job["stages"]["train"]["queue_wait_seconds"] >= 0
        assert job["stages"]["train"]["duration_seconds"] > 0


def test_same_repo_jobs_use_pinned_worktrees_and_isolated_work_dirs(
    tmp_path: Path, monkeypatch
) -> None:
    state_dir = tmp_path / "state"
    repo_url = normalize_repo_url("http://git.local/team/repo.git")
    repo = state_dir / "repos" / repo_key(repo_url)
    repo.parent.mkdir(parents=True)
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    _write_lan_manifest(repo, "one")
    (repo / "revision.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", ".trainerd.yaml", "revision.txt")
    _git(repo, "commit", "-m", "one")
    first_revision = _git(repo, "rev-parse", "HEAD")
    _write_lan_manifest(repo, "two")
    (repo / "revision.txt").write_text("two\n", encoding="utf-8")
    _git(repo, "commit", "-am", "two")
    second_revision = _git(repo, "rev-parse", "HEAD")

    config = _load_lan_config(state_dir, repo, repo_url)
    store = JobStore(tmp_path / "jobs.db")
    stage_queues = {"prepare": "cpu", "train": "gpu"}
    store.create_job(
        "job-a",
        ["prepare", "train"],
        "v1",
        repo_sha=first_revision,
        stage_queues=stage_queues,
    )
    store.create_job(
        "job-b",
        ["prepare", "train"],
        "v2",
        repo_sha=second_revision,
        stage_queues=stage_queues,
    )
    pool = StageQueuePool(cpu=1, gpu=1)
    first_gpu_started = asyncio.Event()
    second_cpu_started = asyncio.Event()
    cpu_running = 0
    max_cpu_running = 0
    gpu_running = 0
    max_gpu_running = 0
    checkouts: dict[str, Path] = {}
    work_dirs: dict[str, Path] = {}

    async def fake_run_cmd(cmd, cwd, logfile, env, timeout, **kwargs):
        nonlocal cpu_running, max_cpu_running, gpu_running, max_gpu_running
        job_id = env["TRAINERD_JOB_ID"]
        checkout = Path(cwd)
        checkouts[job_id] = checkout
        work_dirs[job_id] = Path(env["TRAINERD_ARTIFACT_DIR"])
        command = cmd.split()[0]
        stage, revision_label = command.split("-")
        assert str(work_dirs[job_id]) in cmd
        assert _git(checkout, "rev-parse", "HEAD") == env["TRAINERD_REPO_SHA"]
        assert revision_label == {"job-a": "one", "job-b": "two"}[job_id]
        if stage == "prepare":
            cpu_running += 1
            max_cpu_running = max(max_cpu_running, cpu_running)
        if stage == "train":
            gpu_running += 1
            max_gpu_running = max(max_gpu_running, gpu_running)
        if (job_id, stage) == ("job-a", "train"):
            first_gpu_started.set()
            await asyncio.wait_for(second_cpu_started.wait(), timeout=5)
        if (job_id, stage) == ("job-b", "prepare"):
            second_cpu_started.set()
            await asyncio.wait_for(first_gpu_started.wait(), timeout=5)
        await asyncio.sleep(0.01)
        if stage == "prepare":
            cpu_running -= 1
        if stage == "train":
            gpu_running -= 1
        return True

    monkeypatch.setattr(runner_module, "_run_cmd", fake_run_cmd)
    runner = JobRunner(store, config, queues=pool)

    async def run_jobs() -> None:
        await asyncio.gather(runner.run_job("job-a"), runner.run_job("job-b"))

    asyncio.run(run_jobs())

    assert max_cpu_running == 1
    assert max_gpu_running == 1
    assert checkouts["job-a"] != checkouts["job-b"]
    assert work_dirs["job-a"] != work_dirs["job-b"]
    assert not checkouts["job-a"].exists()
    assert not checkouts["job-b"].exists()
    assert work_dirs["job-a"].is_dir()
    assert work_dirs["job-b"].is_dir()
    assert (repo / "revision.txt").read_text(encoding="utf-8") == "two\n"


def test_restart_reuses_pinned_job_worktree(tmp_path: Path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    repo_url = normalize_repo_url("http://git.local/team/repo.git")
    repo = state_dir / "repos" / repo_key(repo_url)
    repo.parent.mkdir(parents=True)
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    _write_lan_manifest(repo)
    (repo / "tracked.txt").write_text("pinned\n", encoding="utf-8")
    _git(repo, "add", ".trainerd.yaml", "tracked.txt")
    _git(repo, "commit", "-m", "pinned")
    revision = _git(repo, "rev-parse", "HEAD")
    config = _load_lan_config(state_dir, repo, repo_url)
    work_dir = config.work_dir
    checkout = work_dir / "job-a" / "checkout"
    checkout.parent.mkdir(parents=True)
    _git(repo, "worktree", "add", "--detach", str(checkout), revision)
    (checkout / "restart.marker").write_text("reuse me\n", encoding="utf-8")
    interrupted_checkout = work_dir / "job-b" / "checkout"
    interrupted_checkout.parent.mkdir(parents=True)
    _git(repo, "worktree", "add", "--detach", str(interrupted_checkout), revision)
    retained_artifact = interrupted_checkout.parent / "result.json"
    retained_artifact.write_text("{}\n", encoding="utf-8")

    store = JobStore(tmp_path / "jobs.db")
    store.create_job(
        "job-a",
        ["prepare", "train"],
        "v1",
        repo_sha=revision,
        stage_queues={"prepare": "cpu", "train": "gpu"},
    )
    store.set_stage_running("job-a", "prepare")
    store.set_stage_completed("job-a", "prepare", 1.0)
    store.set_running("job-a", "prepare")
    store.create_job(
        "job-b",
        ["prepare", "train"],
        "v2",
        repo_sha=revision,
        stage_queues={"prepare": "cpu", "train": "gpu"},
    )
    store.set_running("job-b", "prepare")
    store.set_stage_running("job-b", "prepare")
    runner = JobRunner(store, config)

    assert runner.recover_interrupted_jobs() == ["job-a", "job-b"]
    assert store.get_job("job-b")["status"] == "failed"
    assert not interrupted_checkout.exists()
    assert retained_artifact.is_file()

    async def fake_run_cmd(cmd, cwd, logfile, env, timeout, **kwargs):
        assert cmd.startswith("train --work-dir ")
        assert (Path(cwd) / "restart.marker").read_text(encoding="utf-8") == "reuse me\n"
        assert env["TRAINERD_REPO_SHA"] == revision
        return True

    monkeypatch.setattr(runner_module, "_run_cmd", fake_run_cmd)
    asyncio.run(runner.run_job("job-a"))

    assert store.get_job("job-a")["status"] == "completed"
    assert not checkout.exists()
    assert (work_dir / "job-a").is_dir()


def test_restart_preserves_completed_stage_and_fails_interrupted_stage(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    store.create_job(
        "job-a",
        ["prepare", "train"],
        "v1",
        stage_queues={"prepare": "cpu", "train": "gpu"},
    )
    store.set_stage_running("job-a", "prepare")
    store.set_stage_completed("job-a", "prepare", 4.0)
    store.set_running("job-a", "train")
    store.set_stage_running("job-a", "train")

    assert store.recover_interrupted_jobs() == ["job-a"]

    job = store.get_job("job-a")
    assert job["status"] == "failed"
    assert job["stages"]["prepare"]["status"] == "completed"
    assert job["stages"]["train"]["status"] == "failed"
    assert job["stages"]["train"]["error"] == "Interrupted — server restarted"


def test_restart_requeues_persisted_handoff_waiting_for_next_queue(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    store.create_job(
        "job-a",
        ["prepare", "train"],
        "v1",
        stage_queues={"prepare": "cpu", "train": "gpu"},
    )
    store.set_running("job-a", "prepare")
    store.set_stage_running("job-a", "prepare")
    store.set_stage_completed("job-a", "prepare", 4.0)

    assert store.recover_interrupted_jobs() == ["job-a"]

    job = store.get_job("job-a")
    assert job["status"] == "pending"
    assert job["stages"]["prepare"]["status"] == "completed"
    assert job["stages"]["train"]["status"] == "pending"
    assert job["stages"]["train"]["queued_at"]
