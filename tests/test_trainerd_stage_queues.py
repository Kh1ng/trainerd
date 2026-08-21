from __future__ import annotations

import asyncio
import time
from pathlib import Path

import trainerd.runner as runner_module
from trainerd.config import RepoConfig, StepConfig, TrainingConfig
from trainerd.runner import JobRunner, StageQueuePool
from trainerd.storage import JobStore


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
        start = time.monotonic()
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
        intervals[(job_id, cmd)] = (start, time.monotonic())
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
