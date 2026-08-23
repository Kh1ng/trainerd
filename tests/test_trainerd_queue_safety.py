import asyncio
import io
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient


def test_queue_wakes_for_submissions_and_released_slots(tmp_path, monkeypatch):
    import trainerd.server as server

    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": "test",
                "repo": {"local_path": str(tmp_path)},
                "work_dir": str(tmp_path / "work"),
                "log_dir": str(tmp_path / "logs"),
                "steps": [
                    {
                        "id": "run",
                        "cmd": subprocess.list2cmdline(
                            [sys.executable, "-c", "import time; time.sleep(0.5)"]
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAINING_CONFIG", str(config_path))
    monkeypatch.delenv("TRAINERD_PROJECTS_CONFIG", raising=False)
    monkeypatch.delenv("TRAINERD_LAN_MODE", raising=False)
    with TestClient(server.app) as client:
        server._runtime.queue_poll_interval = 30
        time.sleep(0.1)
        first = client.post("/api/jobs", json={"steps": ["run"], "version": "v1"})
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if client.get(f"/api/jobs/{first.json()['job_id']}").json()["status"] == "running":
                break
            time.sleep(0.05)
        assert client.get(f"/api/jobs/{first.json()['job_id']}").json()["status"] == "running"

        second = client.post("/api/jobs", json={"steps": ["run"], "version": "v2"})
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if client.get(f"/api/jobs/{second.json()['job_id']}").json()["status"] == "completed":
                break
            time.sleep(0.05)
        assert client.get(f"/api/jobs/{second.json()['job_id']}").json()["status"] == "completed"

def test_refresh_config_keeps_running_handle_cancellable(tmp_path):
    import trainerd.runner as runner_module
    import trainerd.server as server
    from trainerd.config import ConfiguredProject, ServerConfig, load_config
    from trainerd.runner import JobRunner, StageQueuePool
    from trainerd.runtime import ProjectRuntime
    from trainerd.storage import JobStore

    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        "project: test\nrepo:\n  local_path: %s\nsteps: []\n" % tmp_path,
        encoding="utf-8",
    )
    store = JobStore(tmp_path / "jobs.db")
    runner = JobRunner(store, load_config(config_path), config_path=config_path)
    marker = SimpleNamespace(returncode=None)
    runner._running_procs["job"] = marker
    captured = {}
    async def fake_terminate(proc):
        captured["proc"] = proc
        return True
    original_terminate = runner_module._terminate_proc_tree
    runner_module._terminate_proc_tree = fake_terminate
    try:
        project = ProjectRuntime("test", config_path, runner._config, store, runner)
        server._runtime.configure_projects(
            ServerConfig(
                {"test": ConfiguredProject("test", config_path, runner._config)},
                "test",
                "",
                7860,
                1,
                False,
            ),
            StageQueuePool(),
            projects={"test": project},
        )
        server._runtime.refresh(project)
        assert asyncio.run(runner.cancel_job("job")) is True
        assert captured["proc"] is marker
        assert runner._running_procs["job"] is marker
    finally:
        runner_module._terminate_proc_tree = original_terminate
        server._runtime.reset()


def test_run_cmd_requests_new_session_on_posix(monkeypatch):
    import trainerd.runner as runner

    if runner.os.name == "nt":
        return

    captured = {}

    class Output:
        async def read(self, size):
            return b""

    class Process:
        pid = 1234
        returncode = 0
        stdout = Output()

        async def wait(self):
            return 0

    async def fake_create(*args, **kwargs):
        captured.update(kwargs)
        return Process()

    monkeypatch.setattr(runner.asyncio, "create_subprocess_shell", fake_create)
    ok = asyncio.run(runner._run_cmd("echo ok", None, io.StringIO(), None, 5))
    assert ok is True
    assert captured["start_new_session"] is True


def test_run_cmd_streams_output_larger_than_reader_line_limit(tmp_path):
    import trainerd.runner as runner

    script = tmp_path / "large_output.py"
    script.write_text("import sys\nsys.stdout.write('x' * 65537)\n", encoding="utf-8")
    output = io.StringIO()

    ok = asyncio.run(
        runner._run_cmd(
            subprocess.list2cmdline([sys.executable, str(script)]),
            None,
            output,
            None,
            5,
        )
    )

    assert ok is True
    assert output.getvalue() == "x" * 65537


def test_cancel_uses_process_group_termination(monkeypatch):
    import trainerd.runner as runner

    class Process:
        pid = 5678
        returncode = None

        async def wait(self):
            self.returncode = 0
            return 0

    if runner.os.name == "nt":
        calls = []
        def run(args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0)
        monkeypatch.setattr(
            runner.subprocess,
            "run",
            run,
        )
        assert asyncio.run(runner._terminate_proc_tree(Process())) is True
        assert calls[0][0] == ["taskkill", "/F", "/T", "/PID", "5678"]
        assert calls[0][1]["check"] is False
        return

    killed = []
    monkeypatch.setattr(
        runner.os,
        "killpg",
        lambda pid, sig: killed.append((pid, sig)),
    )
    assert asyncio.run(runner._terminate_proc_tree(Process())) is True
    assert killed == [(5678, runner.signal.SIGTERM)]


def test_windows_taskkill_failure_is_reported(monkeypatch):
    import trainerd.runner as runner

    class Process:
        pid = 5678
        returncode = None

        async def wait(self):
            self.returncode = 0
            return 0

    monkeypatch.setattr(runner, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1),
    )

    assert asyncio.run(runner._terminate_proc_tree(Process())) is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process-tree regression")
def test_cancel_job_kills_windows_parent_and_child(tmp_path):
    import ctypes

    from trainerd.config import RepoConfig, StepConfig, TrainingConfig
    from trainerd.runner import JobRunner
    from trainerd.storage import JobStore

    parent_pid_path = tmp_path / "parent.pid"
    child_pid_path = tmp_path / "child.pid"
    script = tmp_path / "process_tree.py"
    script.write_text(
        "import os, pathlib, subprocess, sys, time\n"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        "pathlib.Path(sys.argv[2]).write_text(str(child.pid))\n"
        "time.sleep(300)\n",
        encoding="utf-8",
    )
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    config = TrainingConfig(
        project="test",
        repo=RepoConfig("", "main", str(tmp_path)),
        work_dir=tmp_path / "work",
        steps=[
            StepConfig(
                "run",
                "Run",
                subprocess.list2cmdline(
                    [sys.executable, str(script), str(parent_pid_path), str(child_pid_path)]
                ),
            )
        ],
        validation=None,
        promotion=None,
        api_key="",
        server_port=7860,
        log_dir=log_dir,
    )
    store = JobStore(tmp_path / "jobs.db")
    store.create_job("job-123", steps=["run"], version="v1")
    runner = JobRunner(store, config)

    async def exercise() -> bool:
        job = asyncio.create_task(runner.run_job("job-123"))
        for _ in range(100):
            if child_pid_path.exists():
                break
            await asyncio.sleep(0.05)
        assert child_pid_path.exists()
        killed = await runner.cancel_job("job-123")
        await asyncio.wait_for(job, timeout=5)
        return killed

    def exists(pid: int) -> bool:
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True

    try:
        assert asyncio.run(exercise()) is True
        pids = [int(parent_pid_path.read_text()), int(child_pid_path.read_text())]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(exists(pid) for pid in pids):
            time.sleep(0.05)
        assert not any(exists(pid) for pid in pids)
    finally:
        for pid_path in (parent_pid_path, child_pid_path):
            if pid_path.exists():
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", pid_path.read_text()],
                    capture_output=True,
                    check=False,
                )
