import asyncio
import io
from types import SimpleNamespace


def test_refresh_config_keeps_running_handle_cancellable(tmp_path):
    import trainerd.server as server
    import trainerd.runner as runner_module
    from trainerd.config import load_config
    from trainerd.runner import JobRunner
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
    original = (server._store, server._runner, server._config, server._config_path)
    try:
        server._store, server._runner, server._config, server._config_path = store, runner, runner._config, config_path
        server._refresh_runtime_config()
        assert asyncio.run(runner.cancel_job("job")) is True
        assert captured["proc"] is marker
        assert runner._running_procs["job"] is marker
    finally:
        runner_module._terminate_proc_tree = original_terminate
        server._store, server._runner, server._config, server._config_path = original


def test_run_cmd_requests_new_session_on_posix(monkeypatch):
    import trainerd.runner as runner

    if runner.os.name == "nt":
        return

    captured = {}

    class Output:
        def __aiter__(self):
            async def rows():
                yield b"ok\n"
            return rows().__aiter__()

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
        monkeypatch.setattr(
            runner.subprocess,
            "run",
            lambda args, **kwargs: calls.append((args, kwargs)),
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
