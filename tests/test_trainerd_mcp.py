import asyncio
import io

from mcp import Client


def test_mcp_tools_use_http_boundary_without_exposing_credentials(monkeypatch) -> None:
    import trainerd.mcp as trainerd_mcp

    calls = []

    def request(method, url, api_key, payload=None):
        calls.append((method, url, api_key, payload))
        if url.endswith("/api/queue"):
            return {
                "jobs": [{"job_id": "active-1", "status": "running", "queue_position": None}],
                "pending_jobs": 0,
                "running_jobs": 1,
                "queue_capacity": 0,
            }
        if "/api/jobs?" in url:
            return [
                {"job_id": "active-1", "status": "running", "extra_args": "hidden"},
                {"job_id": "recent-1", "status": "completed", "extra_args": "hidden"},
            ]
        if url.endswith("/api/jobs/recent-1/artifacts"):
            return {
                "job_id": "recent-1",
                "run_label": "v1",
                "produced_at": "now",
                "metadata": {"promotion_eligible": True},
                "artifacts": [
                    {
                        "path": "result.json",
                        "sha256": "a" * 64,
                        "bytes": 3,
                        "download_url": "/secret-adjacent-path",
                    }
                ],
            }
        if url.endswith("/api/jobs/recent-1"):
            return {
                "job_id": "recent-1",
                "project": "alpha",
                "status": "completed",
                "version": "v1",
                "steps": ["train"],
                "extra_args": "hidden",
            }
        if method == "POST" and url.endswith("/api/jobs"):
            return {"job_id": "new-1", "status": "pending", "queued": True}
        if method == "DELETE":
            return {"job_id": "new-1", "status": "failed"}
        if url.endswith("/promote"):
            return {"job_id": "recent-1", "status": "promoting"}
        raise AssertionError((method, url, payload))

    monkeypatch.setattr(trainerd_mcp, "_request_json", request)
    monkeypatch.setattr(
        trainerd_mcp,
        "_request_bounded_text",
        lambda url, key, limit: ("line one\nline two\n", False),
    )

    async def exercise() -> None:
        async with Client(trainerd_mcp.create_server("http://trainerd", "secret")) as client:
            listed = await client.list_tools()
            tools = {tool.name: tool for tool in listed.tools}
            assert set(tools) == {
                "list_jobs",
                "get_job",
                "tail_job_logs",
                "list_job_artifacts",
                "submit_job",
                "cancel_job",
                "promote_job",
            }
            for tool in tools.values():
                assert not ({"api_key", "command", "env", "extra_args"} & set(tool.input_schema["properties"]))

            jobs = (await client.call_tool("list_jobs", {"limit": 10})).structured_content
            assert jobs["active"][0]["job_id"] == "active-1"
            assert jobs["recent"] == [
                {"job_id": "recent-1", "status": "completed", "terminal": True}
            ]

            job = (await client.call_tool("get_job", {"job_id": "recent-1"})).structured_content
            assert job["terminal"] is True
            assert job["queue_position"] is None
            assert "extra_args" not in job

            logs = (
                await client.call_tool("tail_job_logs", {"job_id": "recent-1", "lines": 2})
            ).structured_content
            assert logs == {
                "job_id": "recent-1",
                "lines": 2,
                "text": "line one\nline two\n",
                "truncated": False,
            }

            artifacts = (
                await client.call_tool("list_job_artifacts", {"job_id": "recent-1"})
            ).structured_content
            assert artifacts["artifacts"] == [
                {"path": "result.json", "sha256": "a" * 64, "bytes": 3}
            ]
            assert "download_url" not in artifacts["artifacts"][0]

            submitted = (
                await client.call_tool(
                    "submit_job",
                    {"repo": "http://git.local/team/repo.git", "task": "train"},
                )
            ).structured_content
            assert submitted["terminal"] is False
            assert calls[-1][3]["triggered_by"] == "mcp"

            cancelled = (
                await client.call_tool("cancel_job", {"job_id": "new-1"})
            ).structured_content
            assert cancelled["terminal"] is True

            promoted = (
                await client.call_tool("promote_job", {"job_id": "recent-1"})
            ).structured_content
            assert promoted["terminal"] is False

        assert all(call[2] == "secret" for call in calls)

    asyncio.run(exercise())


def test_mcp_log_tail_is_bounded() -> None:
    import trainerd.mcp as trainerd_mcp

    async def exercise() -> None:
        async with Client(trainerd_mcp.create_server("http://trainerd", "")) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert tools["tail_job_logs"].input_schema["properties"]["lines"]["maximum"] == 500
            result = await client.call_tool(
                "tail_job_logs", {"job_id": "job-1", "lines": 501}
            )
            assert result.is_error is True

    asyncio.run(exercise())


def test_mcp_log_bytes_and_credentials_are_bounded(monkeypatch) -> None:
    import trainerd.cli as trainerd_cli
    import trainerd.mcp as trainerd_mcp

    monkeypatch.setattr(
        trainerd_mcp.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(b"x" * 11),
    )
    text, truncated = trainerd_mcp._request_bounded_text(
        "http://trainerd/api/jobs/one/logs?tail=1", "secret", 10
    )
    assert text == "x" * 10
    assert truncated is True

    captured = {}
    monkeypatch.setenv("TRAINERD_SERVER_URL", "http://trainerd")
    monkeypatch.setenv("TRAINERD_API_KEY", "secret")
    monkeypatch.setattr(
        trainerd_mcp,
        "run",
        lambda server_url, api_key: captured.update(
            server_url=server_url, api_key=api_key
        ),
    )
    args = trainerd_cli.build_parser().parse_args(["mcp"])
    assert "api_key" not in vars(args)
    assert trainerd_cli._cmd_mcp(args) == 0
    assert captured == {"server_url": "http://trainerd", "api_key": "secret"}
