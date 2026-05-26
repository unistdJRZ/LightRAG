import asyncio
from types import SimpleNamespace

from lightrag.api import opencode_client
from lightrag.api.opencode_client import _build_message


def test_build_message_guides_agent_to_use_latest_refs_and_doc_qa():
    message = _build_message(
        agent_search_id="agent-search-123",
        workspace="default",
        retrieval_target=(
            "[history]\n"
            '[{"role":"assistant","content":"Earlier answer","references":[{"chunk_id":"old-chunk"}]}]\n'
            "[latest_query]\n"
            "What about this source?\n"
            '[{"reference_id":"1","chunk_id":"chunk-1"}]'
        ),
        prior_rag_context=(
            "[doc_qa]\n"
            " - belong_chunk=chunk-1 | question=Preset question | answer=Reference context"
        ),
    )

    assert "agent_submit_id: agent-search-123" in message
    assert "Read [latest_query] first, including its reference list" in message
    assert "Extract all chunk_id values from the latest message references" in message
    assert "inspect [doc_qa] rows whose belong_chunk matches those chunk_id values" in message
    assert "preset questions and reference information" in message
    assert "prior_rag_context:" in message
    assert "retrieval_target:" in message


class _FakeResponse:
    def __init__(self, payload=None, *, error: Exception | None = None):
        self._payload = payload
        self._error = error

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self._error is not None:
            raise self._error


class _FakeOpenCodeClient:
    instances = []

    def __init__(self, *args, prompt_error: Exception | None = None, **kwargs):
        self.calls = []
        self.prompt_error = prompt_error
        _FakeOpenCodeClient.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get(self, path):
        self.calls.append(("GET", path))
        if path == "/global/health":
            return _FakeResponse({"healthy": True, "version": "test"})
        if path == "/session/session-1/message":
            return _FakeResponse(
                [
                    {
                        "info": {"role": "assistant"},
                        "parts": [{"type": "text", "text": "done"}],
                    }
                ]
            )
        return _FakeResponse({})

    async def post(self, path, json=None):
        self.calls.append(("POST", path, json))
        if path == "/session":
            return _FakeResponse({"id": "session-1"})
        if path == "/session/session-1/prompt_async":
            return _FakeResponse({}, error=self.prompt_error)
        return _FakeResponse({})

    async def delete(self, path):
        self.calls.append(("DELETE", path))
        return _FakeResponse(True)


async def _fake_stream_events(*args, **kwargs):
    await asyncio.Event().wait()


def _configure_opencode(monkeypatch):
    monkeypatch.setattr(
        opencode_client,
        "global_args",
        SimpleNamespace(
            opencode_server_url="http://opencode",
            opencode_rag_agent="rag-agent",
            opencode_server_username="opencode",
            opencode_server_password=None,
            opencode_timeout=1.0,
        ),
    )
    monkeypatch.setattr(opencode_client, "_validate_agent_exists", AsyncNoop())
    monkeypatch.setattr(opencode_client, "_stream_events", _fake_stream_events)
    monkeypatch.setattr(
        opencode_client,
        "_wait_until_session_complete",
        AsyncReturn(None),
    )


class AsyncNoop:
    async def __call__(self, *args, **kwargs):
        return None


class AsyncReturn:
    def __init__(self, value):
        self.value = value

    async def __call__(self, *args, **kwargs):
        return self.value


def test_run_agent_search_deletes_opencode_session_after_success(monkeypatch):
    _configure_opencode(monkeypatch)
    _FakeOpenCodeClient.instances = []
    monkeypatch.setattr(opencode_client.httpx, "AsyncClient", _FakeOpenCodeClient)

    result = asyncio.run(
        opencode_client.run_agent_search(
            agent_search_id="agent-search-1",
            workspace="default",
            retrieval_target="target",
        )
    )

    assert result.ok is True
    assert result.session_id == "session-1"
    assert result.final_output == "done"
    assert ("DELETE", "/session/session-1") in _FakeOpenCodeClient.instances[0].calls


def test_run_agent_search_deletes_opencode_session_after_prompt_failure(
    monkeypatch,
):
    _configure_opencode(monkeypatch)
    _FakeOpenCodeClient.instances = []

    class _PromptFailingClient(_FakeOpenCodeClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, prompt_error=RuntimeError("prompt failed"), **kwargs)

    monkeypatch.setattr(opencode_client.httpx, "AsyncClient", _PromptFailingClient)

    result = asyncio.run(
        opencode_client.run_agent_search(
            agent_search_id="agent-search-1",
            workspace="default",
            retrieval_target="target",
        )
    )

    assert result.ok is False
    assert result.session_id == "session-1"
    assert result.error == "prompt failed"
    assert ("DELETE", "/session/session-1") in _FakeOpenCodeClient.instances[0].calls
