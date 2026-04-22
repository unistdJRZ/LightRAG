import asyncio
import json
import sys
import types

ascii_colors_stub = types.ModuleType("ascii_colors")


class _ASCIIColors:
    @staticmethod
    def yellow(message: str) -> None:
        return None

    @staticmethod
    def red(message: str) -> None:
        return None


ascii_colors_stub.ASCIIColors = _ASCIIColors
sys.modules.setdefault("ascii_colors", ascii_colors_stub)

from lightrag.api.config import global_args
from lightrag.api.opencode_client import run_agent_search

global_args.opencode_server_url = "http://localhost:4096"
global_args.opencode_server_username = "opencode"
global_args.opencode_server_password = "opencodepasswd"
global_args.opencode_rag_agent = "RAG search"
global_args.opencode_timeout = 90.0


async def main() -> None:
    events: list[dict[str, str | None]] = []

    async def on_status(status) -> None:
        events.append(
            {
                "phase": status.phase,
                "message": status.message,
                "event_type": status.event_type,
                "session_id": status.session_id,
            }
        )

    result = await run_agent_search(
        agent_search_id="agent-search-connectivity-test",
        workspace="msre",
        retrieval_target="[history]\nuser: find LightRAG related chunks and entities\n[latest_query]\nWhat is LightRAG?",
        callback=on_status,
    )
    print(
        json.dumps(
            {
                "ok": result.ok,
                "session_id": result.session_id,
                "error": result.error,
                "final_output": result.final_output,
                "event_count": len(events),
                "events": events[:10],
            },
            ensure_ascii=False,
        )
    )

if __name__ == "__main__":
    asyncio.run(main())
