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
