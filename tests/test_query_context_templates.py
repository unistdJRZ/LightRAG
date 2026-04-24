from lightrag.prompt import PROMPTS


def test_context_only_templates_omit_reference_document_list():
    kg_context = PROMPTS["kg_query_context_response"].format(
        entities_str="{}",
        relations_str="{}",
        text_chunks_str="{}",
    )
    naive_context = PROMPTS["naive_query_context_response"].format(
        text_chunks_str="{}",
    )

    assert "Reference Document List" not in kg_context
    assert "Reference Document List" not in naive_context
    assert "reference_list_str" not in kg_context
    assert "reference_list_str" not in naive_context


def test_llm_context_templates_keep_reference_document_list():
    assert "Reference Document List" in PROMPTS["kg_query_context"]
    assert "Reference Document List" in PROMPTS["naive_query_context"]
    assert "{reference_list_str}" in PROMPTS["kg_query_context"]
    assert "{reference_list_str}" in PROMPTS["naive_query_context"]
