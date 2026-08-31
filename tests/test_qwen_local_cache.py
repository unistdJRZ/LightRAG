from lightrag.llm import transforms_qwen


def test_resolve_cached_model_path_keeps_existing_local_path(
    tmp_path, monkeypatch
):
    def fail_if_called(**_kwargs):
        raise AssertionError("snapshot_download should not be called for a local path")

    monkeypatch.setattr(transforms_qwen, "snapshot_download", fail_if_called)

    assert transforms_qwen._resolve_cached_model_path(str(tmp_path)) == str(tmp_path)


def test_resolve_cached_model_path_prefers_cached_snapshot(tmp_path, monkeypatch):
    snapshot_path = tmp_path / "snapshot"
    snapshot_path.mkdir()

    def resolve_snapshot(*, repo_id, local_files_only):
        assert repo_id == "Qwen/Qwen3-VL-Reranker-2B"
        assert local_files_only is True
        return str(snapshot_path)

    monkeypatch.setattr(transforms_qwen, "snapshot_download", resolve_snapshot)

    assert transforms_qwen._resolve_cached_model_path(
        "Qwen/Qwen3-VL-Reranker-2B"
    ) == str(snapshot_path)


def test_resolve_cached_model_path_falls_back_to_repo_id(monkeypatch):
    def cache_miss(**_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(transforms_qwen, "snapshot_download", cache_miss)

    model_id = "Qwen/not-cached"
    assert transforms_qwen._resolve_cached_model_path(model_id) == model_id
