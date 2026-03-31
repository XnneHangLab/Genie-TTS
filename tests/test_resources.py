from __future__ import annotations


def test_download_roberta_data_uses_expected_repo_and_patterns(monkeypatch, tmp_path):
    import genie_tts.Core.Resources as mod

    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(mod, "GENIE_DATA_DIR", str(tmp_path / "GenieData"))
    monkeypatch.setattr(mod, "snapshot_download", fake_snapshot_download)

    target_dir = mod.download_roberta_data(model_variant="fp16")

    assert target_dir.endswith("roberta-wwm-ext-large-onnx")
    assert len(calls) == 1
    assert calls[0]["repo_id"] == mod.ROBERTA_REPO_ID
    assert calls[0]["allow_patterns"] == ["model_fp16.onnx", "tokenizer.json"]
    assert calls[0]["local_dir"] == target_dir


def test_download_genie_data_can_include_roberta(monkeypatch):
    import genie_tts.Core.Resources as mod

    calls = []
    roberta_calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)

    def fake_download_roberta_data(model_variant="fp32"):
        roberta_calls.append(model_variant)
        return "dummy"

    monkeypatch.setattr(mod, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(mod, "download_roberta_data", fake_download_roberta_data)

    mod.download_genie_data(include_roberta=True, roberta_model_variant="fp16")

    assert len(calls) == 1
    assert calls[0]["repo_id"] == mod.GENIE_DATA_REPO_ID
    assert calls[0]["allow_patterns"] == "GenieData/*"
    assert roberta_calls == ["fp16"]
