from __future__ import annotations

from pathlib import Path


def test_resolve_roberta_assets_supports_hf_onnx_layout(tmp_path, monkeypatch):
    import genie_tts.ModelManager as mod

    base_dir = tmp_path / "RoBERTa"
    hf_dir = tmp_path / "roberta-wwm-ext-large-onnx"
    hf_dir.mkdir()
    (hf_dir / "model.onnx").write_bytes(b"")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(mod, "ROBERTA_MODEL_DIR", str(base_dir))

    model_path, tokenizer_path = mod.resolve_roberta_assets(str(base_dir))

    assert Path(model_path) == hf_dir / "model.onnx"
    assert Path(tokenizer_path) == hf_dir / "tokenizer.json"


def test_resolve_roberta_assets_supports_arbitrary_onnx_filename(tmp_path, monkeypatch):
    import genie_tts.ModelManager as mod

    base_dir = tmp_path / "RoBERTa"
    hf_dir = tmp_path / "roberta-wwm-ext-large-onnx"
    hf_dir.mkdir()
    (hf_dir / "roberta-wwm-ext-large-onnx.onnx").write_bytes(b"")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(mod, "ROBERTA_MODEL_DIR", str(base_dir))

    model_path, tokenizer_path = mod.resolve_roberta_assets(str(base_dir))

    assert Path(model_path) == hf_dir / "roberta-wwm-ext-large-onnx.onnx"
    assert Path(tokenizer_path) == hf_dir / "tokenizer.json"
