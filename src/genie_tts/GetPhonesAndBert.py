"""Phone sequence and BERT feature extraction with multi-language dispatch.

Supported language values (after normalize_language()):
  'Japanese', 'English', 'Chinese', 'Korean',
  'Hybrid-Chinese-English', 'auto'
"""
from __future__ import annotations

import re
import logging
import numpy as np
from typing import Tuple

from .Utils.Constants import BERT_FEATURE_DIM
from .ModelManager import model_manager

logger = logging.getLogger(__name__)


def _build_roberta_inputs(encoded, word2ph: list[int]) -> dict[str, np.ndarray]:
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
    input_names = {inp.name for inp in model_manager.roberta_model.get_inputs()}

    ort_inputs: dict[str, np.ndarray] = {}
    if "input_ids" in input_names:
        ort_inputs["input_ids"] = input_ids
    if "attention_mask" in input_names:
        ort_inputs["attention_mask"] = attention_mask
    if "token_type_ids" in input_names:
        ort_inputs["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)
    if "repeats" in input_names:
        ort_inputs["repeats"] = np.array(word2ph, dtype=np.int64)
    return ort_inputs


def _expand_roberta_output(text_bert: np.ndarray, word2ph: list[int], num_phones: int) -> np.ndarray:
    if text_bert.ndim == 3 and text_bert.shape[0] == 1:
        text_bert = text_bert[0]

    if text_bert.shape[0] == num_phones:
        return text_bert.astype(np.float32)

    if text_bert.shape[0] == len(word2ph) + 2:
        text_bert = text_bert[1:-1]

    if text_bert.shape[0] == len(word2ph):
        repeated = [
            np.repeat(text_bert[idx: idx + 1], repeats=count, axis=0)
            for idx, count in enumerate(word2ph)
        ]
        expanded = np.concatenate(repeated, axis=0)
        if expanded.shape[0] != num_phones:
            raise ValueError(
                f"Expanded RoBERTa features to {expanded.shape[0]} phones, expected {num_phones}"
            )
        return expanded.astype(np.float32)

    raise ValueError(
        "Unsupported RoBERTa output layout: "
        f"got first dimension {text_bert.shape[0]}, expected {num_phones} phones "
        f"or {len(word2ph)} / {len(word2ph) + 2} token features"
    )


# ---------------------------------------------------------------------------
# Legacy Hybrid-Chinese-English splitter (preserved for backward compatibility)
# ---------------------------------------------------------------------------

def _split_chinese_english(text: str) -> list[dict]:
    """Split text into Chinese and English chunks via Latin-character regex.

    Kept for backward compatibility with 'Hybrid-Chinese-English' mode.
    For general mixed-language splitting, use Utils.LangDetector.segment_by_language.
    """
    pattern_eng = re.compile(r"[a-zA-Z]+")
    parts = re.split(pattern_eng, text)
    matches = pattern_eng.findall(text)

    result: list[dict] = []
    for i, part in enumerate(parts):
        if part.strip():
            result.append({"language": "chinese", "content": part})
        if i < len(matches):
            result.append({"language": "english", "content": matches[i]})
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_phones_and_bert(
    prompt_text: str, language: str = "japanese", use_roberta: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (phones_seq, text_bert) for *prompt_text* in *language*.

    *language* should already be normalised by normalize_language().
    Handles multi-language modes ('Hybrid-Chinese-English', 'auto') by
    splitting text into per-language chunks and concatenating results.
    """
    lang_lower = language.lower()

    if lang_lower == "hybrid-chinese-english":
        chunks = _split_chinese_english(prompt_text)
        return _process_chunks(chunks, use_roberta=use_roberta)

    if lang_lower == "auto":
        from .Utils.LangDetector import segment_by_language
        chunks = segment_by_language(prompt_text)
        if not chunks:
            logger.warning("LangDetector returned no segments for text %r; falling back to Japanese.", prompt_text[:60])
            return _get_phones_and_bert_single(prompt_text, "japanese", use_roberta=use_roberta)
        if len(chunks) == 1:
            return _get_phones_and_bert_single(chunks[0]["content"], chunks[0]["language"], use_roberta=use_roberta)
        return _process_chunks(chunks, use_roberta=use_roberta)

    return _get_phones_and_bert_single(prompt_text, language, use_roberta=use_roberta)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_chunks(chunks: list[dict], use_roberta: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Run G2P on each chunk and concatenate results."""
    list_phones: list[np.ndarray] = []
    list_berts: list[np.ndarray] = []
    for chunk in chunks:
        phones_seq, text_bert = _get_phones_and_bert_single(
            chunk["content"], chunk["language"], use_roberta=use_roberta
        )
        list_phones.append(phones_seq)
        list_berts.append(text_bert)
    phones_seq = np.concatenate(list_phones, axis=1)
    text_bert = np.concatenate(list_berts, axis=0)
    return phones_seq, text_bert


def _get_phones_and_bert_single(
    prompt_text: str, language: str = "japanese", use_roberta: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Run G2P for a single-language text chunk."""
    lang_lower = language.lower()

    if lang_lower == "english":
        from .G2P.English.EnglishG2P import english_to_phones
        phones = english_to_phones(prompt_text)
        text_bert = np.zeros((len(phones), BERT_FEATURE_DIM), dtype=np.float32)

    elif lang_lower == "chinese":
        from .G2P.Chinese.ChineseG2P import chinese_to_phones
        text_clean, _, phones, word2ph = chinese_to_phones(prompt_text)
        if use_roberta:
            if not model_manager.load_roberta_model():
                raise FileNotFoundError(
                    "Chinese RoBERTa was requested but the ONNX model or tokenizer could not be loaded. "
                    "Place model.onnx (or RoBERTa.onnx) and tokenizer.json under ROBERTA_MODEL_DIR "
                    "or a GenieData subdirectory whose name contains 'roberta'. "
                    "You can also call genie.download_roberta_data() or "
                    "genie.download_genie_data(include_roberta=True)."
                )
            encoded = model_manager.roberta_tokenizer.encode(text_clean)
            ort_inputs = _build_roberta_inputs(encoded, word2ph)
            outputs = model_manager.roberta_model.run(None, ort_inputs)
            text_bert = _expand_roberta_output(outputs[0], word2ph, len(phones))
        else:
            text_bert = np.zeros((len(phones), BERT_FEATURE_DIM), dtype=np.float32)

    elif lang_lower == "korean":
        from .G2P.Korean.KoreanG2P import korean_to_phones
        phones = korean_to_phones(prompt_text)
        text_bert = np.zeros((len(phones), BERT_FEATURE_DIM), dtype=np.float32)

    else:
        if lang_lower not in ("japanese",):
            logger.warning(
                "Unsupported language %r in _get_phones_and_bert_single; "
                "falling back to Japanese G2P.",
                language,
            )
        from .G2P.Japanese.JapaneseG2P import japanese_to_phones
        phones = japanese_to_phones(prompt_text)
        text_bert = np.zeros((len(phones), BERT_FEATURE_DIM), dtype=np.float32)

    phones_seq = np.array([phones], dtype=np.int64)
    return phones_seq, text_bert
