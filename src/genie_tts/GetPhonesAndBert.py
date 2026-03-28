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

_STRONG_SPLITS = set(['!', '?', '…', '.', '。', '！', '？'])
_ALL_PUNCTUATION = set(['!', '?', '…', ',', '.', '-', ' ', '。', '！', '？', '，', '、', '；', '：'])


def _get_first(text: str) -> str:
    pattern = "[" + "".join(re.escape(sep) for sep in _ALL_PUNCTUATION) + "]"
    return re.split(pattern, text)[0].strip()


def _replace_consecutive_punctuation(text: str) -> str:
    punctuations = ''.join(re.escape(p) for p in _ALL_PUNCTUATION)
    pattern = f'([{punctuations}])([{punctuations}])+'
    return re.sub(pattern, r'\1', text)


def _soften_text(prompt_text: str) -> str:
    text = prompt_text
    text = re.sub(r'[!！]{2,}', '！', text)
    text = re.sub(r'[?？]{2,}', '？', text)
    text = re.sub(r'[.…]{2,}', '。', text)
    text = text.replace('——', '，').replace('—', '，')
    return text


def _prepare_text(prompt_text: str, language: str) -> str:
    text = prompt_text.strip("\n ")
    if not text:
        return text

    lang_lower = language.lower()
    text = _soften_text(text)
    text = _replace_consecutive_punctuation(text)

    if text[0] not in _STRONG_SPLITS and len(_get_first(text)) < 2:
        text = ("。" if lang_lower != "english" else ".") + text

    return text


def _empty_bert(phone_count: int) -> np.ndarray:
    return np.zeros((phone_count, BERT_FEATURE_DIM), dtype=np.float32)


def _ensure_bert_shape(text_bert: np.ndarray, phone_count: int) -> np.ndarray:
    arr = np.asarray(text_bert, dtype=np.float32)
    if arr.ndim != 2:
        return _empty_bert(phone_count)

    if arr.shape == (phone_count, BERT_FEATURE_DIM):
        return arr
    if arr.shape == (BERT_FEATURE_DIM, phone_count):
        return arr.T

    if arr.shape[1] == BERT_FEATURE_DIM:
        return arr
    if arr.shape[0] == BERT_FEATURE_DIM:
        return arr.T

    return _empty_bert(phone_count)


def _split_chinese_english(text: str) -> list[dict]:
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


def get_phones_and_bert(
    prompt_text: str, language: str = "japanese"
) -> Tuple[np.ndarray, np.ndarray]:
    prompt_text = _prepare_text(prompt_text, language)
    lang_lower = language.lower()

    if lang_lower == "hybrid-chinese-english":
        chunks = _split_chinese_english(prompt_text)
        return _process_chunks(chunks)

    if lang_lower == "auto":
        from .Utils.LangDetector import segment_by_language
        chunks = segment_by_language(prompt_text)
        if not chunks:
            logger.warning("LangDetector returned no segments for text %r; falling back to Japanese.", prompt_text[:60])
            return _get_phones_and_bert_single(prompt_text, "japanese")
        if len(chunks) == 1:
            return _get_phones_and_bert_single(_prepare_text(chunks[0]["content"], chunks[0]["language"]), chunks[0]["language"])
        return _process_chunks(chunks)

    return _get_phones_and_bert_single(prompt_text, language)


def _process_chunks(chunks: list[dict]) -> Tuple[np.ndarray, np.ndarray]:
    list_phones: list[np.ndarray] = []
    list_berts: list[np.ndarray] = []
    for chunk in chunks:
        content = _prepare_text(chunk["content"], chunk["language"])
        phones_seq, text_bert = _get_phones_and_bert_single(content, chunk["language"])
        list_phones.append(phones_seq)
        list_berts.append(text_bert)
    phones_seq = np.concatenate(list_phones, axis=1)
    text_bert = np.concatenate(list_berts, axis=0)
    return phones_seq, text_bert


def _get_phones_and_bert_single(
    prompt_text: str, language: str = "japanese"
) -> Tuple[np.ndarray, np.ndarray]:
    lang_lower = language.lower()

    if lang_lower == "english":
        from .G2P.English.EnglishG2P import english_to_phones
        phones = english_to_phones(prompt_text)
        text_bert = _empty_bert(len(phones))

    elif lang_lower == "chinese":
        from .G2P.Chinese.ChineseG2P import chinese_to_phones
        text_clean, _, phones, word2ph = chinese_to_phones(prompt_text)
        if model_manager.load_roberta_model():
            encoded = model_manager.roberta_tokenizer.encode(text_clean)
            input_ids = np.array([encoded.ids], dtype=np.int64)
            attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "repeats": np.array(word2ph, dtype=np.int64),
            }
            outputs = model_manager.roberta_model.run(None, ort_inputs)
            text_bert = _ensure_bert_shape(outputs[0], len(phones))
        else:
            text_bert = _empty_bert(len(phones))

    elif lang_lower == "korean":
        from .G2P.Korean.KoreanG2P import korean_to_phones
        phones = korean_to_phones(prompt_text)
        text_bert = _empty_bert(len(phones))

    else:
        if lang_lower not in ("japanese",):
            logger.warning(
                "Unsupported language %r in _get_phones_and_bert_single; "
                "falling back to Japanese G2P.",
                language,
            )
        from .G2P.Japanese.JapaneseG2P import japanese_to_phones
        phones = japanese_to_phones(prompt_text)
        text_bert = _empty_bert(len(phones))

    phones_seq = np.array([phones], dtype=np.int64)
    return phones_seq, text_bert
