import os
import re
from typing import List, Tuple, Dict
import logging

from pypinyin import lazy_pinyin, Style
import jieba_fast
import jieba_fast.posseg as psg

from ...Core.Resources import Chinese_G2P_DIR
from ..SymbolsV2 import symbols_v2, symbol_to_id_v2
from .ToneSandhi import ToneSandhi
from .Normalization.text_normlization import TextNormalizer

jieba_fast.setLogLevel(logging.ERROR)

PUNCTUATION = ["!", "?", "…", ",", ".", "-"]
PUNCTUATION_REPLACEMENTS = {
    "：": ",", "；": ",", "，": ",", "。": ".", "！": "!",
    "？": "?", "\n": ".", "·": ",", "、": ",", "$": ".",
    "/": ",", "—": "-", "~": "…", "～": "…",
}
SPECIAL_REPLACEMENTS = {"...": "…"}


class ChineseG2P:
    def __init__(self):
        self.tone_modifier: ToneSandhi = ToneSandhi()
        self.text_normalizer: TextNormalizer = TextNormalizer()
        self.pinyin_to_symbol_map: Dict[str, str] = {}

        allowed_chars = "".join(re.escape(p) for p in PUNCTUATION)
        self.pattern_punct_map = re.compile("|".join(re.escape(p) for p in PUNCTUATION_REPLACEMENTS.keys()))
        self.pattern_filter = re.compile(r"[^\u4e00-\u9fa5" + allowed_chars + r"]+")
        self.pattern_split = re.compile(r"(?<=[{0}])\s*".format(allowed_chars))
        self.pattern_consecutive = re.compile(f"([{allowed_chars}])\\1+")
        self.pattern_eng = re.compile(r"[a-zA-Z]+")

        self.v_rep_map = {"uei": "ui", "iou": "iu", "uen": "un"}
        self.pinyin_rep_map = {"ing": "ying", "i": "yi", "in": "yin", "u": "wu"}
        self.single_rep_map = {"v": "yu", "e": "e", "i": "y", "u": "w"}

        self.load_opencpop_dict()

    def load_opencpop_dict(self):
        map_path = os.path.join(Chinese_G2P_DIR, "opencpop-strict.txt")
        with open(map_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    self.pinyin_to_symbol_map[parts[0]] = parts[1]

    def _replace_punctuation(self, text: str) -> str:
        text = text.replace("嗯", "恩").replace("呣", "母")
        for k, v in SPECIAL_REPLACEMENTS.items():
            text = text.replace(k, v)
        text = self.pattern_punct_map.sub(lambda x: PUNCTUATION_REPLACEMENTS[x.group()], text)
        text = self.pattern_filter.sub("", text)
        return text

    def normalize_text(self, text: str) -> str:
        sentences = self.text_normalizer.normalize(text)
        dest_parts = [self._replace_punctuation(s) for s in sentences]
        dest_text = "".join(dest_parts)
        dest_text = self.pattern_consecutive.sub(r"\1", dest_text)
        return dest_text

    @staticmethod
    def _get_initials_finals(word: str) -> Tuple[List[str], List[str]]:
        initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
        finals = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
        return initials, finals

    def _pinyin_to_opencpop_phones(self, c: str, v: str) -> List[str]:
        v_without_tone = v[:-1]
        tone = v[-1]
        if c:
            final = self.v_rep_map.get(v_without_tone, v_without_tone)
            pinyin_key = c + final
        else:
            temp_key = c + v_without_tone
            if temp_key in self.pinyin_rep_map:
                pinyin_key = self.pinyin_rep_map[temp_key]
            else:
                if temp_key and temp_key[0] in self.single_rep_map:
                    pinyin_key = self.single_rep_map[temp_key[0]] + temp_key[1:]
                else:
                    pinyin_key = temp_key
        phone_str = self.pinyin_to_symbol_map[pinyin_key]
        new_c, new_v = phone_str.split(" ")
        new_v = new_v + tone
        return [new_c, new_v]

    def g2p(self, text: str) -> Tuple[List[str], List[int]]:
        sentences = [i for i in self.pattern_split.split(text) if i.strip() != ""]
        all_phones: List[str] = []
        all_word2ph: List[int] = []

        for seg in sentences:
            seg = self.pattern_eng.sub("", seg)
            seg_cut = psg.lcut(seg)
            seg_cut = self.tone_modifier.pre_merge_for_modify(seg_cut)

            initials: List[str] = []
            finals: List[str] = []
            for word, pos in seg_cut:
                if pos == "eng":
                    continue
                sub_initials, sub_finals = self._get_initials_finals(word)
                sub_finals = self.tone_modifier.modified_tone(word, pos, sub_finals)
                initials.extend(sub_initials)
                finals.extend(sub_finals)

            for c, v in zip(initials, finals):
                if c == v:
                    all_phones.append(c)
                    all_word2ph.append(1)
                else:
                    try:
                        phone_pair = self._pinyin_to_opencpop_phones(c, v)
                        all_phones.extend(phone_pair)
                        all_word2ph.append(len(phone_pair))
                    except KeyError:
                        continue

        return all_phones, all_word2ph

    def process(self, text: str) -> Tuple[str, List[str], List[int], List[int]]:
        normalized_text = self.normalize_text(text)
        phones, word2ph = self.g2p(normalized_text)
        phones = [ph for ph in phones if ph in symbols_v2]
        phones_ids = [symbol_to_id_v2[ph] for ph in phones]
        return normalized_text, phones, phones_ids, word2ph


processor: ChineseG2P = ChineseG2P()


def chinese_to_phones(text: str) -> Tuple[str, List[str], List[int], List[int]]:
    return processor.process(text)
