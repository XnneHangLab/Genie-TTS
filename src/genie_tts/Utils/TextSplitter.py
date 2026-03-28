import re
from typing import List, Set, Pattern


class TextSplitter:
    def __init__(self, max_len: int = 100, min_len: int = 5):
        """
        初始化文本切分器。

        更贴近 fast_gsv：默认更长的软切分长度，避免中文句子被过早切碎。

        :param max_len: 软限制最大长度 (Effective Length)。超过此长度遇到分隔符时会切分。
        :param min_len: 硬限制最小长度 (Effective Length)。小于此长度遇到终止符也不会切分。
        """
        self.max_len: int = max_len
        self.min_len: int = min_len

        self.end_chars: Set[str] = {
            '。', '！', '？', '…',
            '!', '?', '.'
        }

        self.all_puncts_chars: Set[str] = self.end_chars | {
            '，', '、', '；', '：', '——',
            ',', ';', ':',
            '“', '”', '‘', '’', '"', "'",
        }

        sorted_puncts: List[str] = sorted(list(self.all_puncts_chars), key=len, reverse=True)
        escaped_puncts: List[str] = [re.escape(p) for p in sorted_puncts]
        self.pattern: Pattern = re.compile(f"((?:{'|'.join(escaped_puncts)})+)")

    @staticmethod
    def get_char_width(char: str) -> int:
        cp = ord(char)
        if cp < 128:
            return 1
        if 0x3130 <= cp <= 0x318F:
            return 1
        return 2

    def get_effective_len(self, text: str) -> int:
        length = 0
        for char in text:
            if char in self.all_puncts_chars:
                continue
            length += self.get_char_width(char)
        return length

    def is_terminator_block(self, block: str) -> bool:
        for char in block:
            if char in self.end_chars:
                return True
        return False

    def split(self, text: str) -> List[str]:
        if not text:
            return []

        text = text.replace('\n', '')
        segments: List[str] = self.pattern.split(text)

        sentences: List[str] = []
        current_buffer: str = ""

        for segment in segments:
            if not segment:
                continue

            is_punct_block = segment[0] in self.all_puncts_chars

            if is_punct_block:
                current_buffer += segment
                eff_len = self.get_effective_len(current_buffer)

                if self.is_terminator_block(segment):
                    if eff_len >= self.min_len:
                        sentences.append(current_buffer.strip())
                        current_buffer = ""
                else:
                    if eff_len >= self.max_len:
                        sentences.append(current_buffer.strip())
                        current_buffer = ""
            else:
                current_buffer += segment

        if current_buffer:
            self._flush_buffer(sentences, current_buffer)

        return sentences

    def _flush_buffer(self, sentences: List[str], buffer: str):
        candidate = buffer.strip()
        if not candidate:
            return
        eff_len = self.get_effective_len(candidate)
        if eff_len > 0:
            sentences.append(candidate)
        elif sentences:
            sentences[-1] += candidate
