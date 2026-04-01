import onnxruntime as ort
import numpy as np
from typing import List, Optional
import threading

from ..Audio.ReferenceAudio import ReferenceAudio
from ..GetPhonesAndBert import get_phones_and_bert

MAX_T2S_LEN = 1000


def _should_stop_decoding(stop_condition_tensor: object) -> bool:
    stop_array = np.asarray(stop_condition_tensor)
    if stop_array.size == 0:
        return False
    return bool(stop_array.reshape(-1)[0])


def _extract_generated_semantic_tokens(y: np.ndarray, generated_steps: int) -> np.ndarray:
    if generated_steps <= 0:
        raise RuntimeError("GENIE decoder returned zero generated semantic steps")
    if y.ndim != 2:
        raise ValueError(f"Expected semantic tokens with 2 dims, got shape={y.shape}")

    slice_width = min(generated_steps, y.shape[1])
    return np.expand_dims(y[:, -slice_width:], axis=0)


class GENIE:
    def __init__(self):
        self.stop_event: threading.Event = threading.Event()

    def tts(
            self,
            text: str,
            prompt_audio: ReferenceAudio,
            encoder: ort.InferenceSession,
            first_stage_decoder: ort.InferenceSession,
            stage_decoder: ort.InferenceSession,
            vocoder: ort.InferenceSession,
            prompt_encoder: Optional[ort.InferenceSession],
            language: str = 'japanese',
    ) -> Optional[np.ndarray]:
        text = '。' + text  # 防止漏第一句。
        text_seq, text_bert = get_phones_and_bert(
            text,
            language=language,
            use_roberta=getattr(prompt_audio, "use_roberta", False),
        )

        semantic_tokens: np.ndarray = self.t2s_cpu(
            ref_seq=prompt_audio.phonemes_seq,
            ref_bert=prompt_audio.text_bert,
            text_seq=text_seq,
            text_bert=text_bert,
            ssl_content=prompt_audio.ssl_content,
            encoder=encoder,
            first_stage_decoder=first_stage_decoder,
            stage_decoder=stage_decoder,
        )
        if semantic_tokens is None:
            return None

        eos_indices = np.where(semantic_tokens >= 1024)  # 剔除不合法的元素，例如 EOS Token。
        if len(eos_indices[0]) > 0:
            first_eos_index = eos_indices[-1][0]
            semantic_tokens = semantic_tokens[..., :first_eos_index]

        if prompt_encoder is None:
            return vocoder.run(None, {
                "text_seq": text_seq,
                "pred_semantic": semantic_tokens,
                "ref_audio": prompt_audio.audio_32k
            })[0]
        else:
            # V2ProPlus 新增。
            prompt_audio.update_global_emb(prompt_encoder=prompt_encoder)
            audio_chunk = vocoder.run(None, {
                "text_seq": text_seq,
                "pred_semantic": semantic_tokens,
                "ge": prompt_audio.global_emb,
                "ge_advanced": prompt_audio.global_emb_advanced,
            })[0]
            return audio_chunk

    def t2s_cpu(
            self,
            ref_seq: np.ndarray,
            ref_bert: np.ndarray,
            text_seq: np.ndarray,
            text_bert: np.ndarray,
            ssl_content: np.ndarray,
            encoder: ort.InferenceSession,
            first_stage_decoder: ort.InferenceSession,
            stage_decoder: ort.InferenceSession,
    ) -> Optional[np.ndarray]:
        """在CPU上运行T2S模型"""
        # Encoder
        x, prompts = encoder.run(
            None,
            {
                "ref_seq": ref_seq,
                "text_seq": text_seq,
                "ref_bert": ref_bert,
                "text_bert": text_bert,
                "ssl_content": ssl_content,
            },
        )

        # First Stage Decoder
        y, y_emb, *present_key_values = first_stage_decoder.run(
            None, {"x": x, "prompts": prompts}
        )

        # Stage Decoder
        input_names: List[str] = [inp.name for inp in stage_decoder.get_inputs()]
        generated_steps = 0
        for _ in range(0, 500):
            if self.stop_event.is_set():
                return None
            input_feed = {
                name: data
                for name, data in zip(input_names, [y, y_emb, *present_key_values])
            }
            outputs = stage_decoder.run(None, input_feed)
            y, y_emb, stop_condition_tensor, *present_key_values = outputs
            generated_steps += 1

            if _should_stop_decoding(stop_condition_tensor):
                break

        y[0, -1] = 0
        return _extract_generated_semantic_tokens(y, generated_steps)


tts_client: GENIE = GENIE()
