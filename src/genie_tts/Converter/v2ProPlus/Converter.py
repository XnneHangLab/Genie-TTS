import logging
import traceback
import os
import contextlib
import importlib.resources

from ...Utils.Constants import PACKAGE_NAME
from ..v2.VITSConverter import VITSConverter
from ..v2.T2SConverter import T2SModelConverter
from ..v2.EncoderConverter import EncoderConverter
from ..v2.Converter import (
    ENCODER_RESOURCE_PATH,
    STAGE_DECODER_RESOURCE_PATH,
    FIRST_STAGE_DECODER_RESOURCE_PATH,
    T2S_KEYS_RESOURCE_PATH,
    CACHE_DIR,
    remove_folder,
    _resolve_template_path,
    GENIE_T2S_STAGE_DECODER_TEMPLATE,
    GENIE_T2S_FIRST_STAGE_DECODER_TEMPLATE,
)
from .PromptEncoderConverter import PromptEncoderConverter

logger = logging.getLogger()

VITS_RESOURCE_PATH = "Data/v2ProPlus/Models/vits_fp32.onnx"
PROMPT_ENCODER_RESOURCE_PATH = "Data/v2ProPlus/Models/prompt_encoder_fp32.onnx"
VITS_KEYS_RESOURCE_PATH = "./Data/v2ProPlus/Keys/vits_weights.txt"
PROMPT_ENCODER_KEYS_RESOURCE_PATH = "./Data/v2ProPlus/Keys/prompt_encoder_weights.txt"


def convert(torch_ckpt_path: str, torch_pth_path: str, output_dir: str) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if len(os.listdir(output_dir)) > 0:
        logger.warning(f"The output directory {output_dir} is not empty!")

    with contextlib.ExitStack() as stack:
        files = importlib.resources.files(PACKAGE_NAME)

        def enter(p: str) -> str:
            return str(stack.enter_context(importlib.resources.as_file(files.joinpath(p))))

        encoder_onnx_path = enter(ENCODER_RESOURCE_PATH)
        stage_decoder_path = _resolve_template_path(
            STAGE_DECODER_RESOURCE_PATH,
            GENIE_T2S_STAGE_DECODER_TEMPLATE,
            enter,
        )
        first_stage_decoder_path = _resolve_template_path(
            FIRST_STAGE_DECODER_RESOURCE_PATH,
            GENIE_T2S_FIRST_STAGE_DECODER_TEMPLATE,
            enter,
        )
        vits_onnx_path = enter(VITS_RESOURCE_PATH)
        t2s_keys_path = enter(T2S_KEYS_RESOURCE_PATH)
        vits_keys_path = enter(VITS_KEYS_RESOURCE_PATH)
        prompt_encoder_path = enter(PROMPT_ENCODER_RESOURCE_PATH)
        prompt_encoder_keys_path = enter(PROMPT_ENCODER_KEYS_RESOURCE_PATH)

        converter_1 = T2SModelConverter(
            torch_ckpt_path=torch_ckpt_path,
            stage_decoder_onnx_path=stage_decoder_path,
            first_stage_decoder_onnx_path=first_stage_decoder_path,
            key_list_file=t2s_keys_path,
            output_dir=output_dir,
            cache_dir=CACHE_DIR,
        )
        converter_2 = VITSConverter(
            torch_pth_path=torch_pth_path,
            vits_onnx_path=vits_onnx_path,
            key_list_file=vits_keys_path,
            output_dir=output_dir,
            cache_dir=CACHE_DIR,
        )
        converter_3 = EncoderConverter(
            ckpt_path=torch_ckpt_path,
            pth_path=torch_pth_path,
            onnx_input_path=encoder_onnx_path,
            output_dir=output_dir,
        )
        converter_4 = PromptEncoderConverter(
            torch_pth_path=torch_pth_path,
            prompt_encoder_onnx_path=prompt_encoder_path,
            key_list_file=prompt_encoder_keys_path,
            output_dir=output_dir,
            cache_dir=CACHE_DIR,
        )

        try:
            converter_1.run_full_process()
            converter_2.run_full_process()
            converter_3.run_full_process()
            converter_4.run_full_process()
            logger.info(f"🎉 Conversion successful! Saved to: {os.path.abspath(output_dir)}\n"
                        f"- Model Type: V2ProPlus")
        except Exception:
            logger.error(f"❌ A critical error occurred during the conversion process")
            logger.error(traceback.format_exc())
            remove_folder(output_dir)
        finally:
            remove_folder(CACHE_DIR)
