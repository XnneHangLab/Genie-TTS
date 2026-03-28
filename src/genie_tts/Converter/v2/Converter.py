from .VITSConverter import VITSConverter
from .T2SConverter import T2SModelConverter
from .EncoderConverter import EncoderConverter
from ...Utils.Constants import PACKAGE_NAME

import logging
from typing import Optional, Tuple
import re
import os
import shutil
import traceback
import importlib.resources
import contextlib

logger = logging.getLogger()

CACHE_DIR = os.path.join(os.getcwd(), "Cache")
ENCODER_RESOURCE_PATH = "Data/v2/Models/t2s_encoder_fp32.onnx"
STAGE_DECODER_RESOURCE_PATH = "Data/v2/Models/t2s_stage_decoder_fp32.onnx"
FIRST_STAGE_DECODER_RESOURCE_PATH = "Data/v2/Models/t2s_first_stage_decoder_fp32.onnx"
VITS_RESOURCE_PATH = "Data/v2/Models/vits_fp32.onnx"
T2S_KEYS_RESOURCE_PATH = "Data/v2/Keys/t2s_onnx_keys.txt"
VITS_KEYS_RESOURCE_PATH = "Data/v2/Keys/vits_onnx_keys.txt"

# Optional overrides for decoder templates, so we can align exported models
# with a custom ONNX sampling behavior (e.g. fast_gsv-matched templates).
GENIE_T2S_STAGE_DECODER_TEMPLATE = "GENIE_T2S_STAGE_DECODER_TEMPLATE"
GENIE_T2S_FIRST_STAGE_DECODER_TEMPLATE = "GENIE_T2S_FIRST_STAGE_DECODER_TEMPLATE"


def _resolve_template_path(default_path: str, env_key: str, enter) -> str:
    override = os.environ.get(env_key, "").strip()
    if override:
        logger.info(f"Using override template from ${env_key}: {override}")
        return override
    return str(enter(default_path))


def find_ckpt_and_pth(directory: str) -> Tuple[Optional[str], Optional[str]]:
    """
    在 directory（不递归子目录）里查找：
    - .ckpt：从所有 .ckpt 文件名中搜索 'e{正整数}' 作为 epoch（找不到则视为 e0），
             选择 epoch 最大的那个文件（若无则为 None）
    - .pth ：从所有 .pth 文件名中搜索 'e{正整数}' 作为 epoch（找不到则视为 e0），
             选择 epoch 最大的那个文件（若无则为 None）
    若出现相同 epoch，选修改时间较新的文件以打破平手。
    """
    best_ckpt_path: Optional[str] = None
    best_ckpt_epoch: int = -1

    best_pth_path: Optional[str] = None
    best_pth_epoch: int = -1

    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)

        if not os.path.isfile(full_path):
            continue

        m = re.search(r"e(\d+)", filename, flags=re.IGNORECASE)
        epoch = int(m.group(1)) if m else 0

        if filename.lower().endswith(".ckpt"):
            if (
                    epoch > best_ckpt_epoch
                    or (
                    epoch == best_ckpt_epoch
                    and best_ckpt_path is not None
                    and os.path.getmtime(full_path) > os.path.getmtime(best_ckpt_path)
            )
            ):
                best_ckpt_epoch = epoch
                best_ckpt_path = full_path

        elif filename.lower().endswith(".pth"):
            if (
                    epoch > best_pth_epoch
                    or (
                    epoch == best_pth_epoch
                    and best_pth_path is not None
                    and os.path.getmtime(full_path) > os.path.getmtime(best_pth_path)
            )
            ):
                best_pth_epoch = epoch
                best_pth_path = full_path

    return best_ckpt_path, best_pth_path


def remove_folder(folder: str) -> None:
    try:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            logger.info(f"🧹 Folder cleaned: {folder}")
    except Exception as e:
        logger.error(f"❌ Failed to clean folder {folder}: {e}")


def convert(torch_ckpt_path: str,
            torch_pth_path: str,
            output_dir: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if len(os.listdir(output_dir)) > 0:
        logger.warning(f"The output directory {output_dir} is not empty!")

    with contextlib.ExitStack() as stack:
        files = importlib.resources.files(PACKAGE_NAME)

        def enter(p):
            return stack.enter_context(importlib.resources.as_file(files.joinpath(p)))

        encoder_onnx_path = str(enter(ENCODER_RESOURCE_PATH))
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
        vits_onnx_path = str(enter(VITS_RESOURCE_PATH))
        t2s_keys_path = str(enter(T2S_KEYS_RESOURCE_PATH))
        vits_keys_path = str(enter(VITS_KEYS_RESOURCE_PATH))

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

        try:
            converter_1.run_full_process()
            converter_2.run_full_process()
            converter_3.run_full_process()
            logger.info(f"🎉 Conversion successful! Saved to: {os.path.abspath(output_dir)}\n"
                        f"- Model Type: V2")
        except Exception:
            logger.error(f"❌ A critical error occurred during the conversion process")
            logger.error(traceback.format_exc())
            remove_folder(output_dir)
        finally:
            remove_folder(CACHE_DIR)
