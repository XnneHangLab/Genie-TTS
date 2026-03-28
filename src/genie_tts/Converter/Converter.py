from .v2.Converter import convert as convert_v2
from .v2ProPlus.Converter import convert as convert_v2pp
from .load_state_dict import load_sovits_model

import logging

logger = logging.getLogger(__name__)


def detect_model_version(torch_pth_path: str) -> str:
    """Best-effort detection for GPT-SoVITS V2 vs V2ProPlus.

    Prefer inspecting the loaded SoVITS weight structure over using file size.
    V2ProPlus models contain prompt-encoder related weights (for example
    keys starting with `ref_enc.` or `sv_emb.` / `ge_proj.`), while V2 models do not.
    """
    model = load_sovits_model(torch_pth_path)
    weight = model.get("weight", model)
    keys = set(weight.keys())

    v2pp_markers = (
        "ref_enc.",
        "prompt_encoder.",
        "sv_emb.",
        "ge_proj.",
        "ge_advanced_proj.",
    )
    if any(any(k.startswith(marker) for marker in v2pp_markers) for k in keys):
        return "v2pp"

    return "v2"


def convert(torch_ckpt_path: str, torch_pth_path: str, output_dir: str) -> None:
    version = detect_model_version(torch_pth_path)
    logger.info(f"Detected model version: {version}")

    if version == "v2pp":
        convert_v2pp(torch_ckpt_path, torch_pth_path, output_dir)
    else:
        convert_v2(torch_ckpt_path, torch_pth_path, output_dir)
