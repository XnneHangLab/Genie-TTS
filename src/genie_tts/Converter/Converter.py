from .v2.Converter import convert as convert_v2
from .v2ProPlus.Converter import convert as convert_v2pp
from .load_state_dict import load_sovits_model

import logging

logger = logging.getLogger(__name__)


V2PP_REQUIRED_KEYS = {
    "sv_emb.weight",
}
V2PP_OPTIONAL_PREFIXES = (
    "ref_enc.",
    "prompt_encoder.",
    "ge_proj.",
    "ge_advanced_proj.",
)


def detect_model_version(torch_pth_path: str) -> str:
    """Best-effort detection for GPT-SoVITS V2 vs V2ProPlus.

    Prefer a strict structure check over file-size guessing.
    Only classify as v2ProPlus when hard markers such as `sv_emb.weight`
    are present; otherwise default to v2.
    """
    model = load_sovits_model(torch_pth_path)
    weight = model.get("weight", model)
    keys = set(weight.keys())

    if any(req in keys for req in V2PP_REQUIRED_KEYS):
        return "v2pp"

    if any(any(k.startswith(prefix) for prefix in V2PP_OPTIONAL_PREFIXES) for k in keys):
        logger.warning(
            "Found some V2ProPlus-like prefixes but missing required keys %s; defaulting to v2",
            sorted(V2PP_REQUIRED_KEYS),
        )

    return "v2"


def convert(torch_ckpt_path: str, torch_pth_path: str, output_dir: str, force_version: str | None = None) -> bool:
    version = (force_version or detect_model_version(torch_pth_path)).lower()
    if version not in {"v2", "v2pp"}:
        raise ValueError(f"Unsupported force_version: {force_version}")

    logger.info(f"Detected model version: {version}")

    if version == "v2pp":
        convert_v2pp(torch_ckpt_path, torch_pth_path, output_dir)
    else:
        convert_v2(torch_ckpt_path, torch_pth_path, output_dir)

    return True
