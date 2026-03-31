#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path


DEFAULT_V2_FIRST = "src/genie_tts/Data/v2/Models/t2s_first_stage_decoder_fp32.onnx"
DEFAULT_V2_STAGE = "src/genie_tts/Data/v2/Models/t2s_stage_decoder_fp32.onnx"
DEFAULT_V2PP_FIRST = DEFAULT_V2_FIRST
DEFAULT_V2PP_STAGE = DEFAULT_V2_STAGE


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install custom decoder ONNX templates for Genie exports (e.g. fast_gsv-matched templates)."
    )
    parser.add_argument("--first-stage", required=True, help="Path to custom t2s_first_stage_decoder_fp32.onnx")
    parser.add_argument("--stage", required=True, help="Path to custom t2s_stage_decoder_fp32.onnx")
    parser.add_argument(
        "--repo",
        default=".",
        help="Path to Genie-TTS repo root (default: current directory)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup original decoder templates before replacing them",
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    first_stage = Path(args.first_stage).resolve()
    stage = Path(args.stage).resolve()

    targets = [
        repo / DEFAULT_V2_FIRST,
        repo / DEFAULT_V2_STAGE,
    ]

    if not first_stage.is_file():
        raise FileNotFoundError(first_stage)
    if not stage.is_file():
        raise FileNotFoundError(stage)

    mapping = {
        repo / DEFAULT_V2_FIRST: first_stage,
        repo / DEFAULT_V2_STAGE: stage,
    }

    for target, src in mapping.items():
        if args.backup and target.exists():
            backup = target.with_suffix(target.suffix + ".bak")
            shutil.copy2(target, backup)
            print(f"[backup] {target} -> {backup}")
        shutil.copy2(src, target)
        print(f"[replace] {src} -> {target}")

    print("\n[OK] custom decoder templates installed")
    print("Now re-export your model so the generated _genie directory uses the new decoder behavior.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
