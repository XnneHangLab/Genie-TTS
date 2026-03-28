#!/usr/bin/env python3
import argparse
import os
import sys

import genie_tts as genie


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert GPT-SoVITS V2 / V2ProPlus checkpoints to Genie ONNX format"
    )
    parser.add_argument("--ckpt", required=True, help="Path to GPT checkpoint (.ckpt)")
    parser.add_argument("--pth", required=True, help="Path to SoVITS weights (.pth)")
    parser.add_argument("--out", required=True, help="Output directory for Genie ONNX files")
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.ckpt)
    pth_path = os.path.abspath(args.pth)
    out_dir = os.path.abspath(args.out)

    if not os.path.isfile(ckpt_path):
        print(f"[ERROR] ckpt not found: {ckpt_path}", file=sys.stderr)
        return 1
    if not os.path.isfile(pth_path):
        print(f"[ERROR] pth not found: {pth_path}", file=sys.stderr)
        return 1

    os.makedirs(out_dir, exist_ok=True)

    print("=== Genie Converter ===")
    print(f"ckpt: {ckpt_path}")
    print(f"pth : {pth_path}")
    print(f"out : {out_dir}")

    genie.convert_to_onnx(
        torch_ckpt_path=ckpt_path,
        torch_pth_path=pth_path,
        output_dir=out_dir,
    )

    print("\n[OK] conversion finished")
    print(f"Output directory: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
