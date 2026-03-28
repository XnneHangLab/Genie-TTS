#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path


def pick_single(paths: list[Path], kind: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"No {kind} file found")
    if len(paths) > 1:
        joined = "\n  - ".join(str(p) for p in paths)
        raise RuntimeError(f"Multiple {kind} files found, please disambiguate:\n  - {joined}")
    return paths[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-discover GPT-SoVITS checkpoint files in a directory and invoke Genie conversion"
    )
    parser.add_argument("model_dir", help="Directory containing one .ckpt and one .pth file")
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output directory. Default: <model_dir>_genie",
    )
    parser.add_argument(
        "--force-version",
        choices=["v2", "v2pp"],
        default=None,
        help="Force model version instead of auto-detecting",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use when calling scripts/convert_model.py",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        print(f"[ERROR] model dir not found: {model_dir}", file=sys.stderr)
        return 1

    ckpts = sorted(model_dir.glob("*.ckpt"))
    pths = sorted(model_dir.glob("*.pth"))

    try:
        ckpt = pick_single(ckpts, ".ckpt")
        pth = pick_single(pths, ".pth")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.out).resolve() if args.out else model_dir.with_name(f"{model_dir.name}_genie")

    convert_script = Path(__file__).with_name("convert_model.py")
    cmd = [
        args.python,
        str(convert_script),
        "--ckpt",
        str(ckpt),
        "--pth",
        str(pth),
        "--out",
        str(out_dir),
    ]
    if args.force_version:
        cmd.extend(["--force-version", args.force_version])

    print("=== Genie Auto Convert ===")
    print(f"model_dir: {model_dir}")
    print(f"ckpt     : {ckpt}")
    print(f"pth      : {pth}")
    print(f"out      : {out_dir}")
    print(f"cmd      : {' '.join(cmd)}")

    import subprocess
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
