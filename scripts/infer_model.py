#!/usr/bin/env python3
import argparse
import os
import sys

import genie_tts as genie


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Genie TTS inference from a converted V2 / V2ProPlus ONNX model"
    )
    parser.add_argument("--model-dir", required=True, help="Directory containing converted Genie ONNX files")
    parser.add_argument("--character", default="demo", help="Character name to register")
    parser.add_argument(
        "--language",
        required=True,
        help="Language for the character/reference audio, e.g. zh, en, ja, ko, auto",
    )
    parser.add_argument("--ref-audio", required=True, help="Reference audio path")
    parser.add_argument("--ref-text", required=True, help="Transcript for the reference audio")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--out", default="output.wav", help="Output wav path")
    parser.add_argument("--play", action="store_true", help="Play generated audio")
    parser.add_argument("--no-split", action="store_true", help="Disable sentence splitting")
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    ref_audio = os.path.abspath(args.ref_audio)
    out_path = os.path.abspath(args.out)

    if not os.path.isdir(model_dir):
        print(f"[ERROR] model dir not found: {model_dir}", file=sys.stderr)
        return 1
    if not os.path.isfile(ref_audio):
        print(f"[ERROR] reference audio not found: {ref_audio}", file=sys.stderr)
        return 1

    out_parent = os.path.dirname(out_path)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    print("=== Genie Inference ===")
    print(f"model_dir : {model_dir}")
    print(f"character : {args.character}")
    print(f"language  : {args.language}")
    print(f"ref_audio : {ref_audio}")
    print(f"output    : {out_path}")

    genie.load_character(
        character_name=args.character,
        onnx_model_dir=model_dir,
        language=args.language,
    )

    genie.set_reference_audio(
        character_name=args.character,
        audio_path=ref_audio,
        audio_text=args.ref_text,
        language=args.language,
    )

    genie.tts(
        character_name=args.character,
        text=args.text,
        play=args.play,
        split_sentence=not args.no_split,
        save_path=out_path,
    )

    if args.play:
        genie.wait_for_playback_done()

    print(f"\n[OK] synthesis done: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
