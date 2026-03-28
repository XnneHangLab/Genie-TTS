set shell := ["bash", "-cu"]

python := env_var_or_default("PYTHON", "python")
out_dir := env_var_or_default("OUT_DIR", "./outputs")

_default:
	@just --list

install:
	{{python}} -m pip install -e .

install-dev:
	{{python}} -m pip install -e .
	{{python}} -m pip install torch just

convert ckpt pth out force_version="":
	if [ -n "{{force_version}}" ]; then \
		{{python}} scripts/convert_model.py --ckpt "{{ckpt}}" --pth "{{pth}}" --out "{{out}}" --force-version "{{force_version}}"; \
	else \
		{{python}} scripts/convert_model.py --ckpt "{{ckpt}}" --pth "{{pth}}" --out "{{out}}"; \
	fi

convert-auto model_dir out="" force_version="":
	if [ -n "{{out}}" ]; then \
		if [ -n "{{force_version}}" ]; then \
			{{python}} scripts/convert_auto.py "{{model_dir}}" --out "{{out}}" --force-version "{{force_version}}"; \
		else \
			{{python}} scripts/convert_auto.py "{{model_dir}}" --out "{{out}}"; \
		fi; \
	else \
		if [ -n "{{force_version}}" ]; then \
			{{python}} scripts/convert_auto.py "{{model_dir}}" --force-version "{{force_version}}"; \
		else \
			{{python}} scripts/convert_auto.py "{{model_dir}}"; \
		fi; \
	fi

convert-v2 ckpt pth out:
	{{python}} scripts/convert_model.py --ckpt "{{ckpt}}" --pth "{{pth}}" --out "{{out}}" --force-version v2

convert-v2pp ckpt pth out:
	{{python}} scripts/convert_model.py --ckpt "{{ckpt}}" --pth "{{pth}}" --out "{{out}}" --force-version v2pp

infer model_dir ref_audio ref_text text language="zh" character="demo" out="{{out_dir}}/output.wav":
	{{python}} scripts/infer_model.py --model-dir "{{model_dir}}" --ref-audio "{{ref_audio}}" --ref-text "{{ref_text}}" --text "{{text}}" --language "{{language}}" --character "{{character}}" --out "{{out}}"

infer-v2 model_dir ref_audio ref_text text language="zh" character="demo-v2" out="{{out_dir}}/v2.wav":
	{{python}} scripts/infer_model.py --model-dir "{{model_dir}}" --ref-audio "{{ref_audio}}" --ref-text "{{ref_text}}" --text "{{text}}" --language "{{language}}" --character "{{character}}" --out "{{out}}"

infer-v2pp model_dir ref_audio ref_text text language="zh" character="demo-v2pp" out="{{out_dir}}/v2pp.wav":
	{{python}} scripts/infer_model.py --model-dir "{{model_dir}}" --ref-audio "{{ref_audio}}" --ref-text "{{ref_text}}" --text "{{text}}" --language "{{language}}" --character "{{character}}" --out "{{out}}"

infer-auto model_dir ref_audio ref_text text character="demo-auto" out="{{out_dir}}/auto.wav":
	{{python}} scripts/infer_model.py --model-dir "{{model_dir}}" --ref-audio "{{ref_audio}}" --ref-text "{{ref_text}}" --text "{{text}}" --language auto --character "{{character}}" --out "{{out}}"

test:
	pytest tests
