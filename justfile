set shell := ["bash", "-cu"]

python := env_var_or_default("PYTHON", "python")
out_dir := env_var_or_default("OUT_DIR", "./outputs")

baoqiao_model := ".\\baoqiao_genie"
baoqiao_ref := ".\\baoqiao_genie\\emotions\\这个名字，是不是很适合他，长的像猫咪一样嘛.wav"
baoqiao_ref_text := "这个名字，是不是很适合他，长的像猫咪一样嘛"
baoqiao_text := "我起床的时候已经九点了，透过窗子看到凶案现场，现场的警察需要我做一些解释，毕竟死者死在我家院子里。"
baoqiao_language := "zh"
baoqiao_character := "demo-v2"
baoqiao_out := ".\\outputs\\baoqiao_v2.wav"

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

infer model_dir ref_audio ref_text text language="zh" character="demo" out="{{out_dir}}/output.wav" repeat="1":
	{{python}} scripts/infer_model.py --model-dir "{{model_dir}}" --ref-audio "{{ref_audio}}" --ref-text "{{ref_text}}" --text "{{text}}" --language "{{language}}" --character "{{character}}" --out "{{out}}" --repeat "{{repeat}}"

infer-v2 model_dir ref_audio ref_text text language="zh" character="demo-v2" out="{{out_dir}}/v2.wav" repeat="1":
	{{python}} scripts/infer_model.py --model-dir "{{model_dir}}" --ref-audio "{{ref_audio}}" --ref-text "{{ref_text}}" --text "{{text}}" --language "{{language}}" --character "{{character}}" --out "{{out}}" --repeat "{{repeat}}"

infer-v2pp model_dir ref_audio ref_text text language="zh" character="demo-v2pp" out="{{out_dir}}/v2pp.wav" repeat="1":
	{{python}} scripts/infer_model.py --model-dir "{{model_dir}}" --ref-audio "{{ref_audio}}" --ref-text "{{ref_text}}" --text "{{text}}" --language "{{language}}" --character "{{character}}" --out "{{out}}" --repeat "{{repeat}}"

infer-auto model_dir ref_audio ref_text text character="demo-auto" out="{{out_dir}}/auto.wav" repeat="1":
	{{python}} scripts/infer_model.py --model-dir "{{model_dir}}" --ref-audio "{{ref_audio}}" --ref-text "{{ref_text}}" --text "{{text}}" --language auto --character "{{character}}" --out "{{out}}" --repeat "{{repeat}}"

infer-baoqiao:
	{{python}} scripts/infer_model.py --model-dir "{{baoqiao_model}}" --ref-audio "{{baoqiao_ref}}" --ref-text "{{baoqiao_ref_text}}" --text "{{baoqiao_text}}" --language "{{baoqiao_language}}" --character "{{baoqiao_character}}" --out "{{baoqiao_out}}"

infer-baoqiao-repeat repeat="3":
	{{python}} scripts/infer_model.py --model-dir "{{baoqiao_model}}" --ref-audio "{{baoqiao_ref}}" --ref-text "{{baoqiao_ref_text}}" --text "{{baoqiao_text}}" --language "{{baoqiao_language}}" --character "{{baoqiao_character}}" --out "{{baoqiao_out}}" --repeat "{{repeat}}"

infer-baoqiao-stable:
	{{python}} scripts/infer_model.py --model-dir "{{baoqiao_model}}" --ref-audio "{{baoqiao_ref}}" --ref-text "{{baoqiao_ref_text}}" --text "{{baoqiao_text}}" --language "{{baoqiao_language}}" --character "{{baoqiao_character}}" --out ".\\outputs\\baoqiao_stable.wav"

infer-baoqiao-stable-repeat repeat="5":
	{{python}} scripts/infer_model.py --model-dir "{{baoqiao_model}}" --ref-audio "{{baoqiao_ref}}" --ref-text "{{baoqiao_ref_text}}" --text "{{baoqiao_text}}" --language "{{baoqiao_language}}" --character "{{baoqiao_character}}" --out ".\\outputs\\baoqiao_stable.wav" --repeat "{{repeat}}"

test:
	pytest tests
