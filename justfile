python := "uv run"
out_dir := env_var_or_default("OUT_DIR", "./outputs")

baoqiao_model := ".\\baoqiao_genie"
baoqiao_ref := ".\\baoqiao_genie\\ref_audios\\中间是个很漂亮的大姐姐哦，头发长长的。.wav"
baoqiao_ref_text := "中间是个很漂亮的大姐姐哦，头发长长的。"
baoqiao_text := "虽然我不能亲手送你礼物但是可以为你唱圣诞歌哦。"
baoqiao_language := "zh"
baoqiao_character := "demo-v2"
baoqiao_out := ".\\outputs\\baoqiao_stable.wav"
baoqiao_out_no_split := ".\\outputs\\baoqiao_no_split.wav"

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

infer-roberta model_dir ref_audio ref_text text language="zh" character="demo" out="{{out_dir}}/output_roberta.wav" repeat="1":
	{{python}} scripts/infer_model.py --model-dir "{{model_dir}}" --ref-audio "{{ref_audio}}" --ref-text "{{ref_text}}" --text "{{text}}" --language "{{language}}" --character "{{character}}" --out "{{out}}" --repeat "{{repeat}}" --use-roberta

infer-v2 model_dir ref_audio ref_text text language="zh" character="demo-v2" out="{{out_dir}}/v2.wav" repeat="1":
	{{python}} scripts/infer_model.py --model-dir "{{model_dir}}" --ref-audio "{{ref_audio}}" --ref-text "{{ref_text}}" --text "{{text}}" --language "{{language}}" --character "{{character}}" --out "{{out}}" --repeat "{{repeat}}"

infer-v2-roberta model_dir ref_audio ref_text text language="zh" character="demo-v2" out="{{out_dir}}/v2_roberta.wav" repeat="1":
	{{python}} scripts/infer_model.py --model-dir "{{model_dir}}" --ref-audio "{{ref_audio}}" --ref-text "{{ref_text}}" --text "{{text}}" --language "{{language}}" --character "{{character}}" --out "{{out}}" --repeat "{{repeat}}" --use-roberta

infer-v2pp model_dir ref_audio ref_text text language="zh" character="demo-v2pp" out="{{out_dir}}/v2pp.wav" repeat="1":
	{{python}} scripts/infer_model.py --model-dir "{{model_dir}}" --ref-audio "{{ref_audio}}" --ref-text "{{ref_text}}" --text "{{text}}" --language "{{language}}" --character "{{character}}" --out "{{out}}" --repeat "{{repeat}}"

infer-v2pp-roberta model_dir ref_audio ref_text text language="zh" character="demo-v2pp" out="{{out_dir}}/v2pp_roberta.wav" repeat="1":
	{{python}} scripts/infer_model.py --model-dir "{{model_dir}}" --ref-audio "{{ref_audio}}" --ref-text "{{ref_text}}" --text "{{text}}" --language "{{language}}" --character "{{character}}" --out "{{out}}" --repeat "{{repeat}}" --use-roberta

infer-auto model_dir ref_audio ref_text text character="demo-auto" out="{{out_dir}}/auto.wav" repeat="1":
	{{python}} scripts/infer_model.py --model-dir "{{model_dir}}" --ref-audio "{{ref_audio}}" --ref-text "{{ref_text}}" --text "{{text}}" --language auto --character "{{character}}" --out "{{out}}" --repeat "{{repeat}}"

infer-auto-roberta model_dir ref_audio ref_text text character="demo-auto" out="{{out_dir}}/auto_roberta.wav" repeat="1":
	{{python}} scripts/infer_model.py --model-dir "{{model_dir}}" --ref-audio "{{ref_audio}}" --ref-text "{{ref_text}}" --text "{{text}}" --language auto --character "{{character}}" --out "{{out}}" --repeat "{{repeat}}" --use-roberta

infer-baoqiao:
	{{python}} scripts/infer_model.py --model-dir "{{baoqiao_model}}" --ref-audio "{{baoqiao_ref}}" --ref-text "{{baoqiao_ref_text}}" --text "{{baoqiao_text}}" --language "{{baoqiao_language}}" --character "{{baoqiao_character}}" --out "{{baoqiao_out}}"

infer-baoqiao-roberta:
	{{python}} scripts/infer_model.py --model-dir "{{baoqiao_model}}" --ref-audio "{{baoqiao_ref}}" --ref-text "{{baoqiao_ref_text}}" --text "{{baoqiao_text}}" --language "{{baoqiao_language}}" --character "{{baoqiao_character}}" --out "{{baoqiao_out}}" --use-roberta

infer-baoqiao-repeat repeat="3":
	{{python}} scripts/infer_model.py --model-dir "{{baoqiao_model}}" --ref-audio "{{baoqiao_ref}}" --ref-text "{{baoqiao_ref_text}}" --text "{{baoqiao_text}}" --language "{{baoqiao_language}}" --character "{{baoqiao_character}}" --out "{{baoqiao_out}}" --repeat "{{repeat}}"

infer-baoqiao-repeat-roberta repeat="3":
	{{python}} scripts/infer_model.py --model-dir "{{baoqiao_model}}" --ref-audio "{{baoqiao_ref}}" --ref-text "{{baoqiao_ref_text}}" --text "{{baoqiao_text}}" --language "{{baoqiao_language}}" --character "{{baoqiao_character}}" --out "{{baoqiao_out}}" --repeat "{{repeat}}" --use-roberta

infer-baoqiao-no-split:
	{{python}} scripts/infer_model.py --model-dir "{{baoqiao_model}}" --ref-audio "{{baoqiao_ref}}" --ref-text "{{baoqiao_ref_text}}" --text "{{baoqiao_text}}" --language "{{baoqiao_language}}" --character "{{baoqiao_character}}" --out "{{baoqiao_out_no_split}}" --no-split

infer-baoqiao-no-split-roberta:
	{{python}} scripts/infer_model.py --model-dir "{{baoqiao_model}}" --ref-audio "{{baoqiao_ref}}" --ref-text "{{baoqiao_ref_text}}" --text "{{baoqiao_text}}" --language "{{baoqiao_language}}" --character "{{baoqiao_character}}" --out "{{baoqiao_out_no_split}}" --no-split --use-roberta

infer-baoqiao-repeat-no-split repeat="3":
	{{python}} scripts/infer_model.py --model-dir "{{baoqiao_model}}" --ref-audio "{{baoqiao_ref}}" --ref-text "{{baoqiao_ref_text}}" --text "{{baoqiao_text}}" --language "{{baoqiao_language}}" --character "{{baoqiao_character}}" --out "{{baoqiao_out_no_split}}" --repeat "{{repeat}}" --no-split

infer-baoqiao-repeat-no-split-roberta repeat="3":
	{{python}} scripts/infer_model.py --model-dir "{{baoqiao_model}}" --ref-audio "{{baoqiao_ref}}" --ref-text "{{baoqiao_ref_text}}" --text "{{baoqiao_text}}" --language "{{baoqiao_language}}" --character "{{baoqiao_character}}" --out "{{baoqiao_out_no_split}}" --repeat "{{repeat}}" --no-split --use-roberta

test:
	pytest tests
