<div align="center">
<pre>
██████╗  ███████╗███╗   ██╗██╗███████╗
██╔════╝ ██╔════╝████╗  ██║██║██╔════╝
██║  ███╗█████╗  ██╔██╗ ██║██║█████╗  
██║   ██║██╔══╝  ██║╚██╗██║██║██╔══╝  
╚██████╔╝███████╗██║ ╚████║██║███████╗
 ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝╚══════╝
</pre>
</div>

<div align="center">

# 🔮 GENIE: [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) Lightweight Inference Engine

**Experience near-instantaneous speech synthesis on your CPU**

[简体中文](./README_zh.md) | [English](./README.md)

</div>

---

**GENIE** is a lightweight inference engine built on the open-source TTS
project [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS). It integrates TTS inference, ONNX model conversion, API
server, and other core features, aiming to provide ultimate performance and convenience.

* **✅ Supported Model Version:** GPT-SoVITS V2, V2ProPlus
* **✅ Supported Language:** Japanese, English, Chinese, Korean, Auto-detect (`language="auto"`)
* **✅ Supported Python Version:** >= 3.10

---

## 🎬 Demo Video

- **[➡️ Watch the demo video (Chinese)](https://www.bilibili.com/video/BV1d2hHzJEz9)**

---

## 🚀 Performance Advantages

GENIE optimizes the original model for outstanding CPU performance.

| Feature                     |  🔮 GENIE   | Official PyTorch Model | Official ONNX Model |
|:----------------------------|:-----------:|:----------------------:|:-------------------:|
| **First Inference Latency** |  **1.13s**  |         1.35s          |        3.57s        |
| **Runtime Size**            | **\~200MB** |      \~several GB      |  Similar to GENIE   |
| **Model Size**              | **\~230MB** |    Similar to GENIE    |       \~750MB       |

> 📝 **Latency Test Info:** All latency data is based on a test set of 100 Japanese sentences (~20 characters each),
> averaged. Tested on CPU i7-13620H.

---

## 🏁 QuickStart

> **⚠️ Important:** It is recommended to run GENIE in **Administrator mode** to avoid potential performance degradation.

### 📦 Installation

Install via pip:

```bash
pip install genie-tts
```

If you want to use the local utility scripts in this repo:

```bash
pip install -e .
pip install torch just
```

### 🛠️ Using the included `justfile`

This fork includes a `justfile` plus ready-to-run scripts for:

- converting GPT-SoVITS **V2** models
- converting GPT-SoVITS **V2ProPlus** models
- testing inference for both converted model types
- testing `language="auto"`

Install `just` first:

```bash
cargo install just
```

or:

```bash
pip install just
```

List available commands:

```bash
just
```

### 🔧 Convert a model

Generic conversion:

```bash
just convert /path/to/s1.ckpt /path/to/s2G.pth /path/to/output_dir
```

V2 example:

```bash
just convert-v2 /path/to/s1.ckpt /path/to/s2G.pth /path/to/v2_onnx
```

V2ProPlus example:

```bash
just convert-v2pp /path/to/s1.ckpt /path/to/s2G.pth /path/to/v2pp_onnx
```

All three commands currently use the same converter entrypoint. The converter auto-detects whether the checkpoint is V2 or V2ProPlus.

### 🎤 Run inference tests

Generic inference:

```bash
just infer /path/to/model_dir /path/to/ref.wav "reference text" "text to synthesize" zh demo ./outputs/demo.wav
```

V2 inference example:

```bash
just infer-v2 /path/to/v2_onnx /path/to/ref.wav "今天天气真不错。" "你好呀，我是沐雪。" zh muxue-v2 ./outputs/v2.wav
```

V2ProPlus inference example:

```bash
just infer-v2pp /path/to/v2pp_onnx /path/to/ref.wav "今天天气真不错。" "你好呀，我是沐雪。" zh muxue-v2pp ./outputs/v2pp.wav
```

Auto language detection example:

```bash
just infer-auto /path/to/model_dir /path/to/ref.wav "你好，今天过得怎么样？" "你好，I love Tokyo！"
```

The `justfile` recipes call these scripts:

- `scripts/convert_model.py`
- `scripts/infer_model.py`

You can also run them directly if you prefer plain Python commands.

## 📥 Pretrained Models

When running GENIE for the first time, it requires downloading resource files (**~391MB**). You can follow the library's
prompts to download them automatically.

> Alternatively, you can manually download the files
> from [HuggingFace](https://huggingface.co/High-Logic/Genie/tree/main/GenieData)
> and place them in a local folder. Then set the `GENIE_DATA_DIR` environment variable **before** importing the library:

```python
import os

# Set the path to your manually downloaded resource files
# Note: Do this BEFORE importing genie_tts
os.environ["GENIE_DATA_DIR"] = r"C:\path\to\your\GenieData"

import genie_tts as genie

# The library will now load resources from the specified directory
```

### ⚡️ Quick Tryout

No GPT-SoVITS model yet? No problem!
GENIE includes several predefined speaker characters you can use immediately —
for example:

* **Mika (聖園ミカ)** — *Blue Archive* (Japanese)
* **ThirtySeven (37)** — *Reverse: 1999* (English)
* **Feibi (菲比)** — *Wuthering Waves* (Chinese)

You can browse all available characters here:
**[https://huggingface.co/High-Logic/Genie/tree/main/CharacterModels](https://huggingface.co/High-Logic/Genie/tree/main/CharacterModels)**

Try it out with the example below:

```python
import genie_tts as genie

genie.load_predefined_character('mika')

genie.tts(
    character_name='mika',
    text='どうしようかな……やっぱりやりたいかも……！',
    play=True,
)

genie.wait_for_playback_done()
```

### 🎤 TTS Best Practices

A simple TTS inference example:

```python
import genie_tts as genie

genie.load_character(
    character_name='<CHARACTER_NAME>',
    onnx_model_dir=r"<PATH_TO_CHARACTER_ONNX_MODEL_DIR>",
    language='<LANGUAGE_CODE>',
)

genie.set_reference_audio(
    character_name='<CHARACTER_NAME>',
    audio_path=r"<PATH_TO_REFERENCE_AUDIO>",
    audio_text="<REFERENCE_AUDIO_TEXT>",
)

genie.tts(
    character_name='<CHARACTER_NAME>',
    text="<TEXT_TO_SYNTHESIZE>",
    play=True,
    save_path="<OUTPUT_AUDIO_PATH>",
)

genie.wait_for_playback_done()
```

---

## 🌐 Launch FastAPI Server

GENIE includes a lightweight FastAPI server:

```python
import genie_tts as genie

genie.start_server(
    host="0.0.0.0",
    port=8000,
    workers=1
)
```

---

## 📝 Roadmap

* [x] **🌐 Language Expansion**

    * [x] Add support for **Chinese** and **English**.

* [x] **🚀 Model Compatibility**

    * [x] Support for `V2Proplus`.
    * [ ] Support for `V3`, `V4`, and more.

* [x] **📦 Easy Deployment**

    * [ ] Release **Official Docker images**.
    * [x] Provide out-of-the-box **Windows bundles**.

---
