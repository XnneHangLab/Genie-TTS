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

# 🔮 GENIE: [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 轻量级推理引擎

**在 CPU 上体验近乎即时的语音合成**

[简体中文](./README_zh.md) | [English](./README.md)

</div>

---

**GENIE** 是一个基于开源 TTS 项目 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 构建的轻量级推理引擎。它集成了 TTS 推理、ONNX 模型转换、API 服务端以及其他核心功能，旨在提供极致的性能和便利性。

* **✅ 支持的模型版本：** GPT-SoVITS V2, V2ProPlus
* **✅ 支持的语言：** 日语、英语、中文、韩语、自动检测（`language="auto"`）
* **✅ 支持的 Python 版本：** >= 3.10

---

## 🎬 演示视频

- **[➡️ 观看演示视频（中文）](https://www.bilibili.com/video/BV1d2hHzJEz9)**

---

## 🚀 性能优势

GENIE 针对原始模型进行了优化，以实现出色的 CPU 性能。

| 特性         |  🔮 GENIE   | 官方 PyTorch 模型 | 官方 ONNX 模型 |
|:-----------|:-----------:|:-------------:|:----------:|
| **首次推理延迟** |  **1.13s**  |     1.35s     |   3.57s    |
| **运行时大小**  | **\~200MB** |    \~数 GB     | 与 GENIE 相似 |
| **模型大小**   | **\~230MB** |  与 GENIE 相似   |  \~750MB   |

> 📝 **延迟测试说明：** 所有延迟数据均基于 100 个日语句子（每句约 20 个字符）的测试集取平均值。测试环境为 CPU i7-13620H。

---

## 🏁 快速开始

> **⚠️ 重要提示：** 建议在 **管理员模式** 下运行 GENIE，以避免潜在的性能下降。

### 📦 安装

通过 pip 安装：

```bash
pip install genie-tts
```

如果你想使用这个 fork 里附带的转换 / 推理脚本：

```bash
pip install -e .
pip install torch just
```

### 🛠️ 使用仓库内置的 `justfile`

这个 fork 额外提供了一个 `justfile`，配套了可直接调用的脚本，用来：

- 转换 GPT-SoVITS **V2** 模型
- 转换 GPT-SoVITS **V2ProPlus** 模型
- 测试两类模型的推理
- 测试 `language="auto"`

先安装 `just`：

```bash
cargo install just
```

或者：

```bash
pip install just
```

查看全部命令：

```bash
just
```

### 🔧 模型转换

通用转换：

```bash
just convert /path/to/s1.ckpt /path/to/s2G.pth /path/to/output_dir
```

V2 示例：

```bash
just convert-v2 /path/to/s1.ckpt /path/to/s2G.pth /path/to/v2_onnx
```

V2ProPlus 示例：

```bash
just convert-v2pp /path/to/s1.ckpt /path/to/s2G.pth /path/to/v2pp_onnx
```

目前这三个命令都走同一个转换入口，转换器会自动判断输入更像 V2 还是 V2ProPlus。

### 🎤 推理测试

通用推理：

```bash
just infer /path/to/model_dir /path/to/ref.wav "参考音频文本" "要合成的文本" zh demo ./outputs/demo.wav
```

V2 推理示例：

```bash
just infer-v2 /path/to/v2_onnx /path/to/ref.wav "今天天气真不错。" "你好呀，我是沐雪。" zh muxue-v2 ./outputs/v2.wav
```

V2ProPlus 推理示例：

```bash
just infer-v2pp /path/to/v2pp_onnx /path/to/ref.wav "今天天气真不错。" "你好呀，我是沐雪。" zh muxue-v2pp ./outputs/v2pp.wav
```

自动语言检测示例：

```bash
just infer-auto /path/to/model_dir /path/to/ref.wav "你好，今天过得怎么样？" "你好，I love Tokyo！"
```

`justfile` 实际调用的是这两个脚本：

- `scripts/convert_model.py`
- `scripts/infer_model.py`

如果你更习惯直接写 Python 命令，也可以绕过 `just` 直接调用它们。

## 📥 预训练模型

首次运行 GENIE 时，需要下载资源文件（**~391MB**）。你可以按照库的提示自动下载。

> 或者，你也可以从 [HuggingFace](https://huggingface.co/High-Logic/Genie/tree/main/GenieData) 手动下载文件并放到本地目录，然后在导入库之前设置 `GENIE_DATA_DIR` 环境变量：

```python
import os

os.environ["GENIE_DATA_DIR"] = r"C:\path\to\your\GenieData"

import genie_tts as genie
```

### ⚡️ 快速试用

还没有 GPT-SoVITS 模型？没关系。
GENIE 内置了一些预定义角色，例如：

* **Mika (聖園ミカ)** — *蔚蓝档案*（日语）
* **ThirtySeven (37)** — *重返未来：1999*（英语）
* **Feibi (菲比)** — *鸣潮*（中文）

可用角色见：
**[https://huggingface.co/High-Logic/Genie/tree/main/CharacterModels](https://huggingface.co/High-Logic/Genie/tree/main/CharacterModels)**

示例：

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

### 🎤 TTS 基本用法

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

## 🌐 启动 FastAPI 服务

```python
import genie_tts as genie

genie.start_server(
    host="0.0.0.0",
    port=8000,
    workers=1
)
```

---

## 📝 路线图

* [x] **🌐 语言扩展**

    * [x] 添加对 **中文** 和 **英文** 的支持。

* [x] **🚀 模型兼容性**

    * [x] 支持 `V2Proplus`。
    * [ ] 支持 `V3`、`V4` 等更多版本。

* [x] **📦 简易部署**

    * [ ] 发布 **官方 Docker 镜像**。
    * [x] 提供开箱即用的 **Windows 整合包**。
