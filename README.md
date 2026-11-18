# 皮肤健康多模态问诊 Agent

一个可本地运行的对话式智能体：**实拍照片 + 语音描述 + 文字提问** → 皮肤症状的科普性初步分析。

全部推理在本地完成，图片与语音不出本机。

> ⚠️ 本项目仅用于健康科普与教学演示，**不构成医疗诊断**，不能替代执业医师意见。

## 功能

- **图文理解**：Qwen3-VL-2B 视觉语言模型，识别照片中的皮损颜色、边界、形态、分布等客观特征
- **语音输入**：FunASR paraformer-zh 中文语音识别（含 VAD 断句与标点恢复）
- **多轮对话**：支持连续追问，上下文保留
- **分层输出**：观察所见 → 可能方向（2-3 种，按可能性排序）→ 居家护理建议 → 就医提示
- **安全护栏**：系统提示词禁止确定性诊断与处方药推荐，每条回复强制附加免责声明
- **显存友好**：历史图片只保留在最新一轮；GPU 下自动启用 4-bit 量化（约 2.5GB 显存）

## 技术栈

| 环节 | 方案 |
| --- | --- |
| 视觉理解 | Qwen3-VL-2B-Instruct（transformers 原生接口，可选 MiniCPM-V） |
| 语音识别 | FunASR paraformer-zh + fsmn-vad + ct-punc |
| Web 界面 | Gradio |
| 量化推理 | bitsandbytes NF4 4-bit |

## 项目结构

```
skin-health-agent/
├── app.py            # 全部逻辑：模型封装 + 语音识别 + Web 界面
├── requirements.txt  # 依赖清单
├── run.bat           # Windows 一键启动
├── README.md
└── models/           # 模型权重目录（.gitignore 已排除，需自行下载）
```

## 环境要求

- Python 3.10 – 3.12
- 有 NVIDIA 显卡：显存 ≥ 6GB 时自动开启 4-bit 量化，回复数秒级
- 无独显：纯 CPU 运行，约需 8GB 内存，单条回复 1–3 分钟

## 快速开始

### 1. 准备环境

```bash
git clone https://github.com/<your-name>/skin-health-agent.git
cd skin-health-agent

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

有 NVIDIA 显卡时建议改装 CUDA 版 PyTorch（否则默认是 CPU 版，速度很慢）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. 下载视觉模型（约 4.3GB）

```bash
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen3-VL-2B-Instruct', local_dir='./models/qwen3-vl-2b'))"
```

国内使用 ModelScope 比 HuggingFace 镜像稳定，不会触发 429 限流。

### 3. 启动

```bash
python app.py --model ./models/qwen3-vl-2b
```

Windows 也可以直接双击 `run.bat`。

语音识别模型（FunASR，约 1GB）首次运行时会自动下载。

看到下面的日志即启动成功，浏览器打开对应地址：

```
* Running on local URL:  http://127.0.0.1:7860
```

服务窗口需保持运行，关闭窗口即停止服务。

## 命令行参数

| 参数 | 说明 |
| --- | --- |
| `--model` | 模型 ID 或本地路径，默认优先使用 `./models/qwen3-vl-2b` |
| `--device` | `cuda` / `cpu`，默认自动检测 |
| `--backend` | `qwen` / `minicpm`，默认按模型名判断 |
| `--no-asr` | 跳过 FunASR，仅图文模式 |
| `--share` | 生成 Gradio 公网临时链接 |

## 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `403 GatedRepoError` | MiniCPM-V 在 HuggingFace 为受限仓库 | 改用 Qwen 模型，或 ModelScope 下载后传本地路径 |
| `429 Rate limited` | hf-mirror 限流 | 等自动重试，或改用 ModelScope 下载 |
| `requires accelerate` | 缺少量化依赖 | `pip install accelerate bitsandbytes` |
| `Chatbot got unexpected keyword 'type'` | Gradio 版本过旧 | `pip install -U "gradio>=4.44"` |
| 浏览器“拒绝连接” | 服务未运行 | 重新启动并等待 `Running on local URL` 出现 |
| `ffmpeg is not installed` | 提示性警告 | 可忽略，torchaudio 可正常读取音频 |

## 后续规划

1. 接入皮肤科指南/教科书做 RAG 检索增强
2. 使用 HAM10000 数据集 LoRA 微调，提升皮损分类准确率
3. 增加 TTS 语音播报回复

## 免责声明

本项目输出由 AI 生成，仅供健康科普参考，不构成医疗诊断，请勿据此自行用药。如有疑虑或症状持续加重，请及时前往正规医院皮肤科就诊。
