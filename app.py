"""皮肤健康多模态问诊 Agent (MVP)。

组件:
  - MiniCPM-V : 图文多模态理解 (实拍照片 + 症状描述 -> 初步分析)
  - FunASR    : 语音转文字 (麦克风/音频文件 -> 症状描述)
  - Gradio    : Web 聊天界面

声明: 本项目仅用于健康科普与教学演示, 不能替代专业医疗诊断。
"""

import argparse
import os

import gradio as gr

# 每次回复末尾附加的安全声明
DISCLAIMER = (
    "\n\n---\n"
    "⚠️ 以上内容为 AI 生成的健康科普参考，不构成医疗诊断。"
    "如症状持续、加重或伴有发热等全身表现，请及时前往正规医院皮肤科就诊。"
)

# 皮肤分诊助手的系统提示词
SYSTEM_PROMPT = (
    "你是一名皮肤健康咨询助手，面向普通用户提供科普性质的初步分析。"
    "请根据用户描述的症状与上传的实拍照片，按以下结构回复：\n"
    "1. 观察所见：照片中可观察到的客观特征（颜色、边界、形态、大小、分布等）；\n"
    "2. 可能方向：列出 2-3 种常见可能性（按可能性排序），使用通俗语言，不做确定性诊断；\n"
    "3. 居家护理建议：温和、安全的一般性护理措施；\n"
    "4. 就医提示：出现哪些情况时应尽快就医。\n"
    "要求：不推荐处方药；语气耐心温和；始终用中文回复。"
)


class MiniCPMChat:
    """MiniCPM-V 图文多模态对话封装（需从 ModelScope 下载到本地后用 --model 指定）。"""

    def __init__(self, model_id, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        print(f"[VLM] 使用设备: {self.device} (dtype={dtype})")
        self.model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    def generate(self, msgs):
        """输入完整多轮 msgs（最新一轮可含图片），返回模型回复文本。"""
        out = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=0.6,
            system_prompt=SYSTEM_PROMPT,
            stream=False,
        )
        if isinstance(out, tuple):  # 兼容不同版本返回值
            out = out[0]
        return str(out).strip()


class QwenVLChat:
    """Qwen2.5-VL 图文多模态对话封装（模型开放下载，无需申请访问权限）。"""

    def __init__(self, model_id, device=None):
        import torch
        from transformers import AutoProcessor
        try:  # 新版 transformers 改名，旧版保留旧类名
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVLM

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            # 4-bit 量化：2B 模型约占 2.5GB 显存，6GB 卡可舒适运行
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            print("[VLM] 使用设备: cuda (4-bit 量化加载)")
            self.model = AutoVLM.from_pretrained(
                model_id, quantization_config=quant, device_map="cuda:0"
            ).eval()
        else:
            print("[VLM] 使用设备: cpu (dtype=torch.float32)")
            self.model = AutoVLM.from_pretrained(
                model_id, dtype=torch.float32
            ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def generate(self, msgs):
        qmsgs = [{"role": "system",
                  "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
        image = None
        for m in msgs:
            if isinstance(m["content"], list):
                texts = [c for c in m["content"] if isinstance(c, str)]
                imgs = [c for c in m["content"] if not isinstance(c, str)]
                if imgs:
                    image = imgs[0]
                content = [{"type": "image"}] + [
                    {"type": "text", "text": t} for t in texts
                ]
            else:
                content = [{"type": "text", "text": m["content"]}]
            qmsgs.append({"role": m["role"], "content": content})

        text = self.processor.apply_chat_template(
            qmsgs, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image] if image is not None else None,
            padding=True, return_tensors="pt",
        ).to(self.device)
        gen = self.model.generate(
            **inputs, max_new_tokens=512,
            do_sample=True, temperature=0.6, top_p=0.9,
        )
        out = gen[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()


def make_vlm(model_id, backend, device):
    """按模型名自动选择接口：Qwen 系列走原生实现，其余走 MiniCPM chat 接口。"""
    backend = backend or ("qwen" if "qwen" in model_id.lower() else "minicpm")
    cls = QwenVLChat if backend == "qwen" else MiniCPMChat
    return cls(model_id, device)


class SpeechToText:
    """FunASR 中文语音识别封装（paraformer-zh，自动下载权重）。"""

    def __init__(self):
        from funasr import AutoModel

        self.model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            disable_update=True,
        )

    def transcribe(self, audio_path):
        res = self.model.generate(input=audio_path, batch_size_s=60)
        return (res[0].get("text") or "").strip()


UI_CSS = """
/* 顶部横幅：青绿渐变（医疗健康感） */
#hero {
    background: linear-gradient(120deg, #0f766e 0%, #14b8a6 55%, #2dd4bf 100%);
    border-radius: 16px;
    padding: 22px 28px;
    margin-bottom: 14px;
}
#hero h1 { color: #ffffff; margin: 0 0 6px; font-size: 1.5em; letter-spacing: 1px; }
#hero p  { color: #d9f5f1; margin: 0; font-size: 0.95em; }
/* 底部免责声明 */
#footer-note { color: #6b7280; font-size: 0.85em; text-align: center; margin-top: 10px; }
/* 图片/音频/文本框圆角 */
.form { border-radius: 14px; }
"""


def build_ui(vlm, asr):
    with gr.Blocks(title="皮肤健康问诊助手") as demo:  # theme/css 在 launch() 中传入（gradio 6 新规）
        gr.Markdown(
            "# 🩺 皮肤健康多模态问诊助手\n"
            "上传实拍照片 · 语音描述 · 文字追问，获取科普性质的初步分析",
            elem_id="hero",
        )
        chatbot = gr.Chatbot(height=440, label="对话")  # gradio 6 默认即为 messages 格式
        with gr.Row():
            image_in = gr.Image(
                type="pil", label="实拍照片（可选）", sources=["upload", "webcam"], height=220
            )
            with gr.Column():
                audio_in = gr.Audio(
                    type="filepath", label="语音描述（可选）",
                    sources=["microphone", "upload"], format="wav",
                )
                text_in = gr.Textbox(
                    label="文字描述",
                    placeholder="例如：手背有一块红斑，边界清楚，微痒，三天了……",
                    lines=2,
                )
        gr.Examples(
            examples=[
                ["手背有一块红斑，边界清楚，微痒，三天了"],
                ["脸上最近反复长痘，红肿，按压会痛"],
                ["脚趾甲变厚发黄，还有点碎屑"],
            ],
            inputs=[text_in],
            label="试试这些例子",
        )
        with gr.Row():
            send_btn = gr.Button("发送", variant="primary")
            clear_btn = gr.Button("清空对话")
        gr.Markdown(
            "⚠️ 本工具输出仅供健康科普参考，不构成医疗诊断；如有疑虑请及时就医。",
            elem_id="footer-note",
        )
        # 模型侧多轮上下文（图片只保留在最新一轮，避免显存膨胀）
        msgs_state = gr.State([])

        def respond(text, audio, img, history, msgs):
            text = (text or "").strip()

            # 1) 语音优先转写，与文字合并
            if audio:
                try:
                    t = asr.transcribe(audio)
                except Exception as e:  # ASR 失败不阻断对话
                    history = history + [{"role": "assistant",
                                          "content": f"语音识别失败：{e}\n请直接用文字描述。"}]
                    return history, msgs, "", None, None
                if t:
                    text = f"{text}（语音转写：{t}）" if text else t

            if not text and img is None:
                return history, msgs, "", None, None

            # 2) 组装模型上下文与界面消息（历史图片只保留文字，控制内存）
            if img is not None:
                msgs = [
                    {"role": m["role"],
                     "content": "".join(c for c in m["content"] if isinstance(c, str))
                     if isinstance(m["content"], list) else m["content"]}
                    for m in msgs
                ]
                content = [img, text]
            else:
                content = text
            msgs = msgs + [{"role": "user", "content": content}]
            history = history + [{
                "role": "user",
                "content": text + ("\n📷 已附实拍照片" if img is not None else ""),
            }]

            # 3) 调用 VLM 生成回复
            try:
                reply = vlm.generate(msgs)
            except Exception as e:
                reply = f"模型推理失败：{e}"
            msgs = msgs + [{"role": "assistant", "content": reply}]
            history = history + [{"role": "assistant", "content": reply + DISCLAIMER}]

            return history, msgs, "", None, None

        def clear_all():
            return [], [], "", None, None

        send_btn.click(
            respond,
            [text_in, audio_in, image_in, chatbot, msgs_state],
            [chatbot, msgs_state, text_in, image_in, audio_in],
        )
        text_in.submit(
            respond,
            [text_in, audio_in, image_in, chatbot, msgs_state],
            [chatbot, msgs_state, text_in, image_in, audio_in],
        )
        clear_btn.click(
            clear_all, None, [chatbot, msgs_state, text_in, image_in, audio_in]
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description="皮肤健康多模态问诊 Agent MVP")
    default_model = (
        "./models/qwen3-vl-2b" if os.path.isdir("./models/qwen3-vl-2b")
        else os.environ.get("VLM_MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct")
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help="VLM 模型 ID 或本地路径，默认优先使用本地 qwen3-vl-2b",
    )
    parser.add_argument("--device", default=None, help="cuda / cpu，默认自动检测")
    parser.add_argument(
        "--backend", default=None, choices=["minicpm", "qwen"],
        help="模型接口类型，默认按模型名自动判断",
    )
    parser.add_argument("--no-asr", action="store_true", help="跳过 FunASR，仅图文模式")
    parser.add_argument("--share", action="store_true", help="创建公网访问链接")
    args = parser.parse_args()

    print(f"[VLM] 加载 {args.model} ...（首次运行会自动下载权重，请耐心等待）")
    vlm = make_vlm(args.model, args.backend, args.device)

    asr = None
    if not args.no_asr:
        try:
            print("[ASR] 加载 FunASR paraformer-zh ...")
            asr = SpeechToText()
        except Exception as e:
            print(f"[ASR] 加载失败，将以纯图文模式运行：{e}")

    build_ui(vlm, asr).launch(
        share=args.share,
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
        css=UI_CSS,
    )


if __name__ == "__main__":
    main()
