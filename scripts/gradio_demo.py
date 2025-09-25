import gradio as gr
import requests
import time
import os

TTS_URL = "http://0.0.0.0:12346/tts"
GET_URL = "http://0.0.0.0:12346/get"
TIMESTEPS_URL = "http://0.0.0.0:12346/timesteps"

def update_timesteps(timesteps):
    try:
        r = requests.post(TIMESTEPS_URL, json={"timesteps": timesteps}, timeout=5)
        if r.status_code != 200:
            return None, "❌ TTS 请求失败"
    except Exception as e:
        return None, f"❌ TTS 请求异常: {e}"

def run_tts(text):
    # Step1: 提交 TTS 请求
    try:
        r = requests.post(TTS_URL, json={"text": text}, timeout=5)
        if r.status_code != 200:
            return None, "❌ TTS 请求失败"
    except Exception as e:
        return None, f"❌ TTS 请求异常: {e}"

    # Step2: 循环调用 /get 获取进度
    progress = gr.Progress()
    wav_file = None
    for i in range(100):  # 最多尝试100次，避免死循环
        time.sleep(0.5)
        try:
            resp = requests.post(GET_URL, data="", timeout=5).json()
        except Exception as e:
            return None, f"❌ GET 请求异常: {e}"

        if resp.get("b_tts_runing", True):
            progress(i / 100, desc="正在生成语音...")
        else:
            wav_file = resp.get("wav_file")
            break

    if not wav_file or not os.path.exists(wav_file):
        return None, "❌ 语音文件未生成"

    return wav_file, "✅ 生成完成"


with gr.Blocks() as demo:
    gr.Markdown("### 🎙️ TTS Demo (调用已有服务器接口)")

    with gr.Row():
        text_input = gr.Textbox(value="琦琦，麻烦你适配一下这个新的模型吧。", label="输入文本")
        with gr.Column():
            timesteps = gr.Slider(minimum=4, maximum=30, value=7, step=1, label="Timesteps")
            run_btn = gr.Button("生成语音")

    status = gr.Label(label="状态")
    audio_out = gr.Audio(label="生成结果", type="filepath")

    run_btn.click(fn=run_tts, inputs=[text_input], outputs=[audio_out, status])
    timesteps.change(fn=update_timesteps, inputs=timesteps)

demo.launch(server_name="0.0.0.0", server_port=12347)
