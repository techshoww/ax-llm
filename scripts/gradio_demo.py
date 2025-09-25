import argparse
import shutil
import gradio as gr
import numpy as np
import requests
import time
import os

import torch
from frontend import CosyVoiceFrontEnd
import torchaudio
import logging
logging.basicConfig(level=logging.WARNING)

import subprocess
import re

def get_all_local_ips():
    result = subprocess.run(['ip', 'a'], capture_output=True, text=True)
    output = result.stdout

    # 匹配所有IPv4
    ips = re.findall(r'inet (\d+\.\d+\.\d+\.\d+)', output)

    # 过滤掉回环地址
    real_ips = [ip for ip in ips if not ip.startswith('127.')]

    return real_ips


TTS_URL = "http://0.0.0.0:12346/tts"
GET_URL = "http://0.0.0.0:12346/get"
TIMESTEPS_URL = "http://0.0.0.0:12346/timesteps"
PROMPT_FILES_URL = "http://0.0.0.0:12346/prompt_files"

args = argparse.ArgumentParser()
args.add_argument('--model_dir', type=str, default="scripts/CosyVoice-BlankEN", help="tokenizer configuration directionary")
args.add_argument('--wetext_dir', type=str, default="pengzhendong/wetext", help="path to wetext")
args.add_argument('--sample_rate', type=int, default=24000, help="Sampling rate for prompt audio")
args = args.parse_args()
frontend = CosyVoiceFrontEnd(f"{args.model_dir}",
                                args.wetext_dir,
                                "frontend-onnx/campplus.onnx",
                                "frontend-onnx/speech_tokenizer_v2.onnx",
                                f"{args.model_dir}/spk2info.pt",
                                "all")

def update_audio(audio_input_path, audio_text):
    def load_wav(wav, target_sr):
        speech, sample_rate = torchaudio.load(wav, backend='soundfile')
        speech = speech.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
            speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
        return speech
    output_dir = './output_temp'
    # clear output_dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    zero_shot_spk_id = ""
    prompt_speech_16k = load_wav(audio_input_path, 16000)
    prompt_text = audio_text
    print("prompt_text",prompt_text)
    model_input = frontend.process_prompt( prompt_text, prompt_speech_16k, args.sample_rate, zero_shot_spk_id)
    print("prompt speech token size:", model_input["flow_prompt_speech_token"].shape)
    assert model_input["flow_prompt_speech_token"].shape[1] >=75, f"speech_token length should >= 75, bug get {model_input['flow_prompt_speech_token'].shape[1]}"
    for k, v in model_input.items():
        if "_len" in k:
            continue
        shapes = [str(s) for s in v.shape]
        shape_str = "_".join(shapes)
        if v.dtype in (torch.int32, torch.int64):
            np.savetxt(f"{output_dir}/{k}.txt", v.detach().cpu().numpy().reshape(-1), fmt="%d", delimiter=",")
        else:
            np.savetxt(f"{output_dir}/{k}.txt", v.detach().cpu().numpy().reshape(-1), delimiter=",")

    try:
        r = requests.post(PROMPT_FILES_URL, json={"prompt_files": output_dir}, timeout=5)
        if r.status_code != 200:
            return None, "❌ TTS 请求失败"
    except Exception as e:
        return None, f"❌ TTS 请求异常: {e}"
    

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
    gr.Markdown("### 🎙️ AXERA CosyVoice2 Demo")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="输入音频", type="filepath")
        with gr.Column():
            audio_text = gr.Textbox(label="音频文本(自己改一下或者照着念)", value="锄禾日当午，汗滴禾下土。")
            btn_update = gr.Button("更新音源")
        

    with gr.Row():
        text_input = gr.Textbox(value="琦琦，麻烦你适配一下这个新的模型吧。", label="输入文本")
        with gr.Column():
            timesteps = gr.Slider(minimum=4, maximum=30, value=7, step=1, label="Timesteps")
            run_btn = gr.Button("生成语音")

    status = gr.Label(label="状态")
    audio_out = gr.Audio(label="生成结果", type="filepath")

    run_btn.click(fn=run_tts, inputs=[text_input], outputs=[audio_out, status])
    timesteps.change(fn=update_timesteps, inputs=timesteps)
    
    btn_update.click(fn=update_audio, inputs=[audio_input, audio_text])

ips = get_all_local_ips()
for ip in ips:
    print(f"* Running on local URL:  https://{ip}:7860")


demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    ssl_certfile="./server.crt",
    ssl_keyfile="./server.key",
    ssl_verify=False
)
