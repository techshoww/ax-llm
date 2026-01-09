import argparse
import os
import torch
import torchaudio
import numpy as np
from frontend import CosyVoiceFrontEnd

def load_wav(wav, target_sr, min_sr=16000):
    speech, sample_rate = torchaudio.load(wav, backend='soundfile')
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate >= min_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument('--model_dir', type=str, default="../Fun-CosyVoice3-0.5B-2512/CosyVoice-BlankEN/", help="tokenizer configuration directionary")
    args.add_argument('--wetext_dir', type=str, default="../CosyVoice/pengzhendong/wetext", help="path to wetext")
    args.add_argument('--sample_rate', type=int, default=24000, help="Sampling rate for prompt audio")
    args.add_argument('--prompt_text', type=str, default="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。", help="The text content of the prompt(reference) audio. Text or file path.")
    args.add_argument('--prompt_speech', type=str, default="asset/zero_shot_prompt.wav", help="The path to prompt(reference) audio.")
    args.add_argument('--output', type=str, default="prompt_files", help="Output data storage directory")
    args = args.parse_args()

    os.makedirs(args.output, exist_ok=True)

    frontend = CosyVoiceFrontEnd(f"{args.model_dir}",
                                args.wetext_dir,
                                "../CosyVoice/frontend-onnx/campplus.onnx",
                                "../CosyVoice/frontend-onnx/speech_tokenizer_v3.onnx",
                                f"{args.model_dir}/spk2info.pt",
                                "all")

    prompt_speech_16k = load_wav(args.prompt_speech, 16000)
    zero_shot_spk_id = ""

    if os.path.isfile(args.prompt_text):
        with open(args.prompt_text, "r") as f:
            prompt_text = f.read()
    else:
        prompt_text = args.prompt_text 
    print("prompt_text",prompt_text)
    model_input = frontend.process_prompt( prompt_text, prompt_speech_16k, args.sample_rate, zero_shot_spk_id)
    
    # model_input = {'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
    #                        'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
    #                        'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
    #                        'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
    #                        'llm_embedding': embedding, 'flow_embedding': embedding}
    print("prompt speech token size:", model_input["flow_prompt_speech_token"].shape)
    assert model_input["flow_prompt_speech_token"].shape[1] >=75, f"speech_token length should >= 75, bug get {model_input['flow_prompt_speech_token'].shape[1]}"
    for k, v in model_input.items():
        if "_len" in k:
            continue
        shapes = [str(s) for s in v.shape]
        shape_str = "_".join(shapes)
        if v.dtype in (torch.int32, torch.int64):
            np.savetxt(f"{args.output}/{k}.txt", v.detach().cpu().numpy().reshape(-1), fmt="%d", delimiter=",")
        else:
            np.savetxt(f"{args.output}/{k}.txt", v.detach().cpu().numpy().reshape(-1), delimiter=",")
