import argparse
import torch
import torchaudio
import numpy as np
from frontend import CosyVoiceFrontEnd

def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav, backend='soundfile')
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument('--model_dir', type=str, default="../../model_convert/pretrained_models/CosyVoice2-0.5B/")
    args.add_argument('--wetext_dir', type=str, default="../../model_convert/pengzhendong/wetext/")
    args.add_argument('--sample_rate', type=int, default=24000)
    args.add_argument('--zero_shot_spk_id', type=str, default="")
    args.add_argument('--tts_text', type=str, default="君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。")
    args.add_argument('--prompt_text', type=str, default="希望你以后能够做的比我还好呦。")
    args.add_argument('--prompt_speech', type=str, default="../../model_convert/asset/zero_shot_prompt.wav")
    args = args.parse_args()


    frontend = CosyVoiceFrontEnd(f"{args.model_dir}/CosyVoice-BlankEN/",
                                args.wetext_dir,
                                f"{args.model_dir}/campplus.onnx",
                                f"{args.model_dir}/speech_tokenizer_v2.onnx",
                                f"{args.model_dir}/spk2info.pt",
                                "all")

    prompt_speech_16k = load_wav(args.prompt_speech, 16000)
    model_input = frontend.frontend_zero_shot(args.tts_text, args.prompt_text, prompt_speech_16k, args.sample_rate, args.zero_shot_spk_id)
    
    # model_input = {'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
    #                        'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
    #                        'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
    #                        'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
    #                        'llm_embedding': embedding, 'flow_embedding': embedding}
    
    for k, v in model_input.items():
        if "_len" in k:
            continue
        shapes = [str(s) for s in v.shape]
        shape_str = "_".join(shapes)
        if v.dtype in (torch.int32, torch.int64):
            np.savetxt(f"{k}_{shape_str}.txt", v.detach().cpu().numpy().reshape(-1), fmt="%d", delimiter=",")
        else:
            np.savetxt(f"{k}_{shape_str}.txt", v.detach().cpu().numpy().reshape(-1), delimiter=",")


    rand_noise = torch.randn([1, 80,  300])
    np.savetxt("rand_noise_1_80_300.txt", rand_noise.numpy().reshape(-1), delimiter=",")

    speech_window = np.hamming(2 * 8 * 480)
    np.savetxt("speech_window_2x8x480.txt", speech_window.reshape(-1), delimiter=",")