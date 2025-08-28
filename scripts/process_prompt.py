import argparse
import torchaudio
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
    print(model_input)