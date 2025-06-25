#include "signal.h"

#include "runner/LLM.hpp"

#include "cmdline.hpp"

#include <opencv2/opencv.hpp>

#include "runner/utils/image_processor.hpp"

#include "runner/utils/files.hpp"

#include "runner/utils/mrope.hpp"

#include "runner/OmniModel.hpp"

static OmniModel lLaMa;

void __sigExit(int iSigNo)
{
    lLaMa.Stop();
    return;
}

void llm_running_callback(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve)
{
    fprintf(stdout, "%s", p_str);
    fflush(stdout);
}

std::string prompt_complete(std::string prompt, TokenizerType tokenizer_type)
{
    std::ostringstream oss_prompt;
    switch (tokenizer_type)
    {
    case TKT_LLaMa:
        oss_prompt << "<|user|>\n"
                   << prompt << "</s><|assistant|>\n";
        break;
    case TKT_MINICPM:
        oss_prompt << "<用户><image></image>\n";
        oss_prompt << prompt << "<AI>";
        break;
    case TKT_Phi3:
        oss_prompt << prompt << " ";
        break;
    case TKT_Qwen:
        oss_prompt << "<|im_start|>system\nYou are a helpful assistant.<|im_end|>";
        oss_prompt << "\n<|im_start|>user\n"
                   << prompt << "<|im_end|>\n<|im_start|>assistant\n";
        break;
    case TKT_HTTP:
    default:
        oss_prompt << prompt;
        break;
    }

    return oss_prompt.str();
}
int main(int argc, char *argv[])
{
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, __sigExit);

    OmniAttr attr;
    attr.path_audio_encoder = "../../Qwen2.5-Omni-3B-AX650N-prefill352/audio_tower.axmodel";
    attr.path_visual_encoder = "../../Qwen2.5-Omni-3B-AX650N-prefill352/Qwen2.5-Omni-3B_vision.axmodel";
    attr.path_token2wav_dit = "../../Qwen2.5-Omni-3B-AX650N-talker-prefill352/token2wav_dit.axmodel";
    attr.path_token2wav_bigvgan = "../model_convert/build-output-bigvgan/token2wav_bigvgan.axmodel";
    LLMAttrType attr_thinker;
    attr_thinker.template_filename_axmodel = "../../Qwen2.5-Omni-3B-AX650N-prefill352/qwen2_5_omni_text_p352_l%d_together.axmodel";
    attr_thinker.axmodel_num = 36;
    attr_thinker.filename_post_axmodel = "../../Qwen2.5-Omni-3B-AX650N-prefill352/qwen2_5_omni_text_post.axmodel";
    attr_thinker.tokenizer_type = TKT_HTTP;
    attr_thinker.filename_tokenizer_model = "http://10.122.86.184:8080";
    attr_thinker.b_bos = false;
    attr_thinker.b_eos = false;
    attr_thinker.filename_tokens_embed = "../../Qwen2.5-Omni-3B-AX650N-prefill352/model.embed_tokens.weight.bfloat16.bin";
    attr_thinker.tokens_embed_num = 151936;
    attr_thinker.tokens_embed_size = 2048;
    attr_thinker.b_dynamic_load_axmodel_layer = true;
    attr_thinker.b_use_mmap_load_embed = true;

    attr.attr_thinker_text_model = attr_thinker;

    TalkerAttr attr_talker;
    attr_talker.template_filename_axmodel = "../../Qwen2.5-Omni-3B-AX650N-talker-prefill352/qwen2_5_omni_talker_p352_l%d_together.axmodel";
    attr_talker.axmodel_num = 24;
    attr_talker.filename_post_axmodel = "../../Qwen2.5-Omni-3B-AX650N-talker-prefill352/qwen2_5_omni_talker_post.axmodel";
    attr_talker.b_bos = false;
    attr_talker.b_eos = false;
    attr_talker.filename_tokens_embed = "../../Qwen2.5-Omni-3B-AX650N-talker-prefill352/model.embed_tokens.weight.bfloat16.bin";
    attr_talker.tokens_embed_num = 8448;
    attr_talker.tokens_embed_size = 2048;
    attr_talker.b_dynamic_load_axmodel_layer = true;
    attr_talker.b_use_mmap_load_embed = true;
    attr_talker.filename_proj_prefill_axmodel = "../../Qwen2.5-Omni-3B-AX650N-talker-prefill352/thinker_to_talker_proj_prefill_352.axmodel";
    attr_talker.filename_proj_decode_axmodel = "../../Qwen2.5-Omni-3B-AX650N-talker-prefill352/thinker_to_talker_proj_decode.axmodel";
 

    attr.attr_talker_model = attr_talker;

    if (!lLaMa.Init(attr))
    {
        return -1;
    }

    std::string path = "2.mp4";
    lLaMa.Run(path);

    lLaMa.Deinit();

    return 0;
}