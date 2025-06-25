#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "bfloat16.hpp"
#include "Tokenizer/Tokenizer.hpp"
#include "LLMEmbedSelector.hpp"
#include "ax_model_runner/ax_model_runner_ax650.hpp"
#include "ax_cmm_utils.hpp"
#include "cqdm.h"
#include "timer.hpp"
#include "opencv2/opencv.hpp"
#include "ax_sys_api.h"
#include "LLMPostprocess.hpp"
#include "image_processor.hpp"
#include "mrope.hpp"

typedef void (*LLMRuningCallback)(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve);

static int FindMax(unsigned short *p, int n, float *val = 0)
    {
        float max_val = -MAXFLOAT;
        int max_index = 0;
        for (int i = 0; i < n; i++)
        {
            unsigned int proc = p[i] << 16;
            float tmp = *reinterpret_cast<float *>(&proc);
            if (tmp > max_val)
            {
                max_val = tmp;
                max_index = i;
            }
        }

        if (val)
            *val = max_val;
        return max_index;
    }


struct LLMAttrType
{
    std::string template_filename_axmodel;
    int axmodel_num;

    int prefill_token_num; // auto calc

    std::string filename_post_axmodel;

    TokenizerType tokenizer_type;
    std::string filename_tokenizer_model;
    bool b_bos = true, b_eos = false;
    std::string filename_tokens_embed;
    int tokens_embed_num ;
    int tokens_embed_size;

    int max_token_len; // auto calc

    int kv_cache_num; // auto calc
    int kv_cache_size; // auto calc

    bool b_use_mmap_load_embed = false;
    bool b_dynamic_load_axmodel_layer = false;

    bool b_use_mmap_load_layer = true;

    bool b_use_topk = false;
    std::string post_config_path = "post_config.json";

    // bool b_live_print = true;
    LLMRuningCallback runing_callback = nullptr;
    void *reserve = nullptr;
};
