#pragma once
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>
#include <atomic>
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
#include "utils/sampling.hpp"
#include "utils/utils.hpp"
#include "MiniCPM.hpp"
#include "LocEnc.hpp"
#include "UnifiedCFM.hpp"
#include "Tokenizer/Tokenizer.hpp"


struct VoxCPMEncoderConfig
{
    int hidden_dim = 1024;
    int num_layers = 4;
};


struct VoxCPMDitConfig
{
    int hidden_dim = 1024;
    int num_layers = 4;
    CfmConfig cfm_config;
}

struct VoxCPMConfig
{
    LLMAttrType lm_config;
    int feat_dim = 64;
    int patch_size = 2;
    int residual_lm_num_layers = 6;
    int scalar_quantization_latent_dim = 256;
    int scalar_quantization_scale = 9;

    VoxCPMEncoderConfig encoder_config;
    VoxCPMDitConfig dit_config;

    int max_length = 4096;

    std::string dir_base_lm;
    std::string dir_residual_lm;
    std::string dir_feat_encoder;
    std::string dir_decoder_estimator;
    std::string dir_axmodels;
    
};

class VoxCPM
{
private:
    int audio_start_token = 101;
    int audio_end_token = 102;

    std::shared_ptr<BaseTokenizer> tokenizer;
    TokenizerType tokenizer_type = TKT_HTTP;
    
    MiniCPM base_lm;
    MiniCPM residual_lm;
    LocEnc feat_encoder;
    UnifiedCFM feat_decoder;

public:
    bool Init(VoxCPMConfig &config )
    {
        ALOGI("VoxCPM init start");
       
        
        tokenizer = CreateTokenizer(tokenizer_type);
        if (!tokenizer->Init(attr.filename_tokenizer_model, attr.b_bos, attr.b_eos))
        {
            ALOGE("tokenizer.Init(%s, %d, %d) failed", attr.filename_tokenizer_model.c_str(), attr.b_bos, attr.b_eos);
            return false;
        }

        config.lm_config.template_filename_axmodel = config.dir_base_lm + "/" + "MiniCPMForCausalLM_p64_l%d_together.axmodel";
        config.lm_config.filename_post_axmodel = config.dir_base_lm + "/" + "MiniCPMForCausalLM_post.axmodel";
        base_lm.Init(config.lm_config);

        LLMAttrType residual_lm_config = config.lm_config;
        residual_lm_config.template_filename_axmodel = config.dir_residual_lm + "/" + "MiniCPMForCausalLM_p64_l%d_together.axmodel";
        residual_lm_config.filename_post_axmodel = config.dir_residual_lm + "/" + "MiniCPMForCausalLM_post.axmodel";
        residual_lm_config.axmodel_num = config.residual_lm_num_layers;

        residual_lm.Init(residual_lm_config);

        LLMAttrType encoder_lm_config = config.lm_config;
        encoder_lm_config.template_filename_axmodel = config.dir_feat_encoder + "/" + "MiniCPMForCausalLM_p64_l%d_together.axmodel";
        encoder_lm_config.filename_post_axmodel = config.dir_feat_encoder + "/" + "MiniCPMForCausalLM_post.axmodel";
        encoder_lm_config.axmodel_num = config.encoder_config.num_layers;
        encoder_lm_config.hidden_size = config.encoder_config.hidden_dim;
        feat_encoder.Init(encoder_lm_config, config.dir_axmodels);


        LLMAttrType decoder_lm_config = config.lm_config;
        decoder_lm_config.template_filename_axmodel = config.dir_decoder_estimator + "/" + "MiniCPMForCausalLM_p64_l%d_together.axmodel";
        decoder_lm_config.filename_post_axmodel = config.dir_decoder_estimator + "/" + "MiniCPMForCausalLM_post.axmodel";
        decoder_lm_config.axmodel_num = config.dit_config.num_layers;
        decoder_lm_config.hidden_size = config.dit_config.hidden_dim;
        feat_decoder.Init(config.feat_dim, config.dit_config.cfm_config, decoder_lm_config, config.dir_axmodels);
        

    }
    
};