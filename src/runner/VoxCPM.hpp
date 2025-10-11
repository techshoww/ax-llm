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
#include "SimpleLayer.hpp"
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
    SimpleLayer fsq_layer;
    SimpleLayer enc_to_lm_proj;
    SimpleLayer lm_to_dit_proj;
    SimpleLayer res_to_dit_proj;
    SimpleLayer stop_predictor;

    VoxCPMConfig config;

public:
    bool Init(VoxCPMConfig &config )
    {
        ALOGI("VoxCPM init start");
        
        config = config;
        
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
        
        fsq_layer.Init(config.dir_axmodels+"/fsq_layer.axmodel", config.lm_config.hidden_size, config.lm_config.hidden_size);
        enc_to_lm_proj.Init(config.dir_axmodels+"/enc_to_lm_proj.axmodel", config.encoder_config.hidden_dim, config.lm_config.hidden_size);
        lm_to_dit_proj.Init(config.dir_axmodels+"/lm_to_dit_proj.axmodel", config.lm_config.hidden_size, config.dit_config.hidden_dim);
        res_to_dit_proj.Init(config.dir_axmodels+"/res_to_dit_proj.axmodel", config.lm_config.hidden_size, config.dit_config.hidden_dim)
        stop_predictor.Init(config.dir_axmodels+"/stop_predictor.axmodel", config.lm_config.hidden_size, 1)
    }

    void Deinit()
    {
        base_lm.Deinit();
        residual_lm.Deinit();
        feat_encoder.Deinit();
        feat_decoder.Deinit();
        fsq_layer.Deinit();
        enc_to_lm_proj.Deinit();
        lm_to_dit_proj.Deinit();
        res_to_dit_proj.Deinit();
        stop_predictor.Deinit();
    }

    int GenerateWithPromptCache()
    {

    }

    // streaming inference
    int Inference(std::vector<int> &text, std::vector<int> &text_mask, std::vector<float> &feat, std::vector<int> &feat_mask, 
                    int min_len=2, int max_len=2000, int inference_timesteps=10, float cfg_value=2.0,
                    )
    {
        int ret;
        std::vector<float> feat_embed;
        ret = feat_encoder.Foward(feat, feat_embed);
        if(ret!=0)
        {
            ALOGE("feat_encoder failed");
            return -1;
        }

        ret = enc_to_lm_proj.Forward(feat_embed, feat_embed);
        if(ret!=0)
        {
            ALOGE("enc_to_lm_proj.Forward failed");
            return -1;
        }

        std::vector<unsigned short> text_embed;
        base_lm.TextToken2Embeds(text, text_embed);
        
        int hidden_size = config.lm_config.hidden_size;
        std::vector<unsigned short> combined_embed(text_embed.size(),  bfloat16(0.0f).data);
        
        for(int i=0; i<text_mask.size(); i++)
        {
            int tm = text_mask[i];
            int fm = feat_mask[i];

            float tmp_fp32;
            unsigned int tmp_u32; 

            if(tm==0 && fm==0)
            {
                std::fill(combined_embed.begin() + i * hidden_size, combined_embed + (i + 1) * hidden_size, bfloat16(0.0f).data);
            }
            else if(tm!=0 && fm==0)
            {
                std::copy(text_embed.begin() + i * hidden_size, text_embed.begin() + (i + 1) * hidden_size, combined_embed.begin() + i * hidden_size);
            }
            else if(tm==0 && fm!=0)
            {
                // float32 to bfloat16
                for(int j=i*hidden_size; j<(i + 1) * hidden_size; j++)
                {
                    combined_embed[j] = bfloat16(feat_embed[j]).data;
                }
            }
            else // tm!=0 && fm!=0
            {
                
                for(int j=i*hidden_size; j<(i + 1) * hidden_size; j++)
                {
                    // bfloat16 to float32
                    unsigned int tmp_u32 = text_embed[j] << 16;
                    float tmp_fp32 = *reinterpret_cast<float *>(&tmp_u32);

                    // float32 to bfloat16
                    combined_embed[j] = bfloat16( tmp_fp32 + feat_embed[j]).data;
                }
            }
        }

        ret = base_lm.Forward(combined_embed, true);
        if(ret!=0)
        {
            ALOGE("base_lm.Forward failed");
            return -1;
        }

        int prefill_len = combined_embed.size() / hidden_size;
        
        std::vector<float> enc_outputs(combined_embed.size(), 0.0f);
        for(int i=0; i<combined_embed.size(); i++)
        {
            // bfloat16 to float32
            unsigned int tmp_u32 = combined_embed[i] << 16;
            enc_outputs[i] = *reinterpret_cast<float *>(&tmp_u32);
        }

        std::vector<float> fsq_outputs;
        ret = fsq_layer.Forward(enc_outputs, fsq_outputs);
        if(ret!=0)
        {
            ALOGE("fsq_layer.Forward failed");
            return -1;
        }

        for(int i=0; i<text_mask.size(); i++)
        {
            float tm = text_mask[i];
            float fm = feat_mask[i];

            for(int j=i*hidden_size; j<(i+1)*hidden_size; j++)
            {
                enc_outputs[j] = fsq_outputs[j] * fm + enc_outputs[j] * tm;
            }
        }

        std::vector<float> lm_hidden(enc_outputs.end() - hidden_size, enc_outputs.end());

        std::vector<unsigned short> io_res_lm(enc_outputs.size());
        for(int i=0; i<feat_mask.size(); i++)
        {
            float fm = feat_mask[i];
            for(int j=i*hidden_size; j<(i+1)*hidden_size; j++)
            {
                float tmp_fp32 = enc_outputs[j] + fm * feat_embed[j];
                // float32 to bfloat16
                io_res_lm[j] = bfloat16(tmp_fp32).data;
            }
        }

        ret = residual_lm.Forward(io_res_lm, true);
        if(ret!=0)
        {
            ALOGE("residual_lm.Forward failed");
            return -1;
        }


        std::vector<unsigned short> residual_hidden(io_res_lm.end() - hidden_size, io_res_lm.end());

        int position_id = prefill_len;

        std:vector<float> prefix_feat_cond(feat.end()-config.patch_size*config.feat_dim, feat.end());

        std::vector<float> pred_feat_seq;
        for(int i=0; i< max_len; i++)
        {
            std::vector<float> dit_hidden_1;
            ret = lm_to_dit_proj.Forward(lm_hidden, dit_hidden_1);
            
            std::vector<float> residual_hidden_fp32(residual_hidden.size());
            // bfloat16 to float32
            for(int j=0; j<residual_hidden.size(); j++)
            {
                unsigned int tmp_u32 = residual_hidden[j] << 16;
                residual_hidden_fp32[j] = *reinterpret_cast<float *>(&tmp_u32);
            }
            std::vector<float> dit_hidden_2;
            ret = res_to_dit_proj.Forward(residual_hidden_fp32, dit_hidden_2);

            for(int j=0; j<dit_hidden_1.size(); j++)
            {
                dit_hidden_1[j] += dit_hidden_2[j];
            }

            std::vector<float> out_decoder;
            std::vector<float> cond = transposeVector(prefix_feat_cond, config.patch_size, config.feat_dim);
            ret = feat_decoder.Forward(dit_hidden_1, cond, inference_timesteps, cfg_value, 1.0, true, out_decoder);            
            if(ret!=0)
            {
                ALOGE("feat_decoder.Forward failed");
                return -1;
            }

            std::vector<float> pred_feat = transposeVector(out_decoder, config.feat_dim, config.patch_size); // (64,2) to (2,64)
            std::vector<float> curr_embed;
            ret = feat_encoder.Foward(pred_feat, curr_embed);
            if(ret!=0)
            {
                ALOGE("feat_encoder.Forward failed");
                return -1;
            }
            
            ret = enc_to_lm_proj.Forward(curr_embed, curr_embed);
            
            pred_feat_seq.insert(pred_feat_seq.end(), pred_feat.begin(), pred_feat.end());            
            prefix_feat_cond = std::move(pred_feat);

            pred_feat_chunk = 

        }
    }
    
};