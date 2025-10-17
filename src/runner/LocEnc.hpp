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
#include "BaseRunner.hpp"

class LocEnc
{
private:

    std::vector<unsigned short> special_tokens;
    std::shared_ptr<BaseRunner> in_proj;
    MiniCPM encoder;
    int patch_size = 2;
    int latent_dim = 64;
    int hidden_size = 1024;

public:
    bool Init(LLMAttrType &config, const std::string &dir_axmodel)
    {
        int ret ;
        hidden_size = config.hidden_size;

        std::vector<float> special_tokens_fp32;
        ret = readtxt(dir_axmodel+"/feat_encoder.special_token.txt", special_tokens_fp32);
        if(ret!=0)
        {
            ALOGE("load %s failed %d", (dir_axmodel+"/feat_encoder.special_token.txt").c_str(), ret);
            return false;
        }

        special_tokens.resize(special_tokens_fp32.size());
        for(int i=0; i<special_tokens_fp32.size(); i++)
        {
            special_tokens[i] = bfloat16(special_tokens_fp32[i]).data;
        }

        in_proj = CreateRunner(RT_OnnxRunner);
        if(in_proj == nullptr)
        {
            ALOGE("init in_proj failed");
            return false;
        }
        BaseConfig config_proj;
        config_proj.nthread = 2;
        config_proj.onnx_model = dir_axmodel+"/feat_encoder.in_proj.onnx";
        in_proj->load(config_proj);

        if(!encoder.Init(config))
        {
            ALOGE("init encoder failed");
            return false;
        }

        return true;
    }

    void Deinit()
    {
        std::vector<unsigned short>().swap(special_tokens);
        encoder.Deinit();
    }

    int Forward(std::vector<float> &x, std::vector<float> & out)
    {
        int ret;
        int T = x.size()/(patch_size*latent_dim);
        std::vector<float> out_proj(patch_size*hidden_size, 0);
        std::vector<unsigned short> io_encoder((1+patch_size)*hidden_size, 0);
        out.resize(T * 1 * hidden_size);

        for(int i=0; i<T; i++)
        {
            void * p_data = x.data() + i * patch_size * latent_dim;

            float * p_input = (float *)in_proj->getInputPtr(0);
            memcpy(p_input, p_data, patch_size * latent_dim * sizeof(float));
            
            in_proj->inference();

            memcpy(out_proj.data(), (void *)in_proj->getOutputPtr(0), out_proj.size() * sizeof(float));
            
            std::copy(special_tokens.begin(), special_tokens.end(), io_encoder.begin());
            // float32 to bfloat16
            for(int j=0; j<patch_size*hidden_size; j++)
            {
                io_encoder[hidden_size + j] = bfloat16(out_proj[j]).data;
            }

            #ifdef DEBUG
            if(T>1 && i==0)
            {
                std::vector<float> in_encoder_fp32((1+patch_size)*hidden_size, 0);
                for(int j=0; j<(1+patch_size)*hidden_size; j++)
                {
                    unsigned int tmp = io_encoder[j] << 16;
                    in_encoder_fp32[j] = *reinterpret_cast<float *>(&tmp);
                }
                savetxt<float>("io_encoder_T0.txt", in_encoder_fp32, '\n');
            }
            #endif

            std::vector<unsigned short> out_encoder;
            ret = encoder.Forward(io_encoder, false);
            if(ret!=0)
            {
                ALOGE("encoder Forward failed");
                return -1;
            }

            #ifdef DEBUG
            if(T>1 && i==0)
            {
                std::vector<float> out_encoder_fp32((1+patch_size)*hidden_size, 0);
                for(int j=0; j<(1+patch_size)*hidden_size; j++)
                {
                    unsigned int tmp = io_encoder[j] << 16;
                    out_encoder_fp32[j] = *reinterpret_cast<float *>(&tmp);
                }
                savetxt<float>("io_encoder_T0_enc.txt", out_encoder_fp32, '\n');
            }
            #endif 
            
            // bfloat16 to float32
            for(int j=0; j<hidden_size; j++)
            {
                unsigned int tmp = io_encoder[j] << 16;
                out[i * hidden_size + j] = *reinterpret_cast<float *>(&tmp);
            }

        }

        return 0;
    }
};