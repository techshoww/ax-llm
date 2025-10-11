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
    bool Init(LLMAttrType &config, std::string &dir_axmodel)
    {
        int ret ;
        hidden_size = config.hidden_size;

        ret = readtxt(dir_axmodel+"/feat_encoder.special_token.txt", special_tokens);
        if(ret<=0)
        {
            ALOGE("load %s failed", (dir_axmodel+"/feat_encoder.special_token.txt").c_str());
            return false;
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

    int Foward(std::vector<float> &x, std::vector<float> & out)
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
            
            std::copy(special_tokens.begin(), special_tokens,end(), io_encoder.begin());
            // float32 to bfloat16
            for(int j=0; j<patch_size*hidden_size; j++)
            {
                io_encoder[hidden_size + j] = bfloat16(out_proj[j]).data;
            }

            ret = encoder.Forward(io_encoder, false);
            if(!ret)
            {
                ALOGE("encoder Forward failed");
                return -1;
            }

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