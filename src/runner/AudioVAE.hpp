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
#include "ax_sys_api.h"
#include "utils/utils.hpp"
#include "BaseRunner.hpp"

class AudioVAE
{
private:
    
    std::shared_ptr<BaseRunner> encoder;
    std::shared_ptr<BaseRunner> decoder;

public:
    int _sample_rate=16000;
    int hop_length = 2*5*8*8;
    int chunk_size = 2*5*8*8;
    int latent_dim = 64;
    
    bool Init(std::string &dir_axmodel)
    {
        encoder = CreateRunner(RT_OnnxRunner);
        if(encoder == nullptr)
        {
            ALOGE("init encoder failed");
            return false;
        }
        BaseConfig config_encoder;
        config_encoder.nthread = 2;
        config_encoder.onnx_model = dir_axmodel+"/audio_vae.encoder.onnx";
        encoder->load(config_encoder);

        decoder = CreateRunner(RT_OnnxRunner);
        if(decoder == nullptr)
        {
            ALOGE("init decoder failed");
            return false;
        }
        BaseConfig config_decoder;
        config_decoder.nthread = 2;
        config_decoder.onnx_model = dir_axmodel+"/audio_vae.decoder.onnx";
        decoder->load(config_decoder);
    }

    void Deinit()
    {}

    int Encode(std::vector<float> &output, std::vector<float> audio_data, int sample_rate=16000)
    {
        if(_sample_rate!=sample_rate)
        {
            ALOGE("Just support sample_rate=%d",_sample_rate);
            return -1
        }

        int len = audio_data.size();
        float *p_input = (float *)encoder->getOutputPtr(0);
        memcpy(p_input, audio_data.data(), len * sizeof(float));

        encoder->inference();

        output.resize(1 * latent_dim * (len / chunk_size));
        memcpy(output.data(), (void *)encoder->getOutputPtr(0), 1 * latent_dim * (len / chunk_size) * sizeof(float));

        return 0;
    }

    int Decode(std::vector<float> &output, std::vector<float> &z)
    {
        int len = z.size() / latent_dim;
        float *p_input = (float *)decoder->getOutputPtr(0);
        memcpy(p_input, z.data(), len * latent_dim * sizeof(float));

        decoder->inference();

        output.resize(1 * 1 * (len * chunk_size));
        memcpy(output.data(), (void *)decoder->getOutputPtr(0), 1 * 1 * (len * chunk_size) * sizeof(float));

        return 0;
    }

};