#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "bfloat16.hpp"
#include "Tokenizer/Tokenizer.hpp"
#include "LLMEmbedSelector.hpp"
#include "ax_model_runner/ax_model_runner_ax650.hpp"
#include "utils/utils.hpp"
#include "ax_cmm_utils.hpp"
#include "cqdm.h"
#include "timer.hpp"
#include "opencv2/opencv.hpp"
#include "ax_sys_api.h"

class Token2Wav
{
private:
    ax_runner_ax650 flow_encoder_28;
    ax_runner_ax650 flow_encoder_53;
    ax_runner_ax650 flow_encoder_78;
    ax_runner_ax650 flow_encoder_50_final;

    ax_runner_ax650 flow_estimator_200;
    ax_runner_ax650 flow_estimator_250;
    ax_runner_ax650 flow_estimator_300;

    ax_runner_ax650 hift_50_first;
    ax_runner_ax650 hift_58;

    std::vector<float> rand_noise(0, 80*1*300);
    std::vector<float> t_span = {0.0000, 0.0123, 0.0489, 0.1090, 0.1910, 0.2929, 0.4122, 0.5460, 0.6910,0.8436, 1.0000};
    float inference_cfg_rate = 0.7;

    LLaMaEmbedSelector flow_embed_selector;
    int flow_embed_num = 6561;
    int flow_embed_size = 512;

    int init_noise(std::string model_dir)
    {
        return 0;
    }

public:
    bool Init(std::string model_dir)
    {
        int ret;

        ret = init_noise(model_dir);
        if(ret != 0){
            ALOGE("init rand noise(%s) failed", (model_dir+"/rand_noise.txt".c_str()));
            return false;
        }

        if (!flow_embed_selector.Init((model_dir+"/flow.input_embedding.bfloat16.bin").c_str(), flow_embed_num, flow_embed_size, false))
        {
            ALOGE("flow_embed_selector.Init(%s, %d, %d) failed", (model_dir+"/flow.input_embedding.bfloat16.bin").c_str(),flow_embed_num, flow_embed_size);
            return false;
        }

        ret = flow_encoder_28.init((mode_dir+"/flow_encoder_28.axmodel").c_str(), false);
        if (ret != 0)
        {
            ALOGE("init axmodel(%s) failed", mode_dir+"/flow_encoder_28.axmodel").c_str());
            return false;
        }

        ret = flow_encoder_53.init((mode_dir+"/flow_encoder_53.axmodel").c_str(), false);
        if (ret != 0)
        {
            ALOGE("init axmodel(%s) failed", mode_dir+"/flow_encoder_53.axmodel").c_str());
            return false;
        }

        ret = flow_encoder_78.init((mode_dir+"/flow_encoder_78.axmodel").c_str(), false);
        if (ret != 0)
        {
            ALOGE("init axmodel(%s) failed", mode_dir+"/flow_encoder_78.axmodel").c_str());
            return false;
        }

        ret = flow_encoder_50_final.init((mode_dir+"/flow_encoder_50_final.axmodel").c_str(), false);
        if (ret != 0)
        {
            ALOGE("init axmodel(%s) failed", mode_dir+"/flow_encoder_50_final.axmodel").c_str());
            return false;
        }

        ret = flow_estimator_200.init((mode_dir+"/flow_estimator_200.axmodel").c_str(), false);
        if (ret != 0)
        {
            ALOGE("init axmodel(%s) failed", mode_dir+"/flow_estimator_200.axmodel").c_str());
            return false;
        }

        ret = flow_estimator_250.init((mode_dir+"/flow_estimator_250.axmodel").c_str(), false);
        if (ret != 0)
        {
            ALOGE("init axmodel(%s) failed", mode_dir+"/flow_estimator_250.axmodel").c_str());
            return false;
        }

        ret = flow_estimator_300.init((mode_dir+"/flow_estimator_300.axmodel").c_str(), false);
        if (ret != 0)
        {
            ALOGE("init axmodel(%s) failed", mode_dir+"/flow_estimator_300.axmodel").c_str());
            return false;
        }

        ret = hift_50_first.init((mode_dir+"/hift_50_first.axmodel").c_str(), false);
        if (ret != 0)
        {
            ALOGE("init axmodel(%s) failed", mode_dir+"/hift_50_first.axmodel").c_str());
            return false;
        }

        ret = hift_58.init((mode_dir+"/hift_58.axmodel").c_str(), false);
        if (ret != 0)
        {
            ALOGE("init axmodel(%s) failed", mode_dir+"/hift_58.axmodel").c_str());
            return false;
        }



    }

    int infer_flow_encoder(
        std::<vector> & token_embeds, std::vector<float> & prompt_feat, std::vector<float> & spk_embeds, int token_len, bool finalize,
        std::<vector> & mu, std::vector<float> & spks, std::vector<float> & cond
        )
    {
        ax_runner_ax650 & model;
        if(!finalize)
        {
            if(token_len == 28)
            {
                model = flow_encoder_28;
            }else if(token_len == 53)
            {
                model = flow_encoder_53;
            }else if(token_len == 78)
            {
                model = flow_encoder_78;
            }else{
                return -1;
            }
        }else if(token_len == 50){
            model = flow_encoder_50_final;
        }else{
            return -1;
        }

        void * p = model.get_input("token_embedding").pVirAddr;
        memcpy(p, token_embeds.data(), token_embeds.size() * sizeof(float));
        p = model.get_input("prompt_feat").pVirAddr;
        memcpy(p, prompt_feat.data(), prompt_feat.size() * sizeof(float));
        p = model.get_input("embedding").pVirAddr;
        memcpy(p, spk_embeds.data(), spk_embeds.size() * sizeof(float));

        model.inference();

        
        auto &output_mu = model.get_output("mu");
        if(mu.empty())
        {
            mu.resize(output_mu.nSize / sizeof(float));
        }
        memcpy(mu.data(), output_mu.pVirAddr, output_mu.nSize);

        auto &output_spks = model.get_output("spks");
        if(spks.empty())
        {
            spks.resize(output_spks.nSize / sizeof(float));
        }
        memcpy(spks.data(), output_spks.pVirAddr, output_spks.nSize);

        audo &output_cond = model.get_output("cond");
        if(cond.empty())
        {
            cond.resize(output_cond.nSize / sizeof(float));
        }
        memcpy(cond.data(), output_cond.pVirAddr, output_cond.nSize);

        return 0;
    }

    int infer_flow_estimator(
        std::vector<float> & x, std::vector<float> & mask, std::vector<float> & t,
        std::vector<float> & mu, std::vector<float> & spks, std::vector<float> & cond, 
        std::vector<float> & dphi_dt
        )
    {
        ax_runner_ax650 & model;
        int len = x.size()/(2*80);
        if(len == 200){
            model = flow_estimator_200;
        }else if(len == 250){
            model = flow_estimator_250;
        }else if(len == 300){
            model = flow_estimator_300;
        }else{
            return -1;
        }
        
        void * p = model.get_input("x").pVirAddr;
        memcpy(p, x.data(), x.size() * sizeof(float));
        p = model.get_input("mask").pVirAddr;
        memcpy(p, mask.data(), mask.size() * sizeof(float));
        p = model.get_input("t").pVirAddr;
        memcpy(p, t.data(), t.size() * sizeof(float));
        p = model.get_input("mu").pVirAddr;
        memcpy(p, mu.data(), mu.size() * sizeof(float));
        p = model.get_input("spks").pVirAddr;
        memcpy(p, spks.data(), spks.size() * sizeof(float));
        p = model.get_input("cond").pVirAddr;
        memcpy(p, cond.data(), cond.size() * sizeof(float));

        model.inference();

        auto &output_dphi_dt = model.get_output("dphi_dt");
        if(dphi_dt.empty())
        {
            dphi_dt.resize(output_dphi_dt.nSize / sizeof(float));
        }
        memcpy(dphi_dt.data(), output_dphi_dt.pVirAddr, output_dphi_dt.nSize);
        
        return 0;
    }

    int infer_hift(std::vector<float> &mel, std::vector<float> &cache_source, 
                    std::vector<float> & tts_speech, std::vector<float> & tts_source)
    {
        ax_runner_ax650 & model
        int len = mel.size()/(80);
        if(len == 50 && cache_source.empty())
        { 
            model = hift_50_first;
        }else if(len == 58 && !cache_source.empty())
        {
            model = hift_58;
        }else
        {
            return -1;
        }

        void * p = model.get_input("mel").pVirAddr;
        memcpy(p, mel.data(), mel.size() * sizeof(float));
        if(!cache_source.empty())
        {
            p = model.get_input("cache_source").pVirAddr;
            memcpy(p, cache_source.data(), cache_source.size() * sizeof(float));
        }

        model.inference();

        auto &output_speech = model.get_output("audio");
        if(tts_speech.empty())
        {
            tts_speech.resize(output_speech.nSize / sizeof(float));
        }
        memcpy(tts_speech.data(), output_speech.pVirAddr, output_speech.nSize);

        auto &output_source = model.get_output("x");
        if(tts_source.empty())
        {
            tts_source.resize(output_source.nSize / sizeof(float));
        }
        memcpy(tts_source.data(), output_source.pVirAddr, output_source.nSize);
        
        return 0;
    }

    int infer_flow_decoder_solve_euler(
        std::vector<float> & x,  std::vector<float> & mu, std::vector<float> & spks, std::vector<float> & cond, std::vector<float> & mask,
        std::vector<float> & mel
    )
    {
        int len = mu.size()/80;

        float t = t_span[0];
        float dt = t_span[1] - t_span[0];

        std::vector<float> x_in(0, 2*80*len);
        std::vector<float> mask_in(0, 2*1*len);
        std::vector<float> mu_in(0, 2*80*len);
        std::vector<float> t_in(0, 2);
        std::vector<float> spks_in(0, 2*80);
        std::vector<float> cond_in(0, 2*80*len);
        for(int step=1; step<t_span.size(); step++)
        {   
            memcpy(x_in.data(), x.data(), x.size() * sizeof(float));
            memcpy(x_in.data()+x.size(),  x.data(), x.size() * sizeof(float));

            memcpy(mask_in.data(), mask.data(), mask.size() * sizeof(float));
            memcpy(mask_in.data()+mask.size(), mask.data(), mask.size() * sizeof(float));

            memcpy(mu_in.data(), mu.data(), mu.size() * sizeof(float));

            t_in[0] = t;
            t_in[1] = t;

            memcpy(spks_in.data(), spks.data(), spks.size() * sizeof(float));
            memcpy(cond_in.data(), cond.data(), cond.size() * sizeof(float));

            std::vector<float> dphi_dt;
            ret = infer_flow_estimator(x_in, mask_in, t_in, mu_in, spks_in, cond_in, dphi_dt);
            if(ret != 0)
            {
                return ret;
            }

            for(int i=0; i<80*len; i++)
            {

                dphi_dt[i] = (1.0 + inference_cfg_rate) * dphi_dt[i] - inference_cfg_rate * dphi_dt[80 * len + i];
                x[i] = x[i] + dt * dphi_dt[i];
            }
            
            t = t + dt;

            if(step < t_span.size()-1)
            {
                dt = t_span[step+1] - t;
            }
            else{
                if(mel.empty() || mel.size()!=x.size())
                {
                    mel.resize(x.size());
                }
                memcpy(mel.data(), x.data(), x.size() * sizeof(float));
            }

        }


    }

    int infer_flow_decoder(
        std::vector<float> & mu, std::vector<float> & spks, std::vector<float> & cond, std::vector<float> & mask,
        std::vector<float> & mel
    )
    {
        std::vector<float> z;
        z.insert(z.end(), rand_noise.begin(), rand_noise.begin() + mu.size());

        int ret = infer_flow_decoder_solve_euler(z, mu, spks, cond, mask, mel);
        return ret;
    }

    int infer_flow(
        std::vector<float> & token_embeds, std::vector<float> & prompt_feat, std::vector<float> & spk_embeds, int token_len, bool finalize,
        std::vector<float> & mel
    )
    {
        int ret; 
        int len;
        std::vector<float> mu;
        std::vector<float> spks;
        std::vector<float> cond;
        
        ret = infer_flow_encoder(token_embeds, prompt_feat, spk_embeds, token_len, finalize, mu, spks, cond);
        if(ret != 0)
        {
            return ret;
        }

        len = mu.size()/80;
        std::vector<float> mask(len, 1.0);

        std::vector<float> all_mel;
        
        ret = infer_flow_decoder(mu, spks, cond, mask, all_mel);
        if(ret != 0)
        {
            return ret;
        }

        int len_mel1 = prompt_feat.size()/80;
        int len_mel2 = all_mel.size()/80 - len_mel1;

        if(mel.empty() || mel.size()!=len_mel2)
        {
            mel.resize(len_mel2);
        }
        memcpy(mel.data(), all_mel.data() + len_mel1 * 80, len_mel2 * sizeof(float));

        return 0;
    }

    int token2wav(std::vector<int> & text_speech_token, std::vector<float> & prompt_speech_embeds, std::vector<float> * prompt_feat,  
                std::vector<float> & spk_embeds, int token_offset, bool finalize,
                std::vector<float> & tts_speech, std::vector<float> & tts_source)
    {
        std::vector<float> speech_embeds;
        std::vector<unsigned short> speech_embeds_one;

        speech_embeds.insert(speech_embeds.end(), prompt_speech_embeds.begin(), prompt_speech_embeds.end());

        for (size_t i = 0; i < text_speech_token.size(); i++)
        {
            flow_embed_selector.getByIndex(text_speech_token[i], speech_embeds_one.data() +  flow_embed_size);

            for (int j = 0; j < flow_embed_size; j++)
                {
                    unsigned int proc = speech_embeds_one[i] << 16;
                    speech_embeds[prompt_speech_embeds.size() + i * flow_embed_size + j] = *reinterpret_cast<float *>(&proc);
                }
        }

        std::vector<float> mel;
    
        int ret = infer_flow(speech_embeds, prompt_feat, spk_embeds, text_speech_token.size(), finalize, mel);   

    }
}