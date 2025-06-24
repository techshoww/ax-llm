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
#include "LLM.hpp"
#include "ThinkerTextModel.hpp"
#include "TalkerModel.hpp"
#include "Token2WavDit.hpp"

struct OmniAttr
{
    std::string path_audio_encoder;
    std::string path_visual_encoder;
    LLMAttrType attr_thinker_text_model;
    TalkerAttr attr_talker_model;
    std::string path_token2wav_dit;
    std::string path_token2wav_bigvgan;
};

class OmniModel
{
private:
    OmniAttr _attr;
    ax_runner_ax650 audio_encoder;
    ax_runner_ax650 visual_encoder;
    ThinkerTextModel thinker_text_model;
    TalkerModel talker_model;
    Token2WavDit token2wav_dit;
    ax_runner_ax650 token2wav_bigvgan;

public:
    bool Init(OmniAttr attr)
    {
        _attr = attr;

        if(!audio_encoder.Init(attr.path_audio_encoder)){
            ALOGE("init model (%s) failed", attr.path_audio_encoder.c_str());
            return false;
        }

        if(!visual_encoder.Init(attr.path_visual_encoder)){
            ALOGE("init model (%s) failed", attr.path_visual_encoder.c_str());
            return false;
        }

        if(!thinker_text_model.Init(attr.attr_thinker_text_model)){
            ALOGE("init thinker_text_model failed");
            return false;
        }

        if(!talker_model.Init(attr.attr_talker_model)){
            ALOGE("init talker_model failed");
            return false;
        }

        if(!token2wav_dit.Init(attr.path_token2wav_dit)){
            ALOGE("init model (%s) failed", attr.path_token2wav_dit.c_str());
            return false;
        }

        if(token2wav_bigvgan.init(attr.path_token2wav_bigvgan.c_str(), false)!=0){
            ALOGE("init model (%s) failed", attr.path_token2wav_bigvgan.c_str());
            return false;
        }

        ALOGI("OmniModel init success");
        return true;   
    }

    OmniAttr *getAttr()
    {
        return &_attr;
    }

    void Deinit()
    {
        audio_encoder.Deinit();
        visual_encoder.Deinit();
        thinker_text_model.Deinit();
        talker_model.Deinit();
        token2wav_dit.Deinit();
        token2wav_bigvgan.release();
       
    }

    int ProcessVideo(std::string path)
    {

    }


};