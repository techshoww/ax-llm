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
#include "AudioEncoder.hpp"
#include "VisualEncoder.hpp"
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
    AudioEncoder audio_encoder;
    VisualEncoder visual_encoder;
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
        ALOGI("init thinker_text_model");
        if(!thinker_text_model.Init(attr.attr_thinker_text_model)){
            ALOGE("init thinker_text_model failed");
            return false;
        }
        ALOGI("init talker model");
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

    void Stop()
    {
        thinker_text_model.Stop();
        talker_model.Stop();
    }

    int ProcessVideo(std::string path, std::vector<float>& audio, std::vector<cv::Mat>& imgs)
    {
        const std::string filename = "../python/input_features.txt";  // 替换为你的文件名
    
        // 1. 打开文本文件
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "错误：无法打开文件 " << filename << std::endl;
            return 1;
        }

        std::string line;

        // 2. 逐行读取文件
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            float value;
            
            // 3. 按空格/逗号分割每行数据
            while (iss >> value) {
                audio.push_back(value);
                
                // 跳过分隔符（支持空格、逗号、制表符等）
                if (iss.peek() == ',' || iss.peek() == ' ' || iss.peek() == '\t') {
                    iss.ignore();
                }
            }
        }
        file.close();

        return 0;
    }
    

    int Run(std::string path)
    {
        int ret;
        std::vector<float> audio;
        std::vector<cv::Mat> imgs;
        ret = ProcessVideo(path, audio, imgs);
        if(ret!=0){
            ALOGE("ProcessVideo failed");
            return -1;
        }

        std::vector<unsigned short> embed_audio;
        std::vector<unsigned short> embed_imgs;
        ret = audio_encoder.Encode(audio, embed_audio);
        if(ret!=0){
            ALOGE("audio encoder failed");
            return -1;
        }

        Config config;    
        config.vision_config.temporal_patch_size = 2;
        config.vision_config.tokens_per_second ;
        config.vision_config.spatial_merge_size;
        config.vision_config.patch_size = 14;
        config.vision_config.width = 308;
        config.vision_config.height = 308;
        config.vision_config.fps = 1;

        config.image_token_id ;
        config.video_token_id ;
        config.vision_start_token_id;
        ret = visual_encoder.Encode(imgs, embed_imgs, config);
        if(ret!=0){
            ALOGE("visual encoder failed");
            return -1;
        }


        return 0;

    }

};