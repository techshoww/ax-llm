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
#include "bfloat16.hpp"
struct OmniAttr
{
    std::string path_audio_encoder;
    std::string path_visual_encoder;
    LLMAttrType attr_thinker_text_model;
    TalkerAttr attr_talker_model;
    std::string path_token2wav_dit;
    std::string path_token2wav_bigvgan;
    int max_len_talker_generate_codes = 600;
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

        ReadImages("../imgs", imgs);

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
        ret = audio_encoder.Run(audio, embed_audio);
        if(ret!=0){
            ALOGE("audio encoder failed");
            return -1;
        }

        Config config;    
        config.vision_config.temporal_patch_size = 2;
        config.vision_config.tokens_per_second = 25;
        config.vision_config.spatial_merge_size = 2;
        config.vision_config.patch_size = 14;
        config.vision_config.width = 308;
        config.vision_config.height = 308;
        config.vision_config.fps = 1;

        config.image_token_id = 151655;
        config.video_token_id = 151656;
        config.vision_start_token_id = 151652;
        config.audio_token_id = 151646;
        config.audio_start_token_id = 151647;

        config.seconds_per_chunk = 2;
        config.position_id_per_seconds = 25;

        ret = visual_encoder.Run(imgs, embed_imgs, config);
        if(ret!=0){
            ALOGE("visual encoder failed");
            return -1;
        }
        ALOGI("embed_imgs size:%d",embed_imgs.size());


        std::vector<std::vector<unsigned short>> thinker_token_embeds;
        std::vector<std::vector<unsigned short>> thinker_hidden_states;
        std::vector<int> thinker_generate_ids;
        std::vector<int> input_ids;
        readtxt("../python/input_ids.txt", input_ids);
        ALOGI("input_ids size:%d",input_ids.size());
        int thinker_hidden_dim = thinker_text_model._attr.tokens_embed_size;
        ALOGI("thinker_hidden_dim:%d",thinker_hidden_dim);
        auto text = thinker_text_model.Run(input_ids, embed_audio, embed_imgs, config, thinker_token_embeds, thinker_hidden_states, thinker_generate_ids);
        ALOGI("text model output:%s", text.c_str());
        ALOGI("thinker_hidden_states.size:%d, thinker_hidden_states[0].size:%d, thinker_hidden_states[1].size:%d",
                thinker_hidden_states.size(), thinker_hidden_states[0].size(), thinker_hidden_states[1].size());
        std::vector<unsigned short> thinker_reply_part( (thinker_hidden_states.size()-1+2)*thinker_hidden_dim, 0);

        
        unsigned int talker_text_bos_token = 151872;
        std::vector<int> talker_input_text_ids;
        talker_input_text_ids.insert(talker_input_text_ids.end(), input_ids.begin(), input_ids.end());
        talker_input_text_ids.push_back(talker_text_bos_token);
        talker_input_text_ids.push_back(thinker_generate_ids[0]);

        for(int i=1; i<thinker_hidden_states.size(); i++){
            for(int j=0; j< thinker_hidden_states[i].size(); j++){
                float a = float(thinker_hidden_states[i][j]);
                float b = float(thinker_token_embeds[i][j]);
                thinker_reply_part[ (i-1)*thinker_hidden_states[i].size() + j ] = bfloat16(a+b).data;
            }
        }


        std::vector<unsigned short> talker_inputs_embeds( thinker_hidden_states[0].size()+ thinker_hidden_states[1].size()*2, 0 );
        for(int i=0; i<thinker_hidden_states[0].size(); i++){
            float a = float(thinker_hidden_states[0][i]);
            float b = float(thinker_token_embeds[0][i]);
        }


        thinker_text_model.embed_selector.getByIndex(talker_text_bos_token, talker_inputs_embeds.data()+thinker_hidden_states[0].size());
        memcpy(talker_inputs_embeds.data()+thinker_hidden_states[0].size()+ thinker_hidden_states[1].size(), thinker_reply_part.data(), thinker_hidden_states[1].size()*sizeof(unsigned short));


        thinker_text_model.embed_selector.getByIndex(talker_model._attr.text_eos_token, thinker_reply_part.data()+ thinker_reply_part.size()-2*thinker_hidden_dim);
        thinker_text_model.embed_selector.getByIndex(talker_model._attr.text_pad_token, thinker_reply_part.data()+ thinker_reply_part.size()-1*thinker_hidden_dim);


        std::vector<int> talker_generate_codes;
        ret = talker_model.Run(talker_inputs_embeds, talker_input_text_ids, thinker_reply_part, config, talker_generate_codes);

        ALOGI("talker_generate_codes size:%d",talker_generate_codes.size());

        timer t;
        t.start();

        talker_generate_codes.pop_back();
        int effictive_len = talker_generate_codes.size();
        if(effictive_len > _attr.max_len_talker_generate_codes){
            effictive_len = _attr.max_len_talker_generate_codes;
        }
        // 大于max_len_talker_generate_codes 截断了，还需要添加处理

        std::vector<int> code(_attr.max_len_talker_generate_codes, 0);
        memcpy(code.data(), talker_generate_codes.data(), effictive_len*sizeof(int));
        
        std::vector<float> mel_spectrogram;

        std::vector<float> cond;
        readtxt("cond.txt", cond);
        std::vector<float> ref_mel;
        readtxt("ref_mel.txt", ref_mel);
        token2wav_dit.sample(cond, ref_mel, code,  0.5, -1.0, mel_spectrogram);


        void *data = token2wav_bigvgan.get_input("apm_mel").pVirAddr;
        memcpy(data, mel_spectrogram.data(), mel_spectrogram.size()*sizeof(float));

        token2wav_bigvgan.inference();

        size_t size = token2wav_bigvgan.get_output(0).nSize / sizeof(float);
        ALOGI("token2wav_bigvgan.get_output size:%d",size);
        int s = effictive_len*480;
        if(s > size){
            s = size;
        }
        std::vector<float> wav(s, 0);

        
        AX_SYS_MinvalidateCache(token2wav_bigvgan.get_output(0).phyAddr, token2wav_bigvgan.get_output(0).pVirAddr, token2wav_bigvgan.get_output(0).nSize);

        float *output_data = (float *)token2wav_bigvgan.get_output(0).pVirAddr;

        memcpy(wav.data(), output_data, s*sizeof(float));

        savetxt("wav.txt", wav);

        ALOGI("token2wav time : %f ms, size : %d", t.cost(), out_embed.size());
        return 0;

    }

};