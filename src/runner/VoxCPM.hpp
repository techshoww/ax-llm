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
#include "utils/whisper.hpp"
#include "MiniCPM.hpp"
#include "LocEnc.hpp"
#include "UnifiedCFM.hpp"
#include "SimpleLayer.hpp"
#include "SimpleLayerONNX.hpp"
#include "AudioVAE.hpp"

using WavBuffer = std::deque<std::vector<float>>;
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
};

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

    std::string url_tokenizer;
    
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
    SimpleLayerONNX stop_predictor;

    VoxCPMConfig config;
    AudioVAE audio_vae;

    bool b_stop = false;

    std::vector<int> prompt_text_proken;
    std::vector<float> prompt_audio_feat;

public:
    bool Init(VoxCPMConfig &config,  const std::string &prompt_text, const std::string &prompt_wav_path )
    {
        ALOGI("VoxCPM init start");
        
        config = config;
        
        tokenizer = CreateTokenizer(tokenizer_type);
        if (!tokenizer->Init(config.url_tokenizer, false, false))
        {
            ALOGE("tokenizer.Init(%s, %d, %d) failed", config.url_tokenizer.c_str(), false, false);
            return false;
        }

        config.lm_config.template_filename_axmodel = config.dir_base_lm + "/" + "MiniCPMForCausalLM_p64_l%d_together.axmodel";
        config.lm_config.filename_post_axmodel = config.dir_base_lm + "/" + "MiniCPMForCausalLM_post.axmodel";
        config.lm_config.filename_tokens_embed = config.dir_base_lm + "/" + "model.embed_tokens.weight.bfloat16.bin"; 
        base_lm.Init(config.lm_config);

        LLMAttrType residual_lm_config = config.lm_config;
        residual_lm_config.template_filename_axmodel = config.dir_residual_lm + "/" + "MiniCPMForCausalLM_p64_l%d_together.axmodel";
        residual_lm_config.filename_post_axmodel = config.dir_residual_lm + "/" + "MiniCPMForCausalLM_post.axmodel";
        residual_lm_config.axmodel_num = config.residual_lm_num_layers;
        residual_lm.Init(residual_lm_config);

        LLMAttrType encoder_lm_config = config.lm_config;
        encoder_lm_config.template_filename_axmodel = config.dir_feat_encoder + "/" + "MiniCPMForCausalLM_p3_l%d_together.axmodel";
        encoder_lm_config.filename_post_axmodel = config.dir_feat_encoder + "/" + "MiniCPMForCausalLM_post.axmodel";
        encoder_lm_config.axmodel_num = config.encoder_config.num_layers;
        encoder_lm_config.hidden_size = config.encoder_config.hidden_dim;
        feat_encoder.Init(encoder_lm_config, config.dir_axmodels);

        LLMAttrType decoder_lm_config = config.lm_config;
        decoder_lm_config.template_filename_axmodel = config.dir_decoder_estimator + "/" + "MiniCPMForCausalLM_p5_l%d_together.axmodel";
        decoder_lm_config.filename_post_axmodel = config.dir_decoder_estimator + "/" + "MiniCPMForCausalLM_post.axmodel";
        decoder_lm_config.axmodel_num = config.dit_config.num_layers;
        decoder_lm_config.hidden_size = config.dit_config.hidden_dim;
        if(!feat_decoder.Init(config.feat_dim, config.dit_config.cfm_config, decoder_lm_config, config.dir_axmodels))
        {
            ALOGE("feat_decoder.Init failed");
            return false;
        }
        
        fsq_layer.Init(config.dir_axmodels+"/fsq_layer.axmodel", config.lm_config.hidden_size, config.lm_config.hidden_size);
        enc_to_lm_proj.Init(config.dir_axmodels+"/enc_to_lm_proj.axmodel", config.encoder_config.hidden_dim, config.lm_config.hidden_size);
        lm_to_dit_proj.Init(config.dir_axmodels+"/lm_to_dit_proj.axmodel", config.lm_config.hidden_size, config.dit_config.hidden_dim);
        if(!res_to_dit_proj.Init(config.dir_axmodels+"/res_to_dit_proj.axmodel", config.lm_config.hidden_size, config.dit_config.hidden_dim))
        {
            ALOGE("res_to_dit_proj.Init failed");
            return false;
        }
        stop_predictor.Init(config.dir_axmodels+"/stop_predictor.onnx", config.lm_config.hidden_size, 2);

        audio_vae.Init(config.dir_axmodels);

        
        if(!prompt_text.empty() && !prompt_wav_path.empty() )
        {
            int ret = BuildPromptCache(prompt_text_proken, prompt_audio_feat, prompt_text, prompt_wav_path);
            if(ret !=0)
            {
                ALOGE("BuildPromptCache failed");
                return false;
            }
        }   

        ALOGI("VoxCPM init finised");
        return true;
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
        audio_vae.Deinit();
    }

    void Stop()
    {
        b_stop = true;
    }

    int GenerateStreaming(WavBuffer& wav_buffer,
                        std::mutex& buffer_mutex,
                        std::condition_variable& buffer_cv,
                        std::atomic<bool>& finished,
                        const std::string &text,
                        float cfg_value=2.0, int inference_timesteps=10, int max_length=4096, 
                        bool normalize=false, bool denoise=false )
    {
        if(normalize)
        {
            ALOGE("not support normalize");
            return -1;
        }
        if(denoise)
        {
            ALOGE("not support denoise");
            return -1;
        }

        int ret;
        
        #ifdef DEBUG
        savetxt<int>(std::string("prompt_text_proken.txt"), prompt_text_proken, '\n');
        savetxt<float>(std::string("prompt_audio_feat.txt"), prompt_audio_feat, '\n');
        #endif
        
        ret = GenerateWithPromptCache(wav_buffer, buffer_mutex, buffer_cv, finished,
            text, prompt_text_proken, prompt_audio_feat, 2, max_length, inference_timesteps, cfg_value);
        if(ret !=0)
        {
            ALOGE("GenerateWithPromptCache failed");
            return -1;
        }
        
        return 0;
    }

    int BuildPromptCache(std::vector<int> &text_token, std::vector<float> &out_feat, const std::string &prompt_text, const std::string &prompt_wav_path)
    {
        ImageInfo img_info;
        img_info.img_prompt = false;
        text_token = tokenizer->Encode(prompt_text, img_info);        

        std::vector<float> audio = load_audio(prompt_wav_path, audio_vae._sample_rate);
        if(audio.empty())
        {
            ALOGE("load_audio %s failed", prompt_wav_path);
            return -1;
        }

        int patch_len = config.patch_size * audio_vae.chunk_size;
        int remainder = audio.size() % patch_len;
        if(remainder != 0)
        {
            int padding_size = patch_len - remainder;
            audio.insert(audio.end(), padding_size, 0.0f);
        }
        
        #ifdef DEBUG
        savetxt("audio.txt", audio, '\n');
        readtxt("../../VoxCPM/audio.txt", audio);
        #endif 

        std::vector<float> audio_feat;
        audio_vae.Encode(audio_feat, audio);
        
        #ifdef DEBUG
        savetxt("audio_feat.txt", audio_feat, '\n');
        #endif 
        
        int N = audio_feat.size() / audio_vae.latent_dim;
        int T = audio_feat.size() / (audio_vae.latent_dim * config.patch_size);
        const int output_size = (T - 1) * config.patch_size * audio_vae.latent_dim;
        out_feat.clear();
        out_feat.reserve(output_size);

        // 模拟 view + permute + 切片操作
        for (int b = 0; b < T - 1; ++b) {       // 切片后第一维: 0 到 T-2
            for (int c = 0; c < config.patch_size; ++c) { // 第二维: 0 到 1
                for (int a = 0; a < audio_vae.latent_dim; ++a) { // 第三维: 0 到 63
                // 计算在原始一维数组中的索引
                // 对应关系: output[b,c,a] = original[a,b,c]
                    int input_index = a * N + b * config.patch_size + c;
                    out_feat.push_back(audio_feat[input_index]);
                }
            }
        }
        return 0;

    }

    int GenerateWithPromptCache(WavBuffer& wav_buffer,
                                std::mutex& buffer_mutex,
                                std::condition_variable& buffer_cv,
                                std::atomic<bool>& finished,
                                const std::string &target_text, std::vector<int> &prompt_text_token, std::vector<float> &prompt_audio_feat,
                                int min_len=2, int max_len=2000, int inference_timesteps=10, float cfg_value=2.0)
    {
        ImageInfo img_info;
        img_info.img_prompt = false;
        std::vector<int> target_text_token = tokenizer->Encode(target_text, img_info);        
        int target_text_length = target_text_token.size();
        std::vector<int> text_token(prompt_text_token);
        text_token.insert(text_token.end(), target_text_token.begin(), target_text_token.end());
        text_token.push_back(audio_start_token);

        int audio_vae_latent_dim = 64;
        int audio_length = prompt_audio_feat.size()/(config.patch_size * audio_vae_latent_dim);
        int text_length = text_token.size();
        std::vector<int> text_pad_token(audio_length, 0);
        std::vector<float> audio_pad_feat(text_token.size() * config.patch_size * audio_vae_latent_dim, 0);

        text_token.insert(text_token.end(), text_pad_token.begin(), text_pad_token.end());
        std::vector<float> audio_feat(audio_pad_feat);
        audio_feat.insert(audio_feat.end(), prompt_audio_feat.begin(), prompt_audio_feat.end());

        std::vector<int> text_mask(text_length + audio_length);
        std::fill_n(text_mask.begin(), text_length, 1);
        std::fill_n(text_mask.begin() + text_length, audio_length, 0);

        std::vector<int> audio_mask(text_length + audio_length);
        std::fill_n(audio_mask.begin(), text_length, 0);
        std::fill_n(audio_mask.begin() + text_length, audio_length, 1);

        std::vector<std::vector<float>> pred_feat;
        int ret = Inference(wav_buffer, buffer_mutex, buffer_cv, finished,
            text_token, text_mask, audio_feat, audio_mask, min_len, max_len, inference_timesteps, cfg_value);
        if(ret!=0)
        {
            ALOGE("Inference failed");
            return -1;
        }

        return 0;
    }

    // streaming inference
    int Inference(WavBuffer& wav_buffer,
                    std::mutex& buffer_mutex,
                    std::condition_variable& buffer_cv,
                    std::atomic<bool>& finished,
                    std::vector<int> &text, std::vector<int> &text_mask, std::vector<float> &feat, std::vector<int> &feat_mask, 
                    int min_len=2, int max_len=2000, int inference_timesteps=10, float cfg_value=2.0)
    {
        int ret;

        #ifdef DEBUG
        savetxt("text_token.txt", text, '\n');
        savetxt("feat.txt", feat, '\n');
        #endif 

        std::vector<float> feat_embed;
        ret = feat_encoder.Forward(feat, feat_embed);
        if(ret!=0)
        {
            ALOGE("feat_encoder failed");
            finished = true;
            buffer_cv.notify_all();
            return -1;
        }
        
        #ifdef DEBUG
        savetxt("feat_embed1.txt", feat_embed, '\n');
        #endif 
        
        ret = enc_to_lm_proj.Forward(feat_embed, feat_embed);
        if(ret!=0)
        {
            ALOGE("enc_to_lm_proj.Forward failed");
            finished = true;
            buffer_cv.notify_all();
            return -1;
        }

        #ifdef DEBUG
        savetxt("feat_embed2.txt", feat_embed, '\n');
        #endif 

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
                std::fill(combined_embed.begin() + i * hidden_size, combined_embed.begin() + (i + 1) * hidden_size, bfloat16(0.0f).data);
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

        #ifdef DEBUG
        std::vector<float> combined_embed_fp32(combined_embed.size(), 0.0f);
        for(int i=0; i<combined_embed.size(); i++)
        {
            // bfloat16 to float32
            unsigned int tmp_u32 = combined_embed[i] << 16;
            combined_embed_fp32[i] = *reinterpret_cast<float *>(&tmp_u32);
        }
        savetxt("combined_embed.txt", combined_embed_fp32, '\n');
        #endif 

        ret = base_lm.Forward(combined_embed, true);
        if(ret!=0)
        {
            ALOGE("base_lm.Forward failed");
            finished = true;
            buffer_cv.notify_all();
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

        #ifdef DEBUG
        savetxt("enc_outputs.txt", enc_outputs, '\n');
        #endif 

        std::vector<float> fsq_outputs;
        ret = fsq_layer.Forward(enc_outputs, fsq_outputs);
        if(ret!=0)
        {
            ALOGE("fsq_layer.Forward failed");
            finished = true;
            buffer_cv.notify_all();
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

        #ifdef DEBUG
        savetxt("enc_outputs1.txt", enc_outputs, '\n');
        #endif 

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

        #ifdef DEBUG
        std::vector<float> io_res_lm_fp32(io_res_lm.size());
        for(int i=0; i<io_res_lm_fp32.size(); i++)
        {
            // bfloat16 to float32
            unsigned int tmp_u32 = io_res_lm[i] << 16;
            io_res_lm_fp32[i] = *reinterpret_cast<float *>(&tmp_u32);
        }
        savetxt("res_lm_input.txt", io_res_lm_fp32, '\n');
        #endif 

        ret = residual_lm.Forward(io_res_lm, true);
        if(ret!=0)
        {
            ALOGE("residual_lm.Forward failed");
            finished = true;
            buffer_cv.notify_all();
            return -1;
        }

        #ifdef DEBUG
        std::vector<float> out_res_lm_fp32(io_res_lm.size());
        for(int i=0; i<out_res_lm_fp32.size(); i++)
        {
            // bfloat16 to float32
            unsigned int tmp_u32 = io_res_lm[i] << 16;
            out_res_lm_fp32[i] = *reinterpret_cast<float *>(&tmp_u32);
        }
        savetxt("res_lm_output.txt", out_res_lm_fp32, '\n');
        #endif 

        std::vector<unsigned short> residual_hidden(io_res_lm.end() - hidden_size, io_res_lm.end());

        #ifdef DEBUG
        std::vector<float> residual_hidden_fp32(residual_hidden.size());
        for(int i=0; i<residual_hidden.size(); i++)
        {
            // bfloat16 to float32
            unsigned int tmp_u32 = residual_hidden[i] << 16;
            residual_hidden_fp32[i] = *reinterpret_cast<float *>(&tmp_u32);
        }
        savetxt("residual_hidden.txt", residual_hidden_fp32, '\n');
        #endif 

        int position_id = prefill_len;

        std::vector<float> prefix_feat_cond(feat.end()-config.patch_size*config.feat_dim, feat.end());

        std::vector<float> pred_feat_seq;
        pred_feat_seq.reserve( 4 * 3 * config.patch_size * config.feat_dim); // 4 可以换成成其他值
        
        // #ifdef DEBUG
        max_len = 100;
        // #endif 

        for(int i=0; i< max_len; i++)
        {
            std::vector<float> dit_hidden_1;
            ret = lm_to_dit_proj.Forward(lm_hidden, dit_hidden_1);

            #ifdef DEBUG
            savetxt(std::string("dit_hidden_1_")+std::to_string(i)+".txt", dit_hidden_1, '\n');
            #endif 

            std::vector<float> residual_hidden_fp32(residual_hidden.size());
            // bfloat16 to float32
            for(int j=0; j<residual_hidden.size(); j++)
            {
                unsigned int tmp_u32 = residual_hidden[j] << 16;
                residual_hidden_fp32[j] = *reinterpret_cast<float *>(&tmp_u32);
            }
            std::vector<float> dit_hidden_2;

            #ifdef DEBUG
            savetxt(std::string("residual_hidden_fp32_")+std::to_string(i)+".txt", residual_hidden_fp32, '\n');
            #endif 

            ret = res_to_dit_proj.Forward(residual_hidden_fp32, dit_hidden_2);

            #ifdef DEBUG
            savetxt(std::string("dit_hidden_2_")+std::to_string(i)+".txt", dit_hidden_2, '\n');
            #endif 

            for(int j=0; j<dit_hidden_1.size(); j++)
            {
                dit_hidden_1[j] += dit_hidden_2[j];
            }

            std::vector<float> out_decoder;
            std::vector<float> cond = transposeVector(prefix_feat_cond, config.patch_size, config.feat_dim);

            #ifdef DEBUG
            savetxt(std::string("cond_")+std::to_string(i)+".txt", cond, '\n');
            savetxt(std::string("dit_hidden_")+std::to_string(i)+".txt", dit_hidden_1, '\n');
            #endif 

            ret = feat_decoder.Forward(dit_hidden_1, cond, inference_timesteps, cfg_value, 1.0, true, out_decoder);            
            if(ret!=0)
            {
                ALOGE("feat_decoder.Forward failed");
                finished = true;
                buffer_cv.notify_all();
                return -1;
            }
            
            #ifdef DEBUG
            savetxt(std::string("pred_feat_before_transpose_")+std::to_string(i)+".txt", out_decoder, '\n');
            #endif 

            std::vector<float> pred_feat = transposeVector(out_decoder, config.feat_dim, config.patch_size); // (64,2) to (2,64)

            #ifdef DEBUG
            savetxt(std::string("pred_feat_")+std::to_string(i)+".txt", pred_feat, '\n');
            #endif 

            std::vector<float> curr_embed;
            ret = feat_encoder.Forward(pred_feat, curr_embed);
            if(ret!=0)
            {
                ALOGE("feat_encoder.Forward failed");
                finished = true;
                buffer_cv.notify_all();
                return -1;
            }
            
            #ifdef DEBUG
            savetxt(std::string("enc_to_lm_proj_input_")+std::to_string(i)+".txt", curr_embed, '\n');
            #endif 

            ret = enc_to_lm_proj.Forward(curr_embed, curr_embed);

            #ifdef DEBUG
            savetxt(std::string("enc_to_lm_proj_output_")+std::to_string(i)+".txt", curr_embed, '\n');
            #endif 

            if(pred_feat_seq.size() >= 4 * 3 * config.patch_size * config.feat_dim - config.patch_size * config.feat_dim)
            {
                pred_feat_seq.erase(pred_feat_seq.begin(), pred_feat_seq.end() - 2 * config.patch_size * config.feat_dim);  // 只保留最后 2 个
            } 

            pred_feat_seq.insert(pred_feat_seq.end(), pred_feat.begin(), pred_feat.end());          
             
            prefix_feat_cond = std::move(pred_feat);

            int offset = std::min( i+1, 3);
            std::vector<float> pred_feat_chunk(pred_feat_seq.end() - offset * config.patch_size * config.feat_dim, pred_feat_seq.end());

            #ifdef DEBUG
            savetxt(std::string("pred_feat_chunk_")+std::to_string(i)+".txt", pred_feat_chunk, '\n');
            #endif 

            std::vector<float> feat_pred = rearrangeVector(pred_feat_chunk, 1, offset, config.patch_size, config.feat_dim);
            
            #ifdef DEBUG
            savetxt(std::string("feat_pred_")+std::to_string(i)+".txt", feat_pred, '\n');
            #endif 

            int patch_len = config.patch_size * audio_vae.chunk_size;
           
            std::vector<float> decode_audio;
            audio_vae.Decode(decode_audio, feat_pred);

            if(decode_audio.size() > patch_len)
            {
                decode_audio.assign(decode_audio.end() - patch_len, decode_audio.end());
            }

            {
                std::lock_guard<std::mutex> lock(buffer_mutex);
                wav_buffer.push_back(decode_audio);
            }
            buffer_cv.notify_one();
            
            if(i==max_len-1)
            {
                finished = true;
                buffer_cv.notify_all();
            }

            #ifdef DEBUG
            savetxt(std::string("stop_input_")+std::to_string(i)+".txt", lm_hidden, '\n');
            #endif 

            std::vector<float> stop_flag;
            ret = stop_predictor.Forward(lm_hidden, stop_flag);
            if(ret!=0)
            {
                ALOGE("stop_predictor.Forward failed");
                finished = true;
                buffer_cv.notify_all();
                return -1;
            }
            
            bool stop = stop_flag[0]<stop_flag[1]? true:false;
            if(i > min_len && stop)
            {
                finished = true;
                buffer_cv.notify_all();
                break;
            }

            std::vector<unsigned short> curr_embed_bf16(curr_embed.size());
            for(int j=0; j<curr_embed.size(); j++)
            {
                // float32 to bfloat16
                curr_embed_bf16[j] = bfloat16(curr_embed[j]).data;
            }

            ret = base_lm.ForwardStep(curr_embed_bf16, position_id);
            if(ret!=0)
            {
                ALOGE("base_lm.ForwardStep failed");
                finished = true;
                buffer_cv.notify_all();
                return -1;
            }

            // bfloat16 to float32
            for(int j=0; j<curr_embed_bf16.size(); j++)
            {
                unsigned int tmp_u32 = curr_embed_bf16[j] << 16;
                lm_hidden[j] = *reinterpret_cast<float *>(&tmp_u32);
            }

            #ifdef DEBUG
            savetxt(std::string("fsq_layer_input_")+std::to_string(i)+".txt", lm_hidden, '\n');
            #endif 

            fsq_layer.Forward(lm_hidden, lm_hidden);
            
            #ifdef DEBUG
            savetxt(std::string("fsq_layer_output_")+std::to_string(i)+".txt", lm_hidden, '\n');
            #endif 
            
            std::vector<unsigned short> io_res_lm(lm_hidden.size());
            // float32  to bfloat16
            for(int j=0; j<lm_hidden.size(); j++)
            {
                io_res_lm[j] = bfloat16(lm_hidden[j] + curr_embed[j]).data;
            }

            ret = residual_lm.ForwardStep(io_res_lm, position_id);
            if(ret!=0)
            {
                ALOGE("residual_lm.ForwardStep failed");
                finished = true;
                buffer_cv.notify_all();
                return -1;
            }

            residual_hidden = std::move(io_res_lm);

            position_id += 1;
        }

        finished = true;
        buffer_cv.notify_all();
        return 0;
    }
    
};