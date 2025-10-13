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


struct LLMAttrType
{
    std::string template_filename_axmodel = "MiniCPMForCausalLM_p64_l%d_together.axmodel";
    std::string filename_post_axmodel = "MiniCPMForCausalLM_post.axmodel";
    int axmodel_num = 24;

    int prefill_token_num = 96; // auto calc
    int prefill_max_token_num = 512;
    std::vector<int> prefill_max_kv_cache_num_grp;
    int precompute_len = 0;
    int prefill_grpid = -1;

    TokenizerType tokenizer_type = TKT_HTTP;
    std::string url_tokenizer = "http://127.0.0.1:12345";
    bool b_bos = false, b_eos = false;
    std::string filename_tokens_embed = "model.embed_tokens.weight.bfloat16.bin"; 
    int tokens_embed_num = 73448;
    int tokens_embed_size = 1024;

    int max_token_len = 127; // auto calc

    int kv_cache_num = 1024; // auto calc
    int kv_cache_size = 128; // auto calc
    int hidden_size = 1024; 

    bool b_use_mmap_load_embed = false;

};

class MiniCPM
{
private:
    std::shared_ptr<BaseTokenizer> tokenizer;
    LLaMaEmbedSelector embed_selector;
    struct LLMLayer
    {
        ax_runner_ax650 layer;
        std::string filename;
        MMap layer_buffer;
        std::vector<char> layer_buffer_vec;
    };

    std::vector<LLMLayer> llama_layers;
    ax_runner_ax650 llama_post;

    // int prefill_grpid = 1;
    int decode_grpid = 0;
    bool b_stop = false;
    int min_len = -1;
    int max_len = -1;

public:
    LLMAttrType _attr;

    bool Init(LLMAttrType attr)
    {
        ALOGI("LLM init start");
        t_cqdm cqdm = create_cqdm(attr.axmodel_num + 3, 32);
        this->_attr = attr;
        tokenizer = CreateTokenizer(attr.tokenizer_type);
        if (!tokenizer->Init(attr.url_tokenizer, attr.b_bos, attr.b_eos))
        {
            ALOGE("tokenizer.Init(%s, %d, %d) failed", attr.url_tokenizer.c_str(), attr.b_bos, attr.b_eos);
            return false;
        }
        update_cqdm(&cqdm, 0, "count", "tokenizer init ok");

        if (!embed_selector.Init(attr.filename_tokens_embed, attr.tokens_embed_num, attr.tokens_embed_size, attr.b_use_mmap_load_embed))
        {
            ALOGE("embed_selector.Init(%s, %d, %d) failed", attr.filename_tokens_embed.c_str(), attr.tokens_embed_num, attr.tokens_embed_size);
            return false;
        }
        
        update_cqdm(&cqdm, 1, "count", "embed_selector init ok");

        llama_layers.resize(attr.axmodel_num);
        // prefill_layers.resize(attr.prefill_axmodel_num);
        ALOGI("attr.axmodel_num:%d",attr.axmodel_num);
        char axmodel_path[1024];
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            sprintf(axmodel_path, attr.template_filename_axmodel.c_str(), i);
            llama_layers[i].filename = axmodel_path;

            int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), false);
            if (ret != 0)
            {
                ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str());
                return false;
            }
            int remain_cmm = get_remaining_cmm_size();
            sprintf(axmodel_path, "init %d axmodel ok,remain_cmm(%d MB)", i, remain_cmm);
            update_cqdm(&cqdm, i + 2, "count", axmodel_path);
            
        }

        int ret = llama_post.init(attr.filename_post_axmodel.c_str(), false);
        if (ret != 0)
        {
            ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str());
            return false;
        }

        int remain_cmm = get_remaining_cmm_size();
        sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 2, "count", axmodel_path);

        {
            _attr.max_token_len = llama_layers[0].layer.get_input("mask").nSize / sizeof(unsigned short) - 1;
            printf("\n");
            ALOGI("max_token_len : %d", _attr.max_token_len);
            // auto &input_k_cache = llama_layers[0].layer.get_input("K_cache");
            // auto &output_k_cache_out = llama_layers[0].layer.get_output("K_cache_out");
            _attr.kv_cache_size = llama_layers[0].layer.get_output("K_cache_out").nSize / sizeof(unsigned short);
            _attr.kv_cache_num = llama_layers[0].layer.get_input("K_cache").nSize / _attr.kv_cache_size / sizeof(unsigned short);
            ALOGI("kv_cache_size : %d, kv_cache_num: %d", _attr.kv_cache_size, _attr.kv_cache_num);
            if (_attr.max_token_len > _attr.kv_cache_num)
            {
                ALOGE("max_token_len(%d) > kv_cache_num(%d)", _attr.max_token_len, _attr.kv_cache_num);
                return false;
            }

            _attr.prefill_token_num = llama_layers[0].layer.get_input(1, "indices").vShape[1];
            ALOGI("prefill_token_num : %d", _attr.prefill_token_num);
			for (size_t i = 0; i < llama_layers[0].layer.get_num_input_groups() - 1; i++)
            {
                int prefill_max_kv_cache_num = llama_layers[0].layer.get_input(i + 1, "K_cache").vShape[1];
                ALOGI("grp: %ld, prefill_max_token_num : %d", i + 1, prefill_max_kv_cache_num);
                _attr.prefill_max_kv_cache_num_grp.push_back(prefill_max_kv_cache_num);
            }
            _attr.prefill_max_token_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_max_kv_cache_num_grp.size() - 1];
            ALOGI("prefill_max_token_num : %d", _attr.prefill_max_token_num);
        }
        

        // Reset();
        ALOGI("LLM init ok");
        return true;
    }

    LLMAttrType *getAttr()
    {
        return &_attr;
    }

    void Deinit()
    {
        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            llama_layers[i].layer.release();
        }
        llama_post.release();
        embed_selector.Deinit();
    }

    void Stop()
    {
        b_stop = true;
    }

    int TextToken2Embeds(std::vector<int> &token_ids,  std::vector<unsigned short> &token_embeds)
    {   
        if(token_embeds.empty() || token_embeds.size() < token_ids.size()* _attr.tokens_embed_size)
        {
            token_embeds.resize(token_ids.size()* _attr.tokens_embed_size);
        }

        for (size_t i = 0; i < token_ids.size(); i++)
        {
            embed_selector.getByIndex(token_ids[i], token_embeds.data() + i * _attr.tokens_embed_size);
        }
        return token_embeds.size();
    }


    int Forward(std::vector<unsigned short>& text_embed, bool is_causal=true)
    {
        b_stop = false;

        int input_embed_num = text_embed.size() / _attr.tokens_embed_size;
        int prefill_split_num = ceil((double)input_embed_num / _attr.prefill_token_num);
        ALOGI("input token num : %d, prefill_split_num : %d", input_embed_num, prefill_split_num);
        if (input_embed_num > _attr.prefill_max_token_num)
        {
            ALOGE("input token num(%d) > prefill_max_token_num(%d)", input_embed_num, _attr.prefill_max_token_num);
            return -1;
        }

        std::vector<std::vector<int>> position_ids(1, std::vector<int>(input_embed_num));
        for (int i = 0; i < input_embed_num; ++i) {
            position_ids[0][i] = i;
        }

        int kv_cache_num;

        for (size_t p = 0; p < prefill_split_num; p++)
        {
            if (b_stop)
            {
                break;
            }
            _attr.prefill_grpid = p + 1;
            kv_cache_num = p * _attr.prefill_token_num;
            std::vector<unsigned short> mask_tmp;
            bfloat16 bf16 = -65536.f;
            mask_tmp.resize(1 * _attr.prefill_token_num * (kv_cache_num + _attr.prefill_token_num), bf16.data);
            int input_num_token = _attr.prefill_token_num;
            if (p == prefill_split_num - 1)
            {
                input_num_token = input_embed_num - p * _attr.prefill_token_num;
            }

            ALOGI("input_num_token:%d", input_num_token);
            for (size_t i = 0; i < _attr.prefill_token_num; i++)
            {
                if (i < input_num_token)
                {
                    int mask_current_start = kv_cache_num;
                    auto mask_ptr = mask_tmp.data() + i * (kv_cache_num + _attr.prefill_token_num);

                    for (int j = 0; j < _attr.precompute_len + p * _attr.prefill_token_num; j++)
                    {
                        mask_ptr[j] = 0;
                    }
                    
                    if(is_causal)
                    {
                        for (int j = mask_current_start; j < mask_current_start + i + 1; j++)
                        {
                            mask_ptr[j] = 0;
                        }
                    }
                    else
                    {
                        for (int j = mask_current_start; j < mask_current_start + _attr.prefill_token_num ; j++)
                        {
                            mask_ptr[j] = 0;
                        }
                    }
                    
                }
            }

            void *  p_data = text_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size ;
            int size_data = -1;
            if (p == (prefill_split_num - 1))
            {
                size_data = (input_embed_num - p * _attr.prefill_token_num) * _attr.tokens_embed_size * sizeof(unsigned short);
            }
            else
            {
                size_data = _attr.prefill_token_num * _attr.tokens_embed_size * sizeof(unsigned short);
            }


            for (unsigned int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop)
                {
                    break;
                }

                auto &layer = llama_layers[m];

                // set indices
                auto &input_indices = layer.layer.get_input(_attr.prefill_grpid, "indices");
                unsigned int *input_indices_ptr = (unsigned int *)input_indices.pVirAddr;
                memset(input_indices_ptr, 0, input_indices.nSize);
                // ALOGI("position_ids");
                for(unsigned int i=0; i< position_ids.size(); i++){
                    for(unsigned int j=_attr.precompute_len + p * _attr.prefill_token_num, jj=0; j<_attr.precompute_len + (p + 1) * _attr.prefill_token_num; j++,jj++){
                        if(j<position_ids[i].size()){
                            input_indices_ptr[ i*_attr.prefill_token_num+jj ] = position_ids[i][j];
                        }
                    }
                }    

                // set mask
                auto &input_mask = layer.layer.get_input(_attr.prefill_grpid, "mask");
                memcpy((void *)input_mask.pVirAddr, (void *)mask_tmp.data(), mask_tmp.size() * sizeof(unsigned short));
                // set input
                auto &input_input = layer.layer.get_input(_attr.prefill_grpid, "input");
                memcpy((void *)input_input.pVirAddr, p_data, size_data);

                layer.layer.inference(_attr.prefill_grpid);

                auto &input_decoder_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                auto &input_decoder_v_cache = layer.layer.get_input(decode_grpid, "V_cache");

                auto &output_k_cache = layer.layer.get_output(_attr.prefill_grpid, "K_cache_out");
                auto &output_v_cache = layer.layer.get_output(_attr.prefill_grpid, "V_cache_out");

                int kv_offset = (_attr.precompute_len + p * _attr.prefill_token_num) * _attr.kv_cache_size;

                memcpy((unsigned short *)input_decoder_k_cache.pVirAddr + kv_offset,
                        (void *)output_k_cache.pVirAddr,
                            sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                memcpy((unsigned short *)input_decoder_v_cache.pVirAddr + kv_offset,
                            (void *)output_v_cache.pVirAddr,
                            sizeof(unsigned short) * input_num_token * _attr.kv_cache_size
                            );

                for(int gid=_attr.prefill_grpid+1; gid<prefill_split_num+1; gid++){
                    auto &input_prefill_k_cache = layer.layer.get_input(gid, "K_cache");
                    memcpy((unsigned short *)input_prefill_k_cache.pVirAddr + kv_offset,
                                (void *)output_k_cache.pVirAddr,
                                sizeof(unsigned short) * input_num_token * _attr.kv_cache_size
                                );
                }

                for(int gid=_attr.prefill_grpid+1; gid<prefill_split_num+1; gid++){
                    auto &input_prefill_v_cache = layer.layer.get_input(gid, "V_cache");
                    memcpy((unsigned short *)input_prefill_v_cache.pVirAddr + kv_offset,
                                (void *)output_v_cache.pVirAddr,
                                sizeof(unsigned short) * input_num_token * _attr.kv_cache_size
                                );
                }

                auto &output = layer.layer.get_output(_attr.prefill_grpid, "output");
                memcpy(p_data, (void *)output.pVirAddr, size_data );

            }
            

        }

        for(int i=0; i<input_embed_num; i++)
        {
            void * p_data = text_embed.data() + i * _attr.tokens_embed_size;
            auto &input = llama_post.get_input(0);
            memcpy((void *)input.pVirAddr, p_data, _attr.tokens_embed_size * sizeof(unsigned short));
            llama_post.inference();

            auto &output_post = llama_post.get_output("output_norm");  
            memcpy(p_data, output_post.pVirAddr,  _attr.tokens_embed_size * sizeof(unsigned short));
        }

        return 0;
    }
    
    
    int ForwardStep(std::vector<unsigned short>& embed, unsigned int position_id)
    {
        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        for (size_t i = 0; i < position_id; i++)
        {
            mask[i] = 0;
        }
        unsigned int indices = position_id;
        if (b_stop)
        {
            return 0;
        }

        memcpy((void *)llama_layers[0].layer.get_input(decode_grpid, "input").pVirAddr, embed.data(), llama_layers[0].layer.get_input(decode_grpid, "input").nSize);

        for (int m = 0; m < _attr.axmodel_num; m++)
        {
            if (b_stop)
            {
                break;
            }

            auto &layer = llama_layers[m];

            auto &input_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
            auto &input_v_cache = layer.layer.get_input(decode_grpid, "V_cache");

            auto &input_indices = layer.layer.get_input(decode_grpid, "indices");
            memcpy((void *)input_indices.pVirAddr, &indices, sizeof(indices));

            auto &input_mask = layer.layer.get_input(decode_grpid, "mask");
            memcpy((void *)input_mask.pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));

            layer.layer.inference(decode_grpid);

            auto &output_k_cache = layer.layer.get_output(decode_grpid, "K_cache_out");
            memcpy((unsigned short *)input_k_cache.pVirAddr + indices * _attr.kv_cache_size, (void *)output_k_cache.pVirAddr, output_k_cache.nSize);

            auto &output_v_cache = layer.layer.get_output(decode_grpid, "V_cache_out");
            memcpy((unsigned short *)input_v_cache.pVirAddr + indices * _attr.kv_cache_size, (void *)output_v_cache.pVirAddr, output_v_cache.nSize);

            if (m == _attr.axmodel_num - 1)
            {
                memcpy((void *)llama_post.get_input(0).pVirAddr,
                        (void *)layer.layer.get_output(decode_grpid, "output").pVirAddr, llama_post.get_input(0).nSize);
            }
            else if (m < _attr.axmodel_num - 1)
            {
                memcpy((void *)llama_layers[m + 1].layer.get_input(decode_grpid, "input").pVirAddr,
                        (void *)layer.layer.get_output(decode_grpid, "output").pVirAddr, layer.layer.get_input(decode_grpid, "input").nSize);
            }
        }

        llama_post.inference();
        auto &output_post = llama_post.get_output("output_norm");       
        memcpy(embed.data(), output_post.pVirAddr,  _attr.tokens_embed_size * sizeof(unsigned short));

        return 0;
    }

    void reset()
    { 
        for (size_t i = 0; i < _attr.axmodel_num; i++)
        {
            for (size_t j = 0; j < llama_layers[i].layer.get_num_input_groups(); j++)
            {
                memset((void *)llama_layers[i].layer.get_input(j, "K_cache").pVirAddr, 0, llama_layers[i].layer.get_input(j, "K_cache").nSize);
                memset((void *)llama_layers[i].layer.get_input(j, "V_cache").pVirAddr, 0, llama_layers[i].layer.get_input(j, "V_cache").nSize);
            }
        }
    }
};
