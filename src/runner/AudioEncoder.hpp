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


class AudioEncoder
{
private:
    int n_window = 100;
    int dim_input = 128;
    int dim_output = 2048;
    ax_runner_ax650 model;

public:
    bool Init(std::string model_path)
    {
        if(model.init(model_path.c_str(), false)!=0){
            return false;
        }
        return true;
    }

    void Deinit()
    {
        model.release();
    }

    int Encode(std::vector<float>& input, std::vector<unsigned short> &out_embed)
    {
        timer t;
        t.start();
        
        int feature_lens = input.size()/dim_input;
        int aftercnn_lens = aftercnn_lens/2;
        int chunk_num = (feature_lens + n_window * 2-1) / (n_window * 2); // ceil
        LOGI("chunk_num:%d", chunk_num);

        std::vector<int> chunk_lengths(chunk_num, n_window*2);
        chunk_lengths[chunk_num-1] = feature_lens % (n_window * 2);

        int max_lens_after_cnn =  (n_window*2 - 1) / 2 + 1;
        std::vector<std::vector<float>> padded_feature(chunk_num, std::vector<int>(dim_input*n_window*2, 0));
        std::vector<std::vector<int>> padded_mask(chunk_num, std::vector<int>(n_window*2, 0));

        int start=0;
        for(int i=0; i < chunk_num, i++){
            memcpy(padded_feature[i].begin(), input.begin()+dim_input*start, dim_input*chunk_lengths[i]*sizeof(float));
            std::fill(padded_mask[i].begin(), padded_mask[i].begin()+chunk_lengths[i], 1);
            start = start + chunk_lengths[i];   
        }

        int cumsum=0;
        std::vector<int> cu_seqlens(chunk_num+1, 0);
        for(int i=0; i< chunk_num; i++){
            cumsum +=  (chunk_num[i] - 1) / 2 + 1;
            cu_seqlens[i+1] = cumsum;
        }

        int seq_len = (chunk_num*n_window*2 -1) / 2 + 1;
        // std::vector<float> attention_mask(seq_len*seq_len, 0);
        // for(int i=1; i<cu_seqlens.size();i++){
        //     for(int j=cu_seqlens[i-1]; j<cu_seqlens[i]; j++){
        //         std::fill( attention_mask.begin()+j*seq_len+cu_seqlens[i-1], attention_mask.begin()+j*seq_len+cu_seqlens[i], 1);
        //     }
        // }
        
        int out_lens = (feature_lens - 2) / 2 + 1;

        if(out_embed.empty()){
            out_embed.resize( out_lens * dim_output );
        }

        int cnt = 0;
        for(int i=0; i < chunk_num; i++){
            void *data = model.get_input(0).pVirAddr;
            memcpy(data, padded_feature[i].begin(), dim_input*n_window*2*sizeof(float));
            void *data = model.get_input(1).pVirAddr;
            memcpy(data, padded_mask[i].begin(), n_window*2*sizeof(int));

            
            std::vector<float> attention_mask(n_window*n_window, 0);
            int len = cu_seqlens[i+1]-cu_seqlens[i]
            if(len == n_window){
                std::fill(attention_mask.begin(), attention_mask.end(), 1);
            }
            else{
                for(int j=0; j<len; j++){
                    std::fill(attention_mask.begin()+j*n_window, attention_mask.begin()+j*n_window+len, 1);
                }
            }
            
            void *data = model.get_input(2).pVirAddr;
            memcpy(data, attention_mask.begin(), n_window*n_window*sizeof(float));

            model.inference();

            int len_out =  (chunk_lengths[i] - 2) / 2 + 1;

            size_t size = model.get_output(0).nSize / sizeof(float);
            
            
            AX_SYS_MinvalidateCache(model.get_output(0).phyAddr, model.get_output(0).pVirAddr, model.get_output(0).nSize);

            float *output_data = (float *)model.get_output(0).pVirAddr;
            for (size_t i = 0; i < len_out*dim_output; i++)
            {
                out_embed[cnt++] = bfloat16(output_data[i]).data;
            }
            
        }

        ALOGI("audio encode time : %f ms, size : %d", t.cost(), out_embed.size());
        return 0;

    }

};