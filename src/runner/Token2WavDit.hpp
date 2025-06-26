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

class Fn
{
private:
    std::vector<float> cond;
    std::vector<float> spk;
    std::vector<int> code;
    float guidance_scale;
    ax_runner_ax650 & model;

    int Forward(
            std::vector<float>& x, 
            std::vector<float>& cond,
            std::vector<float>& spk,
            std::vector<int>& code,
            std::vector<float>& time,
            std::vector<float>& output
            ):
    {
         void *data = model.get_input("x").pVirAddr;
        memcpy(data, x.data(), x.size()*sizeof(float));
        data = model.get_input("cond").pVirAddr;
        memcpy(data, cond.data(), cond.size()*sizeof(float));
        data = model.get_input("spk").pVirAddr;
        memcpy(data, spk.data(), spk.size()*sizeof(float));
        data = model.get_input("code").pVirAddr;
        memcpy(data, code.data(), code.size()*sizeof(int));
        data = model.get_input("time").pVirAddr;
        memcpy(data, time.data(), time.size()*sizeof(float));


        model.inference();

        size_t size = model.get_output(0).nSize / sizeof(float);
        if(output.empty()){
            output.resize( size );
        }
        
        AX_SYS_MinvalidateCache(model.get_output(0).phyAddr, model.get_output(0).pVirAddr, model.get_output(0).nSize);

        float *output_data = (float *)model.get_output(0).pVirAddr;

        memcpy(output.data(), output_data, size*sizeof(float));

        

        return 0;
    }

    int Run(float t, std::vector<float>& x, std::vector<float>& ret)
    {
        std::vector<float>& output;
        Forward(x, cond, spk, code, {t}, output);
        float * p_pred = output.data();
        float * p_null_pred = output.data() + size/2;
        
        if(ret.empty()){
            ret.resize(size/2);
        }
        for(int i=0; i<size/2; i++){
            ret[i] = p_pred[i]+(p_pred[i]-p_null_pred[i])*guidance_scale;
        }
        return 0;
    }
}
class Token2WavDit
{
private:
    int mel_dim = 80;
    int repeats = 2;
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

    

    
    int sample(
        std::vector<float>& cond,
        std::vector<float>& ref_mel,
        std::vector<int>& code,
        int num_steps=10,
        float guidance_scale=0.5,
        float sway_coefficient=-1.0,
        std::vector<float> generated_mel_spec
    )
    {
        std::vector<float> 
    }
           


};