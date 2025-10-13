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


class LocDit
{
private:
    ax_runner_ax650 part1;
    ax_runner_ax650 part3;
    MiniCPM decoder;

public:
    bool Init(LLMAttrType &config, std::string &dir_axmodels)
    {
        int ret;
        ret = part1.init((dir_axmodels+"/locdit.part1.axmodel").c_str(), false);
        if (ret != 0)
        {
            ALOGE("init axmodel(%s) failed", (dir_axmodels+"/locdit.part1.axmodel").c_str());
            return false;
        }

        ret = part3.init((dir_axmodels+"/locdit.part3.axmodel").c_str(), false);
        if (ret != 0)
        {
            ALOGE("init axmodel(%s) failed", (dir_axmodels+"/locdit.part3.axmodel").c_str());
            return false;
        }

        if(!decoder.Init(config))
        {
            ALOGE("init decoder failed");
            return false;
        }

        return true;
    }

    void Deinit()
    {
        part1.release();
        part3.release();
        decoder.Deinit();
    }

    int Forward(std::vector<float> &x, std::vector<float> &mu, std::vector<float> &t, std::vector<float> &cond, std::vector<float> &output)
    {
        void * p = part1.get_input("x").pVirAddr;
        memcpy(p, x.data(), x.size() * sizeof(float));
        p = part1.get_input("mu").pVirAddr;
        memcpy(p, mu.data(), mu.size() * sizeof(float));
        p = part1.get_input("t").pVirAddr;
        memcpy(p, t.data(), t.size() * sizeof(float));
        p = part1.get_input("cond").pVirAddr;
        memcpy(p, cond.data(), cond.size() * sizeof(float));

        part1.inference();

        std::vector<unsigned short> out1_part1(5 * decoder._attr.hidden_size, 0);
        std::vector<unsigned short> out2_part1(5 * decoder._attr.hidden_size, 0);

        p = part1.get_output(0).pVirAddr;
    

        // float32 to bfloat16
        for(int j=0; j<out1_part1.size(); j++)
        {
            out1_part1[j] = bfloat16(((float *)p)[j]).data;
        }

        for(int j=0; j<out2_part1.size(); j++)
        { 
            out2_part1[j] = bfloat16(((float *)p)[ 5 * decoder._attr.hidden_size + j]).data;
        }
        

        int ret = decoder.Forward(out1_part1, false);
        if(ret!=0)
        {
            ALOGE("decoder Forward failed");
            return -1;
        }
        ret = decoder.Forward(out2_part1, false);
        if(ret!=0)
        {
            ALOGE("decoder Forward failed");
            return -1;
        }

        p = part3.get_input("hidden").pVirAddr;
        
        // bfloat16 to float32 

        for(int j=0; j<5 * decoder._attr.hidden_size; j++)
        {
            unsigned int tmp = out1_part1[j] << 16;
            ((float *)p)[j] = *reinterpret_cast<float *>(&tmp);
        }

        for(int j=0; j<5 * decoder._attr.hidden_size; j++)
        {
            unsigned int tmp = out2_part1[j] << 16;
            ((float *)p)[5 * decoder._attr.hidden_size + j] = *reinterpret_cast<float *>(&tmp);
        }

        part3.inference();

        auto &out_part3 = part1.get_output(0);
        output.resize(out_part3.nSize/sizeof(float));
        memcpy(output.data(), out_part3.pVirAddr, out_part3.nSize);

        return 0;
    }
};