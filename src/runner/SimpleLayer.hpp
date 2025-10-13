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

class SimpleLayer
{
private:
    ax_runner_ax650 model;
    int in_size_axmodel;
    int out_size_axmodel;
    

public:
    bool Init(const std::string &path_axmodel, int in_size_axmodel, int out_size_axmodel)
    {
        int ret;
        ret = model.init(path_axmodel.c_str(), false);
        if (ret != 0)
        {
            ALOGE("init axmodel(%s) failed", path_axmodel.c_str());
            return false;
        }

        return true;
    }

    void Deinit()
    {
        model.release();
    }

    int Forward(std::vector<float> &hidden, std::vector<float> &out)
    {
        if(hidden.size() > in_size_axmodel)
        {
            int cnt = hidden.size() / in_size_axmodel;
            out.resize(cnt * out_size_axmodel);
            for(int i=0; i<cnt; i++)
            {
                ForwardStep( hidden.data() + i * in_size_axmodel, out.data() + i * out_size_axmodel );
            }
        }
        else
        {
            out.resize(out_size_axmodel);
            ForwardStep(hidden.data(), out.data());
        }

        return 0;
    }

    int ForwardStep(void * p_in , void * p_out)
    {
        void * p = model.get_input("hidden").pVirAddr;
        memcpy(p, p_in, in_size_axmodel * sizeof(float));   

        model.inference();

        p = model.get_output(0).pVirAddr;
        memcpy(p_out, p, out_size_axmodel * sizeof(float));

        return 0;
    }
};