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
#include "BaseRunner.hpp"

class SimpleLayerOnnx
{
private:
    std::shared_ptr<BaseRunner> model;
    int in_size_model;
    int out_size_model;
    

public:
    bool Init(const std::string &path_model, int in_size, int out_size)
    {
        in_size_model = in_size;
        out_size_model = out_size;

        model = CreateRunner(RT_OnnxRunner);
        if(model == nullptr)
        {
            ALOGE("init model failed");
            return false;
        }
        BaseConfig config_proj;
        config_proj.nthread = 2;
        config_proj.onnx_model = path_model;
        model->load(config_proj);

        return true;
    }

    void Deinit()
    {
    }

    int Forward(std::vector<float> &hidden, std::vector<float> &out)
    {
        if(hidden.size() > in_size_model)
        {
            int cnt = hidden.size() / in_size_model;
            out.resize(cnt * out_size_model);
            for(int i=0; i<cnt; i++)
            {
                ForwardStep( hidden.data() + i * in_size_model, out.data() + i * out_size_model );
            }
        }
        else
        {
            out.resize(out_size_model);
            ForwardStep(hidden.data(), out.data());
        }

        return 0;
    }

    int ForwardStep(void * p_in , void * p_out)
    {
        void * p = model->getInputPtr(0);
        memcpy(p, p_in, in_size_model * sizeof(float));   

        model->inference();

        p = model->getOutputPtr(0);
        memcpy(p_out, p, out_size_model * sizeof(float));

        return 0;
    }
};