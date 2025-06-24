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


class Token2WavDit
{
private:
    int mel_dim;
    int repeats;
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

};