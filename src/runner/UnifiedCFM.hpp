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
#include "LocDit.hpp"

struct CfmConfig
{
    float sigma_min = 1e-6;
    std::string solver = "euler";
    std::string t_scheduler = "log-norm";
};

class UnifiedCFM
{
private:
    std::vector<float> rand_noise;
    // std::vector<float> t_span;
    LocDit estimator;

    int init_noise(std::string model_dir)
    {
        return readtxt(model_dir+"/rand_noise.txt", rand_noise);
    }

    // int init_tspan(int n_timesteps)
    // {
    //     if(n_timesteps <4)
    //     {
    //         return -1;
    //     }

    //     n_timesteps = n_timesteps;
    //     t_span = linspace(1.0, 0.0, n_timesteps + 1);
    //     return 0;
    // }

public:
    bool Init(int in_channels, CfmConfig &cfm_params, LLMAttrType &locdit_config, std::string &dir_axmodels)
    {
        int ret;
        ret = init_noise(dir_axmodels);
        if(ret!=0)
        {
            ALOGE("init noise failed", );
            return false;
        }


        if(!estimator.Init(locdit_config, dir_axmodels))
        {
            ALOGE("Init estimator failed");
            return false;
        }


        return true;
    }

    int Forward()
    {

    }
};