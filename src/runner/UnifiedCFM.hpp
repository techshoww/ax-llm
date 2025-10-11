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
    int in_channels;
    std::vector<float> rand_noise;      // shape: (1, in_channels, 2)
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

        in_channels = in_channels;
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

    void Deinit()
    {
        std::vector<float>().swap(rand_noise);
        estimator.Deinit();
    }

    int Forward(std::vector<float> &mu, std::vector<float> &cond, int n_timesteps, 
                int cfg_value, float sway_sampling_coef, bool use_cfg_zero_star,
                std::vector<float> &out)
    {
        auto t_span = linspace(1.0, 0.0, n_timesteps + 1);
        for(int i=0; i<t_span.size(); i++)
        {
            float t = t_span[i];
            t_span[i] = t + sway_sampling_coef * (std::cos(M_PI / 2 * t) - 1 + t);
        }

        out = rand_noise;

        return SolveEuler(out, t_span, mu, cond, cfg_value, use_cfg_zero_star);
    } 

    float OptimizedScale(std::vector<float> &positive_flat, std::vector<float> &negative_flat)
    {
        float dot_product = 0;
        float squared_norm = 1e-8;
        for(int i=0; i<positive_flat.size(); i++)
        {
            dot_product += positive_flat[i] * negative_flat[i];
            squared_norm += negative_flat[i] * negative_flat[i];
        }

        return dot_product / squared_norm ;
    }
    int SolveEuler(std::vector<float> &x, std::vector<float> &t_span, std::vector<float> &mu, std::vector<float> &cond, 
                    int cfg_value, bool use_cfg_zero_star=true)
    {

        int len = x.size()/in_channels;
        float t = t_span[0];
        float dt = t_span[0] - t_span[1];

        std::vector<float> x_in(2*in_channels*len, 0);
        std::vector<float> mu_in(2*in_channels*len,0);
        std::vector<float> t_in(2,0);
        std::vector<float> cond_in(2*in_channels*len, 0);
        std::vector<float> dphi_dt(2*in_channels*len, 0);
        
        int zero_init_steps = std::max(1, int(t_span.size() * 0.04));
        float st_star = 1.0;
        for(int step=1; step<t_span.size(); step++)
        {   
            if(!(use_cfg_zero_star && step <= zero_init_steps))  
            {
                memcpy(x_in.data(), x.data(), x.size() * sizeof(float));
                memcpy(x_in.data()+x.size(),  x.data(), x.size() * sizeof(float));

                memcpy(mu_in.data(), mu.data(), mu.size() * sizeof(float));

                t_in[0] = t;
                t_in[1] = t;

                memcpy(cond_in.data(), cond.data(), cond.size() * sizeof(float));
                memcpy(cond_in.data()+cond.size(),  cond.data(), cond.size() * sizeof(float));

                int ret = -1;
                ret = estimator.Forward(x_in, mu_in, t_in, cond_in, dphi_dt);
                if(ret!=0)
                {
                    ALOGE("UnifiedCFM estimator Foward failed");
                    return -1;
                }

                if(use_cfg_zero_star)
                {
                    std::vector<float> positive_flat(dphi_dt.begin(), dphi_dt.begin()+in_channels*len);
                    std::vector<float> negative_flat(dphi_dt.begin()+in_channels*len, dphi_dt.end());
                    st_star = OptimizedScale(positive_flat, negative_flat);
                }
                else
                {
                    st_star = 1.0;
                }

                for(int i=0; i<in_channels*len; i++)
                {
                    dphi_dt[i] = dphi_dt[in_channels*len + i] * st_star + cfg_value * (dphi_dt[i] - dphi_dt[in_channels*len + i] * st_star);
                }

            }

            for(int i=0; i<in_channels*len; i++)
            {
                x[i] = x[i] - dt * dphi_dt[i];
            }
            t = t - dt;

            if(step < t_span.size()-1)
            {
                dt = t - t_span[step + 1];
            }
            else
            {
                return 0;  // 什么都不做，返回 x 
            }
        }
        
        return 0;

    }
};