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

class Function
{
private:
    std::vector<float> cond;
    std::vector<float> spk;
    std::vector<int> code;
    float guidance_scale;
    ax_runner_ax650 * model;

public:
    Function(std::vector<float>& cond,
                std::vector<float>& spk,
                std::vector<int>& code,
                float guidance_scale,
                ax_runner_ax650 * model
    ){
        this->cond=cond;
        this->spk = spk;
        this->code = code;
        this->guidance_scale = guidance_scale;
        this->model = model;
    }
    int Forward(
            std::vector<float>& x, 
            std::vector<float>& cond,
            std::vector<float>& spk,
            std::vector<int>& code,
            std::vector<float> time,
            std::vector<float>& output
            )
    {
         void *data = model->get_input("x").pVirAddr;
        memcpy(data, x.data(), x.size()*sizeof(float));
        data = model->get_input("cond").pVirAddr;
        memcpy(data, cond.data(), cond.size()*sizeof(float));
        data = model->get_input("spk").pVirAddr;
        memcpy(data, spk.data(), spk.size()*sizeof(float));
        data = model->get_input("code").pVirAddr;
        memcpy(data, code.data(), code.size()*sizeof(int));
        data = model->get_input("time").pVirAddr;
        memcpy(data, time.data(), time.size()*sizeof(float));


        model->inference();

        size_t size = model->get_output(0).nSize / sizeof(float);
        if(output.empty()){
            output.resize( size );
        }
        
        AX_SYS_MinvalidateCache(model->get_output(0).phyAddr, model->get_output(0).pVirAddr, model->get_output(0).nSize);

        float *output_data = (float *)model->get_output(0).pVirAddr;

        memcpy(output.data(), output_data, size*sizeof(float));

        

        return 0;
    }

    int Run(float t, std::vector<float>& x, std::vector<float>& ret)
    {
        std::vector<float> output;
        Forward(x, cond, spk, code, {t}, output);

        int size = output.size();
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
};


class RungeKutta4ODESolver
{
private:
    Function * fn;
    float _one_third = 1 / 3;
    float _two_thirds = 2 / 3;

public:
    
    RungeKutta4ODESolver(Function * fun)
    {
        this->fn = fun;
    }

    int _rk4_step(Function * fun, 
                    float time_start, 
                    float time_step, 
                    float time_end, 
                    std::vector<float>& value_start, 
                    std::vector<float>& k1,
                    std::vector<float>& ret )
    {
        std::vector<float> k2;

        std::vector<float> v2(value_start.size(), 0);
        for(int i=0; i<value_start.size();i++){
            v2[i] = value_start[i] + time_step * k1[i] * _one_third;
        }

        fun->Run(time_start + time_step * _one_third, v2, k2);

        std::vector<float> k3;
        std::vector<float> v3(value_start.size(), 0);
        for(int i=0; i<value_start.size();i++){
            v3[i] = value_start[i] + time_step *(k2[i]-k1[i]*_one_third);
        }
        fun->Run(time_start + time_step * _two_thirds, v3, k3);

        std::vector<float> k4;
        std::vector<float> v4(value_start.size(), 0);
        for(int i=0; i<value_start.size();i++){
            v4[i] = value_start[i] + time_step *(k1[i] - k2[i] + k3[i]);
        }
        fun->Run(time_end, v4, k4);

        if(ret.empty()){
            ret.resize(value_start.size());
        }
        for(int i=0; i<value_start.size(); i++){
            ret[i] = (k1[i] + 3 * (k2[i] + k3[i]) + k4[i]) * time_step / 8;
        }

        return 0;
    }    

    int  _compute_step(Function* fun, 
                        float time_start, 
                        float time_step, 
                        float time_end, 
                        std::vector<float>& value_start,
                        std::vector<float>& ret
                        )
    {
        std::vector<float> function_value_start;
        fun->Run(time_start, value_start, function_value_start);

        _rk4_step(fun, time_start, time_step, time_end, value_start, function_value_start, ret);

        return 0;
    }

    void _linear_interpolation(
        float time_start,
        float time_end,
        std::vector<float>& value_start,
        std::vector<float>& value_end,
        float time_point,
        float * ret
    )
    {
        if(time_point==time_start){
            memcpy(ret, value_start.data(), value_start.size()*sizeof(float));
        }
        else if(time_point==time_end){
            memcpy(ret, value_end.data(), value_end.size()*sizeof(float));
        }
        
        float weight = (time_point - time_start) / (time_end - time_start);
        

        for(int i=0;i<value_start.size();i++){
            ret[i] = value_start[i] + weight * (value_end[i] - value_start[i]);
        }
    }

    int integrate(std::vector<float>& time_points, std::vector<float>& solution)
    {
        int current_index = 1;
        std::vector<float> current_value(1200*80);
        for(int i=0; i<time_points.size()-1; i++){
            float time_start = time_points[i];
            float time_end = time_points[i+1];
            float time_step = time_end - time_start;
            std::vector<float> delta_value;
            _compute_step(fn, time_start, time_step, time_end, current_value, delta_value);

            std::vector<float> next_value(current_value.size(), 0);
            for(int i=0; i<current_value.size(); i++){
                next_value[i] = current_value[i] + delta_value[i];
            }

            while(current_index<time_points.size() && 
                    time_end>=time_points[current_index])
            {
                
                _linear_interpolation(
                    time_start,
                    time_end,
                    current_value,
                    next_value,
                    time_points[current_index],
                    solution.data()+current_value.size()*current_index);

                current_index += 1;
            }
            memcpy(current_value.data(), next_value.data(), next_value.size()*sizeof(float));
            current_value = next_value;
        }

        return 0;
    }
};