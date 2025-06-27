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
#include "RungeKutta4ODESolver.hpp"

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
        float guidance_scale,
        float sway_coefficient,
        std::vector<float>& generated_mel_spec)
    {
        int max_duration = code.size() * repeats;
        
        
        std::vector<std::vector<float>> cond_e(max_duration, cond);

        int num_steps=10;
        std::vector<float> t = {0.0000, 0.0152, 0.0603, 0.1340, 0.2340, 0.3572, 0.5000, 0.6580, 0.8264, 1.0000};
        
        std::vector<float> trajectory(num_steps*max_duration*mel_dim, 0);

        Function * fun = new Function(ref_mel, cond, code, guidance_scale, &model);
        RungeKutta4ODESolver  solver(fun);
        
        solver.integrate(t, trajectory);
       
        if(generated_mel_spec.empty()){
            generated_mel_spec.resize(max_duration*mel_dim);
        }

        int start = (num_steps-1)*max_duration*mel_dim;
        //transpose
        for(int i=0; i<max_duration;i++){
            for(int j=0; j<mel_dim;j++){
                generated_mel_spec[j*max_duration+i] = trajectory[start+i*mel_dim+j];
            }
        }

        return 0;

    }
           


};