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


class VisualEncoder
{
private:
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

    int Encode(std::vector<cv::Mat>& src, std::vector<unsigned short> &out_embed, Config & cfg)
    {
        int temporal_patch_size=cfg.vision_config.temporal_patch_size;
        int merge_size=cfg.vision_config.spatial_merge_size;
        int patch_size=cfg.vision_config.patch_size;
        int ret;
        timer t;
        t.start();

        unsigned int grid_h = cfg.vision_config.height / cfg.vision_config.patch_size;
        unsigned int grid_w = cfg.vision_config.width / cfg.vision_config.patch_size;
       
        std::vector<std::vector<unsigned char>> pixel_values;

        int w=cfg.vision_config.width, h=cfg.vision_config.height;
        
        Qwen2VideoProcessor(  src, pixel_values,
                        h, w,
                        temporal_patch_size, merge_size, patch_size);

        int channel = src[0].channels();
        int hwc = grid_h * grid_w * temporal_patch_size * patch_size * patch_size * channel;

        if(src.size()==1){
            int grid_t = 1;
            cfg.image_grid_thw = {{grid_t, grid_h, grid_w}};
        }else{
            cfg.video_grid_thw = {{pixel_values.size(), grid_h, grid_w}};
        }
        
        int cnt = 0;
        for(auto &pixel : pixel_values){

            void *data = model.get_input(0).pVirAddr;
            memcpy(data, pixel.data(), hwc);
            model.inference();

            size_t size = model.get_output(0).nSize / sizeof(float);
            if(out_embed.empty()){
                out_embed.resize( size * pixel_values.size() );
            }
            
            AX_SYS_MinvalidateCache(model.get_output(0).phyAddr, model.get_output(0).pVirAddr, model.get_output(0).nSize);

            float *output_data = (float *)model.get_output(0).pVirAddr;
            for (size_t i = 0; i < size; i++)
            {
                out_embed[cnt++] = bfloat16(output_data[i]).data;
            }

        }

        ALOGI("image encode time : %f ms, size : %d", t.cost(), out_embed.size());
        return 0;
    }


};