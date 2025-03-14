

#ifndef _MROPE_H_
#define _MROPE_H_

#include <vector>

struct Config {
    struct VisionConfig {
        int spatial_merge_size;
        int tokens_per_second;
    };
    VisionConfig vision_config;
    int image_token_id;
    int video_token_id;
    int vision_start_token_id;
};


std::vector<std::vector<int>> get_rope_index(
    const Config& config,
    const std::vector<int>& input_ids,
    const std::vector<std::vector<int>>& image_grid_thw,
    const std::vector<std::vector<int>>& video_grid_thw,
    const std::vector<double>& second_per_grid_ts); 

#endif 