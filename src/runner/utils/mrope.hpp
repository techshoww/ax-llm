

#ifndef _MROPE_H_
#define _MROPE_H_

#include <vector>

struct Config {
    struct VisionConfig {
        int temporal_patch_size;
        int tokens_per_second;
        int spatial_merge_size;
        int patch_size;
        int width;
        int height;
        int fps;
    };
    VisionConfig vision_config;
    int image_token_id;
    int video_token_id;
    int vision_start_token_id;
    int audio_token_id;
    int audio_start_token_id;
    int position_id_per_seconds;
    int seconds_per_chunk;

    std::vector<std::vector<int>> image_grid_thw;   // auto calc
    std::vector<std::vector<int>> video_grid_thw;   // auto calc
};


std::vector<std::vector<int>> get_rope_index(
    const Config& config,
    const std::vector<int>& input_ids,
    const std::vector<std::vector<int>>& image_grid_thw,
    const std::vector<std::vector<int>>& video_grid_thw,
    const bool use_audio_in_video,
    const std::vector<int>& audio_seqlens,
    const std::vector<int>& second_per_grids);
#endif 