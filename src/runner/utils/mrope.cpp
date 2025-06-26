#include <vector>
#include <algorithm>
#include <optional>
#include <cassert>
#include "mrope.hpp"

#include <iostream>
#include <vector>
#include <numeric>  // std::iota

#include <vector>
#include <algorithm>
#include <limits>  // 用于std::numeric_limits
#include <stdexcept>  // 用于异常处理
#include <utility>

int findMaxIn2DVector(const std::vector<std::vector<int>>& vec) {
    if (vec.empty()) {
        throw std::invalid_argument("输入二维vector为空");  // 处理空vector <button class="citation-flag" data-index="7">
    }

    int max_value = std::numeric_limits<int>::min();  // 初始化为最小值 <button class="citation-flag" data-index="1">
    bool has_elements = false;

    for (const auto& subvec : vec) {
        if (!subvec.empty()) {
            has_elements = true;
            // 使用std::max_element获取子vector的最大值 <button class="citation-flag" data-index="3">
            int sub_max = *std::max_element(subvec.begin(), subvec.end());
            if (sub_max > max_value) {
                max_value = sub_max;
            }
        }
    }

    if (!has_elements) {
        throw std::invalid_argument("所有子vector均为空");  // 处理全空子vector <button class="citation-flag" data-index="7">
    }

    return max_value;
}

// 生成范围序列 [0, text_len-1]
std::vector<int> generateRange(int text_len, int start, int mul=1) {
    std::vector<int> range(text_len);
    std::iota(range.begin(), range.end(), start);  // 填充从0开始的序列 <button class="citation-flag" data-index="4">
    if(mul!=1){
        for(int i=0; i<range.size(); i++){
            range[i] *= mul;
        }
    }
    return range;
}

// 扩展为多行矩阵
std::vector<std::vector<int>> expandToMatrix(const std::vector<int>& range, int rows) {
    std::vector<std::vector<int>> matrix(rows, range);  // 每一行都是range的副本 <button class="citation-flag" data-index="4">
    return matrix;
}

// 生成多维索引
std::vector<std::vector<int>> generateIndices(int grid_t, int grid_h, int grid_w) {
    std::vector<std::vector<int>> indices(3, std::vector<int>(grid_t * grid_h * grid_w));

    int idx = 0;
    for (int t = 0; t < grid_t; ++t) {
        for (int h = 0; h < grid_h; ++h) {
            for (int w = 0; w < grid_w; ++w) {
                indices[0][idx] = t;  // 时间索引
                indices[1][idx] = h;  // 高度索引
                indices[2][idx] = w;  // 宽度索引
                ++idx;
            }
        }
    }

    return indices;
}

std::vector<std::vector<int>> get_llm_pos_ids_for_vision(
    const int start_idx,
    const int spatial_merge_size,
    std::vector<int>& t_index,
    const int grid_h,
    const int grid_w
)
{
    int llm_grid_t = t_index.size();
    int llm_grid_h = grid_h / spatial_merge_size;
    int llm_grid_w = grid_w / spatial_merge_size;

    std::vector<int> h_index;
    for(size_t ti=0; ti<llm_grid_t;ti++){
        for(size_t hi=0; hi<llm_grid_h; hi++){
            for(size_t wi=0; wi<llm_grid_w; wi++){
                h_index.push_back(hi + start_idx);
            }
        }
    } 

    
    std::vector<int> w_index;
    for(size_t ti=0; ti<llm_grid_t;ti++){
        for(size_t hi=0; hi<llm_grid_h;hi++){
            for(size_t wi=0; wi<llm_grid_w; wi++){
                w_index.push_back(wi + start_idx);
            }
        }
    }

    std::vector<int> t_index_expand;
    for(size_t ti=0; ti<llm_grid_t; ti++){
        for(size_t hw=0; hw<llm_grid_h*llm_grid_w; hw++){
            t_index_expand.push_back(t_index[ti] + start_idx);
        }
    }

    std::vector<std::vector<int>> thw_idx;
    thw_idx.push_back(t_index_expand);
    thw_idx.push_back(h_index);
    thw_idx.push_back(w_index);

    return thw_idx;

}

std::vector<std::pair<int, int>> get_chunked_index(
    std::vector<int>& token_indices,    
    const int tokens_per_chunk,
    const int remove_index
)
{
    std::vector<std::pair<int, int>> ret;
    int i=0, start_idx=0, current_chunk=1;
    while(i < token_indices.size()){
        if(token_indices[i]-remove_index >= current_chunk*tokens_per_chunk){
            ret.push_back(std::make_pair(start_idx, i));
            start_idx = i;
            current_chunk += 1;
        }
        i+=1;
    }
    ret.push_back(std::make_pair(start_idx, token_indices.size()));
    return ret;
}


std::vector<std::vector<int>> get_rope_index(
    const Config& config,
    const std::vector<int>& input_ids,
    const std::vector<std::vector<int>>& image_grid_thw,
    const std::vector<std::vector<int>>& video_grid_thw,
    const bool use_audio_in_video,
    const std::vector<int>& audio_seqlens,
    const std::vector<int>& second_per_grids) 
{
    const int spatial_merge_size = config.vision_config.spatial_merge_size;
    const int image_token_id = config.image_token_id;
    const int video_token_id = config.video_token_id;
    const int vision_start_token_id = config.vision_start_token_id;
    const int audio_token_id = config.audio_token_id;
    const int audio_start_token_id = config.audio_start_token_id;
    const int position_id_per_seconds = config.position_id_per_seconds;
    const int seconds_per_chunk = config.seconds_per_chunk;
    
    std::vector<std::vector<int>> position_ids(3);
    std::vector<int> mrope_position_deltas;

    // 处理纯文本情况
    if (input_ids.empty() || (image_grid_thw.empty() && video_grid_thw.empty())) {
        // for (size_t b = 0; b < input_ids.size(); ++b) {
            int b=0;
            for (int i = 0; i < 3; ++i) {
                std::vector<int> seq(input_ids.size());
                // 手动实现递增序列（替代std::iota）
                for (size_t j = 0; j < seq.size(); ++j) {
                    seq[j] = j;
                }
                // position_ids[i].push_back(seq);
                position_ids[i].insert(position_ids[i].end(), seq.begin(),seq.end());
            }

            mrope_position_deltas.push_back(0);
        // }
        // return {position_ids, mrope_position_deltas};
        return position_ids;
    }

    // 处理多模态情况
    // for (size_t batch_idx = 0; batch_idx < input_ids.size(); ++batch_idx) {
        // const auto& ids = input_ids[batch_idx];
        const auto & ids = input_ids;
        // const auto& mask = attention_mask.empty() ? std::vector<int>(ids.size(), 1) : attention_mask[batch_idx];
        const auto mask = std::vector<int>(ids.size(), 1);
        
        // 过滤有效token
        std::vector<int> filtered_ids;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (mask[i]) filtered_ids.push_back(ids[i]);
        }

        // 查找vision_start位置
        std::vector<int> vision_start_indices;
        // int vision_start_idx=-2;
        for (size_t i = 0; i < filtered_ids.size(); ++i) {
            if (filtered_ids[i] == vision_start_token_id) {
                vision_start_indices.push_back(i);
            }
        }
        
        int image_nums = 0, video_nums = 0, audio_nums=0;
        // for(size_t i=vision_start_idx+1; i<ids.size(); ++i){
        //     if(filtered_ids[i]==config.image_token_id){
        //         image_nums++;
        //     }
        //     if(filtered_ids[i]==config.video_token_id){
        //         video_nums++;
        //     }
        // }

        // if(filtered_ids[vision_start_idx+1]==config.image_token_id){
        //     image_nums =1;
        // }
        // if(filtered_ids[vision_start_idx+1]==config.video_token_id){
        //     video_nums =1;
        // }
        
        for (size_t i = 0; i < filtered_ids.size(); ++i) {
            if (filtered_ids[i] == audio_start_token_id) {
                audio_nums += 1;
            }
        }


        for (size_t i = 0; i < vision_start_indices.size(); ++i) {
            int vision_token = filtered_ids[vision_start_indices[i]+1];
            if(vision_token==image_token_id){
                image_nums += 1;
            }
            if(use_audio_in_video && vision_token==audio_start_token_id || vision_token==video_token_id){
                video_nums += 1;
            }
        }

        int image_idx = 0, video_idx = 0, audio_idx = 0;
        int ed_image = 0, ed_video = 0, ed_audio=0;
        std::vector<std::vector<int>> batch_pos(3);
        int st = 0, st_idx=0;
        int remain_images = image_nums;
        int remain_videos = video_nums;
        int remain_audios = audio_nums;
        std::vector<std::vector<std::vector<int>>> llm_pos_ids_list;
        
        int multimodal_nums = 0;
        if(use_audio_in_video){
            multimodal_nums = image_nums + audio_nums;
        }else{
            multimodal_nums = image_nums + video_nums + audio_nums;
        }
        for(size_t i_=0; i_<multimodal_nums; ++i_){
            if(llm_pos_ids_list.size()>0){
                st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
            }else{
                st_idx = 0;
            }

            ed_image = filtered_ids.size()+1;
            if(remain_images>0){
                for(size_t j=st; j<filtered_ids.size(); ++j){
                    if(filtered_ids[j]==config.image_token_id){
                        ed_image = j;
                        break;
                    }
                }
            }

            ed_video = filtered_ids.size()+1;
            if(remain_videos>0){
                for(size_t j=st; j<filtered_ids.size(); ++j){
                    if(filtered_ids[j]==config.video_token_id){
                        ed_video = j;
                        break;
                    }
                }
            }
            
            ed_audio = filtered_ids.size()+1;
            if(remain_audios>0){
                for(size_t j=st; j<filtered_ids.size(); ++j){
                    if(filtered_ids[j]==config.audio_token_id){
                        ed_audio = j;
                        break;
                    }
                }
            }

            int t,h,w;
            double second_per_grid_t;
            int ed;
            int min_ed;
            int text_len, bos_len, audio_len, eos_len;
            if(ed_image < ed_video){
                min_ed = ed_image;
            }else{
                min_ed = ed_video;
            }
            if(min_ed >= ed_audio){
                min_ed = ed_audio;
            }

            if(min_ed == ed_audio){
                text_len = min_ed - st - 1;
                if(text_len!=0){
                    if(llm_pos_ids_list.size()>0){
                        st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
                    }else{
                        st_idx = 0;
                    }

                    llm_pos_ids_list.push_back(expandToMatrix(generateRange(text_len, st_idx), 3));
                }

                if(llm_pos_ids_list.size()>0){
                    st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
                }else{
                    st_idx = 0;
                }

                bos_len = 1;
                llm_pos_ids_list.push_back(expandToMatrix(generateRange(bos_len, st_idx), 3));

                if(llm_pos_ids_list.size()>0){
                    st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
                }else{
                    st_idx = 0;
                }

                audio_len = int((int((audio_seqlens[audio_idx] - 1) / 2) + 1 - 2) / 2) + 1;
                llm_pos_ids_list.push_back(expandToMatrix(generateRange(audio_len, st_idx), 3));

                if(llm_pos_ids_list.size()>0){
                    st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
                }else{
                    st_idx = 0;
                }

                eos_len = 1;
                llm_pos_ids_list.push_back(expandToMatrix(generateRange(eos_len, st_idx), 3));

                st += text_len + bos_len + audio_len + eos_len;
                audio_idx += 1;
                remain_audios -= 1;
            }
            else if(min_ed == ed_image){
                text_len = min_ed -st - 1;
                if(text_len!=0){
                    if(llm_pos_ids_list.size()>0){
                        st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
                    }else{
                        st_idx = 0;
                    }

                    llm_pos_ids_list.push_back(expandToMatrix(generateRange(text_len, st_idx), 3));
                }

                if(llm_pos_ids_list.size()>0){
                    st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
                }else{
                    st_idx = 0;
                }

                bos_len = 1;
                llm_pos_ids_list.push_back(expandToMatrix(generateRange(bos_len, st_idx), 3));

                if(llm_pos_ids_list.size()>0){
                    st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
                }else{
                    st_idx = 0;
                }

           
                t = image_grid_thw[image_idx][0];
                h = image_grid_thw[image_idx][1];
                w = image_grid_thw[image_idx][2];

                auto t_idx = generateRange(t, 0, position_id_per_seconds);
                auto llm_pos_ids = get_llm_pos_ids_for_vision(st_idx, spatial_merge_size, t_idx, h, w);
                llm_pos_ids_list.push_back(llm_pos_ids);

                int image_len = t*h*w / (spatial_merge_size*spatial_merge_size);

                if(llm_pos_ids_list.size()>0){
                    st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
                }else{
                    st_idx = 0;
                }

                eos_len = 1;
                llm_pos_ids_list.push_back(expandToMatrix(generateRange(eos_len, st_idx), 3));

                st += text_len + bos_len + image_len + eos_len;
                image_idx += 1;
                remain_images -= 1;

            }
            else if(min_ed == ed_video && !use_audio_in_video){
                text_len = min_ed - st - 1;
                if(text_len!=0){
                    if(llm_pos_ids_list.size()>0){
                        st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
                    }else{
                        st_idx = 0;
                    }

                    llm_pos_ids_list.push_back(expandToMatrix(generateRange(text_len, st_idx), 3));
                }

                if(llm_pos_ids_list.size()>0){
                    st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
                }else{
                    st_idx = 0;
                }

                bos_len = 1;
                llm_pos_ids_list.push_back(expandToMatrix(generateRange(bos_len, st_idx), 3));

                if(llm_pos_ids_list.size()>0){
                    st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
                }else{
                    st_idx = 0;
                }

                t = video_grid_thw[video_idx][0];
                h = video_grid_thw[video_idx][1];
                w = video_grid_thw[video_idx][2];
                
                auto t_idx = generateRange(t, 0, second_per_grids[video_idx]*position_id_per_seconds);
                auto llm_pos_ids = get_llm_pos_ids_for_vision(st_idx, spatial_merge_size, t_idx, h, w);
                llm_pos_ids_list.push_back(llm_pos_ids);

                int video_len = t*h*w / (spatial_merge_size*spatial_merge_size);

                if(llm_pos_ids_list.size()>0){
                    st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
                }else{
                    st_idx = 0;
                }

                eos_len = 1;
                llm_pos_ids_list.push_back(expandToMatrix(generateRange(eos_len, st_idx), 3));

                st += text_len + bos_len + video_len + eos_len;
                video_idx += 1;
                remain_videos -= 1;
            }
            else if(min_ed == ed_video && use_audio_in_video){
                text_len = min_ed - st - 2;
                if(text_len!=0){
                    if(llm_pos_ids_list.size()>0){
                        st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
                    }else{
                        st_idx = 0;
                    }

                    llm_pos_ids_list.push_back(expandToMatrix(generateRange(text_len, st_idx), 3));
                }

                if(llm_pos_ids_list.size()>0){
                    st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
                }else{
                    st_idx = 0;
                }

                bos_len = 1;
                llm_pos_ids_list.push_back(expandToMatrix(generateRange(bos_len, st_idx), 3));
                llm_pos_ids_list.push_back(expandToMatrix(generateRange(bos_len, st_idx), 3));

                if(llm_pos_ids_list.size()>0){
                    st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
                }else{
                    st_idx = 0;
                }

                audio_len = int((int((audio_seqlens[audio_idx] - 1) / 2) + 1 - 2) / 2) + 1;
                auto audio_llm_pos_ids = expandToMatrix(generateRange(audio_len, st_idx), 3);

                t = video_grid_thw[video_idx][0];
                h = video_grid_thw[video_idx][1];
                w = video_grid_thw[video_idx][2];
                
                auto t_idx = generateRange(t, 0, second_per_grids[video_idx]*position_id_per_seconds);
                auto video_llm_pos_ids = get_llm_pos_ids_for_vision(st_idx, spatial_merge_size, t_idx, h, w);
      
                int t_ntoken_per_chunk = position_id_per_seconds * seconds_per_chunk;
                auto video_chunk_indexes = get_chunked_index(video_llm_pos_ids[0], t_ntoken_per_chunk, st_idx);
                auto audio_chunk_indexes = get_chunked_index(audio_llm_pos_ids[0], t_ntoken_per_chunk, st_idx);

                int sub_len = 0;
                int maxlen_chunk_idxes;
                if(video_chunk_indexes.size()>audio_chunk_indexes.size()){
                    maxlen_chunk_idxes = video_chunk_indexes.size();
                }else{
                    maxlen_chunk_idxes = audio_chunk_indexes.size();
                }

                for(int j=0; j<maxlen_chunk_idxes; j++){
                    if(j < video_chunk_indexes.size()){
                        sub_len += video_chunk_indexes[j].second - video_chunk_indexes[j].first;

                        std::vector<std::vector<int>> sliceResult;
                        for (const auto& row : video_llm_pos_ids) {
                            int start = video_chunk_indexes[j].first;
                            int end = video_chunk_indexes[j].second;
                            std::vector<int> rowSlice(row.begin() + start, row.begin() + end);
                            sliceResult.push_back(rowSlice);
                        }

                        llm_pos_ids_list.push_back(sliceResult);
                    }

                    if(j < audio_chunk_indexes.size()){
                        sub_len += audio_chunk_indexes[j].second - audio_chunk_indexes[j].first;

                        std::vector<std::vector<int>> sliceResult;
                        for (const auto& row : audio_llm_pos_ids) {
                            int start = audio_chunk_indexes[j].first;
                            int end = audio_chunk_indexes[j].second;
                            std::vector<int> rowSlice(row.begin() + start, row.begin() + end);
                            sliceResult.push_back(rowSlice);
                        }

                        llm_pos_ids_list.push_back(sliceResult);
                    }
                }

                int video_len = t*h*w / (spatial_merge_size*spatial_merge_size);

                if(llm_pos_ids_list.size()>0){
                    st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
                }else{
                    st_idx = 0;
                }

                eos_len = 1;
                llm_pos_ids_list.push_back(expandToMatrix(generateRange(eos_len, st_idx), 3));
                llm_pos_ids_list.push_back(expandToMatrix(generateRange(eos_len, st_idx), 3));

                st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2;

                audio_idx += 1;
                video_idx += 1;
                remain_videos -= 1;
                remain_audios -= 1;
            }

        }    

        if(st < filtered_ids.size()){
            if(llm_pos_ids_list.size()>0){
                st_idx = findMaxIn2DVector(llm_pos_ids_list.back()) + 1;
            }else{
                st_idx = 0;
            }

            int text_len = filtered_ids.size() - st;
            llm_pos_ids_list.push_back(expandToMatrix(generateRange(text_len, st_idx), 3));
            
        }

        for(auto & item : llm_pos_ids_list){
            for(size_t pi=0; pi<position_ids.size();pi++){
                position_ids[pi].insert(position_ids[pi].end(), item[pi].begin(), item[pi].end());   
            }
        }
        
    return position_ids;
}