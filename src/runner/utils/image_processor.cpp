#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "files.hpp"
#include "image_processor.hpp"
#include <iostream>
#include <iostream>
#include <vector>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

extern "C" {
// #include <libavformat/avformat.h>
// #include <libswresample/swresample.h>
// #include <libavcodec/avcodec.h>
// #include <libswscale/swscale.h>
// #include <libavutil/imgutils.h>
}

int ReadImages(std::string path, std::vector<cv::Mat>& src){

    if(is_file(path)){
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        src.push_back(img);
    }
    else if(is_directory(path)){
        auto paths = list_files(path);
        
        for(auto &p : paths){
            std::cout<<p<<std::endl;
            cv::Mat img = cv::imread(p, cv::IMREAD_COLOR);
            src.push_back(img);
        }
    }
    else{
        std::cerr << "错误的路径: " << path << std::endl;
        return -1;
    }

    return 0;
}

std::pair<int, int> SmartResize(int height, int width, int factor){
    int h_bar = height/factor;
    int w_bar = width/factor;

    h_bar *= factor;
    w_bar *= factor;
    return {h_bar, w_bar};
}

void normalizeMeanStd(cv::Mat& image) {
    // 确保输入图像是浮点类型（避免整数溢出）
    cv::Mat floatImage;
    image.convertTo(floatImage, CV_32F);  // 转换为32位浮点格式 <button class="citation-flag" data-index="1">

    // 计算均值和标准差
    cv::Scalar mean, stddev;
    cv::meanStdDev(floatImage, mean, stddev);  // 计算均值和标准差 <button class="citation-flag" data-index="2">

    // 避免除以零：如果标准差为0，设置为一个小值（如1e-6）
    for (int i = 0; i < floatImage.channels(); ++i) {
        if (stddev[i] < 1e-6) {
            stddev[i] = 1e-6;
        }
    }

    // 归一化：减去均值并除以标准差
    floatImage -= mean;  // 减去均值 <button class="citation-flag" data-index="4">
    floatImage /= stddev;  // 除以标准差 <button class="citation-flag" data-index="5">

    // 将结果转换回原始数据类型（如8位无符号整数）
    floatImage.convertTo(image, image.type());  // 转换回原始格式 <button class="citation-flag" data-index="6">
}

// std::vector<float> extract_audio_from_video(const char* filename, int target_sample_rate) {
//     // 初始化 FFmpeg
//     avformat_network_init();
//     AVFormatContext* format_ctx = nullptr;
//     if (avformat_open_input(&format_ctx, filename, nullptr, nullptr) != 0) {
//         std::cerr << "Error: Could not open video file" << std::endl;
//         return {};
//     }

//     // 探测流信息
//     if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
//         std::cerr << "Error: Could not find stream info" << std::endl;
//         avformat_close_input(&format_ctx);
//         return {};
//     }

//     // 定位音频流索引
//     int audio_stream_idx = -1;
//     for (int i = 0; i < format_ctx->nb_streams; i++) {
//         if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
//             audio_stream_idx = i;
//             break;
//         }
//     }
//     if (audio_stream_idx == -1) {
//         std::cerr << "Error: No audio stream found" << std::endl;
//         avformat_close_input(&format_ctx);
//         return {};
//     }

//     // 获取解码器并打开
//     AVCodecParameters* codec_params = format_ctx->streams[audio_stream_idx]->codecpar;
//     AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
//     AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
//     avcodec_parameters_to_context(codec_ctx, codec_params);
//     if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
//         std::cerr << "Error: Could not open codec" << std::endl;
//         avformat_close_input(&format_ctx);
//         return {};
//     }

//     // 初始化重采样器（调整采样率/声道）
//     SwrContext* swr_ctx = swr_alloc_set_opts(nullptr,
//         AV_CH_LAYOUT_MONO,                   // 目标声道（单声道）
//         AV_SAMPLE_FMT_FLT,                    // 目标格式（Float PCM）
//         target_sample_rate,                    // 目标采样率（如16000）
//         codec_ctx->channel_layout,            // 源声道布局
//         codec_ctx->sample_fmt,                // 源格式
//         codec_ctx->sample_rate,               // 源采样率
//         0, nullptr);
//     swr_init(swr_ctx);

//     // 读取音频包并解码
//     AVPacket packet;
//     AVFrame* frame = av_frame_alloc();
//     std::vector<float> audio_data;
//     while (av_read_frame(format_ctx, &packet) >= 0) {
//         if (packet.stream_index != audio_stream_idx) continue;
        
//         if (avcodec_send_packet(codec_ctx, &packet) < 0) continue;
//         while (avcodec_receive_frame(codec_ctx, frame) == 0) {
//             // 重采样到目标格式
//             float* buffer;
//             av_samples_alloc((uint8_t**)&buffer, nullptr, 1, frame->nb_samples, AV_SAMPLE_FMT_FLT, 0);
//             int sample_count = swr_convert(swr_ctx, (uint8_t**)&buffer, frame->nb_samples, (const uint8_t**)frame->data, frame->nb_samples);
            
//             // 存储到 vector
//             audio_data.insert(audio_data.end(), buffer, buffer + sample_count);
//             av_freep(&buffer);
//         }
//         av_packet_unref(&packet);
//     }

//     // 清理资源
//     swr_free(&swr_ctx);
//     av_frame_free(&frame);
//     avcodec_free_context(&codec_ctx);
//     avformat_close_input(&format_ctx);
//     return audio_data;
// }


struct VideoTensor {
    std::vector<uint8_t> data;  // 数据布局: T x C x H x W
    int T;                      // 帧数
    int C;                      // 通道数
    int H;                      // 高度
    int W;                      // 宽度
    float sample_fps;           // 采样帧率
};

// VideoTensor read_video_ffmpeg(
//     const std::string& video_path, 
//     int target_frames = -1  // 目标帧数 (-1表示自动计算)
// ) {
//     // 初始化FFmpeg
//     avformat_network_init();
//     AVFormatContext* format_ctx = nullptr;
//     if (avformat_open_input(&format_ctx, video_path.c_str(), nullptr, nullptr) != 0) {
//         throw std::runtime_error("无法打开视频文件");
//     }
//     if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
//         avformat_close_input(&format_ctx);
//         throw std::runtime_error("无法获取流信息");
//     }

//     // 查找视频流
//     int video_stream_idx = -1;
//     for (int i = 0; i < format_ctx->nb_streams; ++i) {
//         if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
//             video_stream_idx = i;
//             break;
//         }
//     }
//     if (video_stream_idx == -1) {
//         avformat_close_input(&format_ctx);
//         throw std::runtime_error("未找到视频流");
//     }

//     // 初始化解码器
//     AVCodecParameters* codec_params = format_ctx->streams[video_stream_idx]->codecpar;
//     const AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
//     AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
//     avcodec_parameters_to_context(codec_ctx, codec_params);
//     if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
//         avcodec_free_context(&codec_ctx);
//         avformat_close_input(&format_ctx);
//         throw std::runtime_error("无法打开解码器");
//     }

//     // 获取视频信息
//     AVStream* stream = format_ctx->streams[video_stream_idx];
//     const int total_frames = stream->nb_frames > 0 ? 
//         stream->nb_frames : 
//         static_cast<int>(stream->duration * av_q2d(stream->time_base) * av_q2d(stream->avg_frame_rate));
//     const float video_fps = av_q2d(stream->avg_frame_rate);

//     // 计算目标帧数
//     const int nframes = (target_frames > 0) ? 
//         std::min(target_frames, total_frames) : 
//         total_frames;
//     std::vector<int> frame_indices;
//     for (int i = 0; i < nframes; ++i) {
//         frame_indices.push_back(static_cast<int>(std::round(
//             i * (total_frames - 1.0) / (nframes - 1.0)
//         )));
//     }

//     // 准备图像转换
//     SwsContext* sws_ctx = sws_getContext(
//         codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
//         codec_ctx->width, codec_ctx->height, AV_PIX_FMT_RGB24,
//         SWS_BILINEAR, nullptr, nullptr, nullptr
//     );

//     // 准备输出张量
//     VideoTensor tensor;
//     tensor.T = nframes;
//     tensor.C = 3;  // RGB
//     tensor.H = codec_ctx->height;
//     tensor.W = codec_ctx->width;
//     tensor.data.resize(nframes * tensor.C * tensor.H * tensor.W);

//     // 解码帧
//     AVFrame* frame = av_frame_alloc();
//     AVFrame* rgb_frame = av_frame_alloc();
//     const int rgb_size = av_image_get_buffer_size(AV_PIX_FMT_RGB24, tensor.W, tensor.H, 1);
//     uint8_t* rgb_buffer = static_cast<uint8_t*>(av_malloc(rgb_size));
//     av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize, rgb_buffer, 
//                          AV_PIX_FMT_RGB24, tensor.W, tensor.H, 1);
    
//     AVPacket packet;
//     int current_frame = 0;
//     size_t next_target = 0;
//     uint8_t* tensor_ptr = tensor.data.data();

//     while (av_read_frame(format_ctx, &packet) >= 0 && next_target < frame_indices.size()) {
//         if (packet.stream_index != video_stream_idx) {
//             av_packet_unref(&packet);
//             continue;
//         }

//         if (avcodec_send_packet(codec_ctx, &packet) < 0) {
//             av_packet_unref(&packet);
//             continue;
//         }

//         while (avcodec_receive_frame(codec_ctx, frame) >= 0) {
//             // 检查是否为需要的帧
//             if (current_frame == frame_indices[next_target]) {
//                 // 转换为RGB
//                 sws_scale(sws_ctx, 
//                           frame->data, frame->linesize, 0, frame->height,
//                           rgb_frame->data, rgb_frame->linesize);
                
//                 // 从HWC转换为CHW
//                 for (int c = 0; c < 3; ++c) {
//                     for (int h = 0; h < tensor.H; ++h) {
//                         const uint8_t* src = rgb_frame->data[0] + h * rgb_frame->linesize[0] + c;
//                         uint8_t* dest = tensor_ptr + 
//                                         c * tensor.H * tensor.W + 
//                                         h * tensor.W;
                        
//                         for (int w = 0; w < tensor.W; ++w) {
//                             dest[w] = src[w * 3];
//                         }
//                     }
//                 }
//                 tensor_ptr += tensor.C * tensor.H * tensor.W;
//                 ++next_target;
//             }
//             ++current_frame;
//             av_frame_unref(frame);
//         }
//         av_packet_unref(&packet);
//     }

//     // 计算采样帧率
//     tensor.sample_fps = (nframes * video_fps) / total_frames;

//     // 清理资源
//     av_free(rgb_buffer);
//     av_frame_free(&rgb_frame);
//     av_frame_free(&frame);
//     avcodec_free_context(&codec_ctx);
//     avformat_close_input(&format_ctx);
//     sws_freeContext(sws_ctx);

//     return tensor;
// }

int Qwen2VideoProcessor( std::vector<cv::Mat>& src, std::vector<std::vector<unsigned char>>& output, 
                            int tgt_h, int tgt_w,
                            int temporal_patch_size, int merge_size, int patch_size){

    if(src.empty()){
        return 0;
    }

    int height = src[0].rows;
    int width = src[0].cols;

    // auto [tgt_h, tgt_w] = SmartResize(height, width, 28);

    cv::Size size(tgt_w, tgt_h);
    std::vector<cv::Mat> imgs_resized;
    
    for(auto& img: src){
        cv::Mat img_rs;
        if(img.cols!=tgt_w || img.rows!=tgt_h){
            cv::resize(img, img_rs, size, 0, 0, cv::INTER_CUBIC);
        }else{
            img_rs = img;
        }
        
        cv::cvtColor(img_rs, img_rs, cv::COLOR_BGR2RGB);
        imgs_resized.push_back(img_rs);
    }
    
    if(imgs_resized.empty()){
        return 0;
    }

    if(imgs_resized.size()%2!=0){
        imgs_resized.push_back(imgs_resized.back());
    }

    std::vector<unsigned char> patches;
    patches.resize( imgs_resized.size()* tgt_w*tgt_h* 3);
    for(size_t i=0; i<imgs_resized.size(); ++i){
        memcpy(patches.data()+i*tgt_w*tgt_h*3, imgs_resized[i].data, tgt_w*tgt_h* 3);
    }

    int grid_t = imgs_resized.size() / temporal_patch_size;
    int channel = imgs_resized[0].channels();
    int grid_h = tgt_h/patch_size;
    int grid_w = tgt_w/patch_size;

    // channel = patches.shape[3]
    // patches = patches.reshape(
    //     grid_t,                     # 0
    //     self.temporal_patch_size,   # 1
    //     grid_h // self.merge_size,  # 2
    //     self.merge_size,            # 3
    //     self.patch_size,            # 4
    //     grid_w // self.merge_size,  # 5
    //     self.merge_size,            # 6
    //     self.patch_size,            # 7
    //     channel                     # 8
    // )   
    // patches = patches.transpose(0, 2, 5, 3, 6, 1, 4, 7, 8 )

    for(size_t d0=0; d0<grid_t; d0++){
        std::vector<unsigned char> out_t;
        for(size_t d2=0; d2<grid_h/merge_size; d2++){
            for(size_t d5=0; d5<grid_w/merge_size; d5++){
                for(size_t d3=0; d3<merge_size; d3++ ){
                    for(size_t d6=0; d6<merge_size; d6++){
                        for(size_t d1=0; d1<temporal_patch_size; d1++){
                            for(size_t d4=0; d4<patch_size; d4++){
                                for(size_t d7=0; d7<patch_size; d7++){
                                    for(size_t d8=0; d8<channel; d8++){
                                        size_t idx = d0*temporal_patch_size*grid_h*patch_size*grid_w*patch_size*channel;
                                        idx += d1*grid_h*patch_size*grid_w*patch_size*channel;
                                        idx += d2*merge_size*patch_size*grid_w*patch_size*channel;
                                        idx += d3*patch_size*grid_w*patch_size*channel;
                                        idx += d4*grid_w*patch_size*channel;
                                        idx += d5*merge_size*patch_size*channel;
                                        idx += d6*patch_size*channel;
                                        idx += d7*channel;
                                        idx += d8;

                                        out_t.push_back(patches[idx]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        output.push_back(out_t);
    }

    // std::vector<size_t> ret={grid_t, grid_h*grid_w, temporal_patch_size*patch_size*patch_size, channel};
    // return ret;
    return 0;

}

