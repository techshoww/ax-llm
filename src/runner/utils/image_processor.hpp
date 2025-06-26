
#ifndef _IMAGE_PROCESSOR_H_
#define _IMAGE_PROCESSOR_H_

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

int ReadImages(std::string path, std::vector<cv::Mat>& src);
int Qwen2VideoProcessor( std::vector<cv::Mat>& src, std::vector<std::vector<unsigned char>>& output,
                        int tgt_h, int tgt_w,
                        int temporal_patch_size=2, int merge_size=2, int patch_size=14);

#endif