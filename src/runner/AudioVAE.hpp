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
#include "bfloat16.hpp"
#include "Tokenizer/Tokenizer.hpp"
#include "LLMEmbedSelector.hpp"
#include "ax_model_runner/ax_model_runner_ax650.hpp"
#include "ax_cmm_utils.hpp"
#include "cqdm.h"
#include "timer.hpp"
#include "utils/utils.hpp"
#include "runner/OnnxWarpper/DynamicInfer.hpp"

/**
 * @brief Encoder模型类
 * 输入: [batch_size, dim, length]
 * 输出: [batch_size, 64, length]
 */
class EncoderModel : public ONNXModelBase {
private:
    int _batch_size;
    int _dim_input;
    int _dim_output;
    int _chunk_size;

public:
    EncoderModel(const std::string& model_path, int batch_size, int dim_input, int dim_output, int chunk_size) 
        : ONNXModelBase(model_path, "Encoder"),
        _batch_size(batch_size), _dim_input(dim_input), _dim_output(dim_output), _chunk_size(chunk_size)
    {}
    
    /**
     * @brief Encoder推理实现
     */
    std::vector<float> inference(const std::vector<float>& input_data) override {
        
        int length_input = input_data.size() / (_batch_size * _dim_input);
        int length_output = length_input / _chunk_size;
        std::vector<int64_t> input_shape = {_batch_size, _dim_input, length_input};
        std::vector<int64_t> output_shape = {_batch_size, _dim_output, length_output};
        
        return run_inference(input_data, input_shape, output_shape);
    }
    
    void get_input_shape(int& batch_size, int& dim, int& length) const override
    {
        batch_size = _batch_size;
        dim = _dim_input;
        length = -1;
    }

    void get_output_shape(int& batch_size, int& dim, int& length) const override 
    {
        batch_size = _batch_size;
        dim = _dim_output;
        length = -1;
    }
};

/**
 * @brief Decoder模型类  
 * 输入: [batch_size, dim, length] 
 * 输出: [batch_size, 1, length]
 */
class DecoderModel : public ONNXModelBase {
private:
    private:
    int _batch_size;
    int _dim_input;
    int _dim_output;
    int _chunk_size;

public:
    DecoderModel(const std::string& model_path, int batch_size, int dim_input, int dim_output, int chunk_size) 
        : ONNXModelBase(model_path, "Decoder"),
        _batch_size(batch_size), _dim_input(dim_input), _dim_output(dim_output), _chunk_size(chunk_size)
    {}
    
    /**
     * @brief Decoder推理实现
     */
    std::vector<float> inference(const std::vector<float>& input_data) override {
        
        int length_input = input_data.size() / (_batch_size * _dim_input);
        int length_output = length_input * _chunk_size; 
        std::vector<int64_t> input_shape = {_batch_size, _dim_input, length_input};
        std::vector<int64_t> output_shape = {_batch_size, _dim_output, length_output};
    
        return run_inference(input_data, input_shape, output_shape);
    }
    
    void get_input_shape(int& batch_size, int& dim, int& length) const override
    {
        batch_size = _batch_size;
        dim = _dim_input;
        length = -1;
    }

    void get_output_shape(int& batch_size, int& dim, int& length) const override 
    {
        batch_size = _batch_size;
        dim = _dim_output;
        length = -1;
    }

};


class AudioVAE
{
private:
    
    std::unique_ptr<EncoderModel> encoder;
    std::unique_ptr<DecoderModel> decoder;

public:
    int _sample_rate=16000;
    int hop_length = 2*5*8*8;
    int chunk_size = 2*5*8*8;
    int latent_dim = 64;
    
    bool Init(const std::string &dir_axmodel)
    {
        std::string encoder_path = dir_axmodel + "/" + "audio_vae.encoder.onnx";
        std::string decoder_path = dir_axmodel + "/" + "audio_vae.decoder.onnx";
        encoder = std::make_unique<EncoderModel>(encoder_path, 1, 1, latent_dim, chunk_size);
        decoder = std::make_unique<DecoderModel>(decoder_path, 1, latent_dim, 1, chunk_size);

        return true;
    }

    void Deinit()
    {}

    int Encode(std::vector<float> &output, std::vector<float> &audio_data, int sample_rate=16000)
    {
        ALOGI("audio vae encode");
        if(_sample_rate!=sample_rate)
        {
            ALOGE("Just support sample_rate=%d",_sample_rate);
            return -1;
        }

        output = encoder->inference(audio_data);
        ALOGI("audio vae encode end");
        return 0;
    }

    int Decode(std::vector<float> &output, std::vector<float> &z)
    {
        output = decoder->inference(z);
        return 0;
    }

};