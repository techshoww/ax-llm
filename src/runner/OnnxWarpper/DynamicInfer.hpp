#include "onnxruntime_cxx_api.h"
#include <vector>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

/**
 * @brief ONNX模型推理基类
 * 提供通用的ONNX模型加载和推理功能
 */
class ONNXModelBase {
protected:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<char *> input_names_;
    std::vector<char *> output_names_;
    std::string model_name_;

public:
    ONNXModelBase(const std::string& model_path, const std::string& model_name = "Unknown")
        : env_(ORT_LOGGING_LEVEL_WARNING, "ONNXModel"), model_name_(model_name) {
        
        // 配置会话选项
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // 创建会话
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        
        // 获取输入输出节点名称
        setup_node_names();
    }
    
    virtual ~ONNXModelBase() = default;
    
    /**
     * @brief 纯虚函数 - 子类必须实现具体的推理逻辑
     */
    virtual std::vector<float> inference(const std::vector<float>& input_data) = 0;
    
    /**
     * @brief 获取模型名称
     */
    const std::string& get_model_name() const { return model_name_; }
    
    /**
     * @brief 获取输入形状信息
     */
    virtual void get_input_shape(int& batch_size, int& dim, int& length) const = 0;
    
    /**
     * @brief 获取输出形状信息
     */
    virtual void get_output_shape(int& batch_size, int& dim, int& length) const = 0;

protected:
    /**
     * @brief 设置输入输出节点名称（基类实现）
     */
    void setup_node_names() {
        Ort::AllocatorWithDefaultOptions allocator;
        
        // 获取输入名称
        size_t input_count = session_->GetInputCount();
        for (size_t i = 0; i < input_count; i++) {
            input_names_.push_back(session_->GetInputNameAllocated(i, allocator).get());
            ALOGI("input name %s",session_->GetInputNameAllocated(i, allocator).get());
        }
        
        // 获取输出名称
        size_t output_count = session_->GetOutputCount();
        for (size_t i = 0; i < output_count; i++) {
            output_names_.push_back(session_->GetOutputNameAllocated(i, allocator).get());
            ALOGI("output name %s",session_->GetOutputNameAllocated(i, allocator).get());
        }
    }
    
    /**
     * @brief 通用的ONNX推理执行函数
     */
    std::vector<float> run_inference(const std::vector<float>& input_data,
                                   const std::vector<int64_t>& input_shape,
                                   const std::vector<int64_t>& expected_output_shape) {
        
        // 验证输入数据大小
        size_t expected_input_size = 1;
        for (auto dim : input_shape) expected_input_size *= dim;
        
        if (input_data.size() != expected_input_size) {
            throw std::invalid_argument(model_name_ + " input data size mismatch. Expected: " + 
                                      std::to_string(expected_input_size) + 
                                      ", Got: " + std::to_string(input_data.size()));
        }
        
        // 创建内存信息
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtDeviceAllocator, OrtMemTypeDefault);
        
        // 创建输入Tensor（注意：需要移除const限定符）
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input_data.data()),
            input_data.size(),
            input_shape.data(),
            input_shape.size()
        );
        
        // 运行推理
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), 
            &input_tensor, 
            1,
            output_names_.data(), 
            output_names_.size()
        );
        
        // 处理输出
        if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
            throw std::runtime_error(model_name_ + " inference failed: invalid output");
        }
        
        // 获取输出数据
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        size_t output_size = tensor_info.GetElementCount();
        
        // 验证输出形状（如果提供了预期形状）
        if (!expected_output_shape.empty()) {
            auto actual_shape = tensor_info.GetShape();
            if (actual_shape.size() != expected_output_shape.size()) {
                throw std::runtime_error(model_name_ + " output shape dimension mismatch");
            }
            for (size_t i = 0; i < actual_shape.size(); i++) {
                if (expected_output_shape[i] != -1 &&  // -1表示动态维度
                    actual_shape[i] != expected_output_shape[i]) {
                    throw std::runtime_error(model_name_ + " output shape mismatch at dimension " + 
                                           std::to_string(i));
                }
            }
        }
        
        return std::vector<float>(output_data, output_data + output_size);
    }
};