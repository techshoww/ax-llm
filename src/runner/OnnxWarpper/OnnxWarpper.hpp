#pragma once

#include "../BaseRunner.hpp"

#include <cstdio>
#include <limits>
#include <stdexcept>

#include "onnxruntime_cxx_api.h"

class OnnxRunner final : public BaseRunner
{
public:
    OnnxRunner() : env_(ORT_LOGGING_LEVEL_WARNING, "cosyvoice") {}

    int load(const BaseConfig &config) override
    {
        try
        {
            if (config.onnx_model.empty())
            {
                return -1;
            }

            Ort::SessionOptions options;
            options.SetIntraOpNumThreads(config.nthread);
            options.SetInterOpNumThreads(config.nthread);
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            session_ = Ort::Session(env_, config.onnx_model.c_str(), options);

            const auto memory_info = Ort::MemoryInfo::CreateCpu(
                OrtDeviceAllocator, OrtMemTypeCPU);
            Ort::AllocatorWithDefaultOptions allocator;
            initialiseTensors(memory_info, allocator, true);
            initialiseTensors(memory_info, allocator, false);
            return 0;
        }
        catch (const std::exception &e)
        {
            std::fprintf(stderr, "ONNX Runtime failed to load '%s': %s\n",
                         config.onnx_model.c_str(), e.what());
            return -1;
        }
    }

    int inference() override
    {
        try
        {
            Ort::RunOptions options;
            session_.Run(options, input_names_cstr_.data(), input_tensors_.data(),
                         input_tensors_.size(), output_names_cstr_.data(),
                         output_tensors_.data(), output_tensors_.size());
            return 0;
        }
        catch (const std::exception &e)
        {
            std::fprintf(stderr, "ONNX Runtime inference failed: %s\n", e.what());
            return -1;
        }
    }

    int getInputCount() const override { return static_cast<int>(input_shapes_.size()); }
    const std::vector<size_t> &getInputShape(int idx) const override { return input_shapes_.at(idx); }
    const std::string &getInputName(int idx) const override { return input_names_.at(idx); }
    float *getInputPtr(int idx) override { return input_data_.at(idx).get(); }

    int getOutputCount() const override { return static_cast<int>(output_shapes_.size()); }
    const std::vector<size_t> &getOutputShape(int idx) const override { return output_shapes_.at(idx); }
    const std::string &getOutputName(int idx) const override { return output_names_.at(idx); }
    const float *getOutputPtr(int idx) const override { return output_data_.at(idx).get(); }

private:
    void initialiseTensors(const Ort::MemoryInfo &memory_info,
                           Ort::AllocatorWithDefaultOptions &allocator,
                           bool input)
    {
        const size_t count = input ? session_.GetInputCount() : session_.GetOutputCount();
        auto &shapes = input ? input_shapes_ : output_shapes_;
        auto &names = input ? input_names_ : output_names_;
        auto &data = input ? input_data_ : output_data_;
        auto &tensors = input ? input_tensors_ : output_tensors_;

        shapes.clear();
        names.clear();
        data.clear();
        tensors.clear();
        shapes.reserve(count);
        names.reserve(count);
        data.reserve(count);
        tensors.reserve(count);

        for (size_t index = 0; index < count; ++index)
        {
            auto allocated_name = input ? session_.GetInputNameAllocated(index, allocator)
                                        : session_.GetOutputNameAllocated(index, allocator);
            names.emplace_back(allocated_name.get());

            auto shape = input
                ? session_.GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape()
                : session_.GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
            size_t elements = 1;
            for (auto &dimension : shape)
            {
                // HIFT P1 exports fixed shapes.  Treat any dynamic dimension as
                // batch 1 so the caller never allocates from a negative value.
                if (dimension < 0)
                {
                    dimension = 1;
                }
                if (dimension == 0 ||
                    elements > std::numeric_limits<size_t>::max() /
                                   static_cast<size_t>(dimension))
                {
                    throw std::runtime_error("invalid ONNX tensor shape");
                }
                elements *= static_cast<size_t>(dimension);
            }

            std::vector<size_t> dimensions;
            dimensions.reserve(shape.size());
            for (const auto dimension : shape)
            {
                dimensions.push_back(static_cast<size_t>(dimension));
            }
            shapes.emplace_back(std::move(dimensions));
            data.emplace_back(new float[elements](), std::default_delete<float[]>());
            tensors.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info, data.back().get(), elements, shape.data(), shape.size()));
        }

        if (input)
        {
            input_names_cstr_.clear();
            input_names_cstr_.reserve(input_names_.size());
            for (const auto &name : input_names_)
            {
                input_names_cstr_.push_back(name.c_str());
            }
        }
        else
        {
            output_names_cstr_.clear();
            output_names_cstr_.reserve(output_names_.size());
            for (const auto &name : output_names_)
            {
                output_names_cstr_.push_back(name.c_str());
            }
        }
    }

    Ort::Env env_;
    Ort::Session session_{nullptr};
    std::vector<std::vector<size_t>> input_shapes_;
    std::vector<std::string> input_names_;
    std::vector<const char *> input_names_cstr_;
    std::vector<std::shared_ptr<float>> input_data_;
    std::vector<Ort::Value> input_tensors_;
    std::vector<std::vector<size_t>> output_shapes_;
    std::vector<std::string> output_names_;
    std::vector<const char *> output_names_cstr_;
    std::vector<std::shared_ptr<float>> output_data_;
    std::vector<Ort::Value> output_tensors_;
};
