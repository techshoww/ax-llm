#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

enum RunnerType
{
    RT_UNKNOWN,
    RT_OnnxRunner,
    RT_END,
};

struct BaseConfig
{
    std::string onnx_model;
    int nthread = 1;
};

class BaseRunner
{
public:
    virtual ~BaseRunner() = default;

    virtual int load(const BaseConfig &config) = 0;
    virtual int inference() = 0;

    virtual int getInputCount() const = 0;
    virtual const std::vector<size_t> &getInputShape(int idx) const = 0;
    virtual const std::string &getInputName(int idx) const = 0;
    virtual float *getInputPtr(int idx) = 0;

    virtual int getOutputCount() const = 0;
    virtual const std::vector<size_t> &getOutputShape(int idx) const = 0;
    virtual const std::string &getOutputName(int idx) const = 0;
    virtual const float *getOutputPtr(int idx) const = 0;
};

std::shared_ptr<BaseRunner> CreateRunner(RunnerType rt);
