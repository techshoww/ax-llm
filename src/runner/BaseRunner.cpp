#include "BaseRunner.hpp"

#include "OnnxWarpper/OnnxWarpper.hpp"

std::shared_ptr<BaseRunner> CreateRunner(RunnerType rt)
{
    if (rt == RT_OnnxRunner)
    {
        return std::make_shared<OnnxRunner>();
    }
    return nullptr;
}
