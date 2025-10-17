#pragma once

#include <fstream>
#include <vector>
#include <iomanip>
#include <type_traits> // 用于类型检查
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <type_traits>
#include <stdexcept> // 用于异常处理

// 函数模板：支持任意元素类型的vector
template <typename T>
void savetxt(const std::string& filename, 
            const std::vector<T>& data,
            char delimiter = ' ', 
            int precision = 6)  // 默认精度调整为6位
{
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }

    // 保存流的原始格式状态
    std::ios_base::fmtflags original_flags = outfile.flags();

    // 仅对浮点类型启用科学计数法
    if constexpr (std::is_floating_point_v<T>) {
        outfile << std::scientific << std::setprecision(precision);
    }

    // 优化输出：先输出第一个元素，再循环输出后续元素
    if (!data.empty()) {
        outfile << data[0];
        for (size_t i = 1; i < data.size(); ++i) {
            outfile << delimiter << data[i];
        }
    }

    // 恢复流的原始格式状态
    outfile.flags(original_flags);
    outfile.close();
}



// 函数模板：读取文本文件到 vector<T>
template <typename T>
int readtxt(const std::string& filename, std::vector<T>& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "错误：无法打开文件 " << filename << std::endl;
        return -1;
    }

    data.clear();
    std::string line;
    while (std::getline(file, line)) {
        // 跳过空行
        if (line.empty()) continue;

        std::istringstream iss(line);
        T value;
        // 解析行中的每个数值
        while (iss >> value) {
            data.push_back(value);
            // 跳过分隔符（兼容逗号、分号等）
            if (iss.peek() == ',' || iss.peek() == ';') iss.ignore();
        }
    }
    file.close();
    return 0;
}

std::vector<float> linspace(float start, float end, std::size_t num_steps) {
    if (num_steps == 0) {
        return {}; // 返回空向量
    }
    if (num_steps == 1) {
        return {start}; // 如果只需要一个点，返回起始值
    }

    std::vector<float> result(num_steps);
    float step_size = (end - start) / (num_steps - 1); // 计算步长

    for (std::size_t i = 0; i < num_steps; ++i) {
        result[i] = start + i * step_size;
    }

    // 确保最后一个值精确等于 end，避免浮点数精度带来的误差
    result.back() = end;

    return result;
}

/**
 * @brief 对存储在一维vector中的二维张量进行转置操作 (transpose(0,1))
 * @tparam T 元素类型（支持int, float, double等）
 * @param input 输入的一维vector，按行优先存储二维数据
 * @param rows 原张量的行数
 * @param cols 原张量的列数
 * @return std::vector<T> 转置后的一维vector，按行优先存储
 * @throws std::invalid_argument 当输入参数不合法时抛出异常
 */
template<typename T>
std::vector<T> transposeVector(const std::vector<T>& input, int rows, int cols) {
    // 1. 检查输入参数的有效性
    if (input.empty()) {
        throw std::invalid_argument("Input vector is empty.");
    }
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Rows and cols must be positive integers.");
    }
    if (input.size() != static_cast<size_t>(rows * cols)) {
        throw std::invalid_argument("Input vector size does not match the given dimensions.");
    }

    // 2. 创建结果vector，大小为 cols * rows
    std::vector<T> result(cols * rows);

    // 3. 执行转置：遍历原矩阵的每个元素，计算其在新矩阵中的位置
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // 原矩阵中(i, j)元素的索引
            int original_index = i * cols + j;
            // 转置后，该元素应放在新矩阵的(j, i)位置
            int transposed_index = j * rows + i;
            // 赋值操作
            result[transposed_index] = input[original_index];
        }
    }

    return result;
}

/**
 * @brief 实现类似 einops.rearrange 的功能，将四维张量从 "b t p d" 重排为 "b d (t p)"
 * @param input 输入的一维向量，按行优先存储四维张量数据
 * @param B 批次大小（b 维度）
 * @param T 时间步长或序列长度（t 维度）
 * @param P 图像块大小（p 维度）
 * @param D 特征维度（d 维度）
 * @return std::vector<float> 重排后的一维向量
 */
std::vector<float> rearrangeVector(const std::vector<float>& input, 
                                   int B, int T, int P, int D) {
    // 参数校验
    if (input.empty()) {
        throw std::invalid_argument("Input vector is empty.");
    }
    if (B <= 0 || T <= 0 || P <= 0 || D <= 0) {
        throw std::invalid_argument("All dimensions must be positive integers.");
    }
    if (input.size() != static_cast<size_t>(B * T * P * D)) {
        throw std::invalid_argument("Input vector size does not match the given dimensions.");
    }

    // 输出张量的形状为 [B, D, T*P]
    int TP = T * P;  // 合并后的维度大小
    std::vector<float> output(B * D * TP);

    // 执行重排操作
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int p = 0; p < P; ++p) {
                for (int d = 0; d < D; ++d) {
                    // 计算在输入向量中的索引 (b, t, p, d)
                    int input_index = b * (T * P * D) + t * (P * D) + p * D + d;
                    
                    // 计算在输出向量中的索引 (b, d, t*P + p)
                    int output_index = b * (D * TP) + d * TP + (t * P + p);
                    
                    // 赋值
                    output[output_index] = input[input_index];
                }
            }
        }
    }

    return output;
}
