#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include "httplib.h"
#include "http_utils.hpp"
#include "json.hpp"

class MiniCPMClient {
private:
    std::shared_ptr<httplib::Client> cli;

public:
    bool Init(const std::string& base_url) {
        cli = std::make_shared<httplib::Client>(base_url);
        if (!cli) {
            std::cerr << "Failed to create HTTP client for " << base_url << std::endl;
            return false;
        }

        // 设置超时时间（可选）
        cli->set_connection_timeout(5, 0); // 5秒连接超时
        cli->set_read_timeout(30, 0);      // 30秒读取超时
        cli->set_write_timeout(30, 0);     // 30秒写入超时

        // 测试连接 - 可以尝试访问一个 FastAPI 自动生成的端点，比如 /docs
        // 或者简单地尝试发送一个请求看是否能连接上
        auto res = cli->Get("/openapi.json"); // FastAPI 提供的 API 文档端点
        if (res && res->status == 200) {
            std::cout << "Connected to " << base_url << " successfully (API doc accessible)." << std::endl;
            return true;
        } else {
            std::cerr << "Failed to connect to " << base_url << std::endl;
            if (res) {
                std::cerr << "Status: " << res->status << std::endl;
                std::cerr << "Body (first 200 chars): " << res->body.substr(0, 200) << "..." << std::endl;
            }
            return false;
        }
    }

    void Deinit()
    {}

    // 对应 Python 服务的 /embed_tokens 接口
    int EmbedTokens(const std::vector<int>& input_ids_flat,
                     const std::vector<int>& shape,
                     std::vector<unsigned short>& output_flat, // 修改为 unsigned short
                     std::vector<int>& output_shape) {
        if (!cli) {
            std::cerr << "Client not initialized." << std::endl;
            return -1;
        }

        // 1. 构建 JSON 请求体
        nlohmann::json req_json;
        req_json["input_ids"] = input_ids_flat; // 传递一维的 token IDs
        req_json["shape"] = shape;              // 传递原始的二维形状

        std::string req_body = req_json.dump();
        // std::cout << "Sending /embed_tokens request: " << req_body << std::endl;

        // 2. 发送 POST 请求
        auto res = cli->Post("/embed_tokens", req_body, "application/json");

        if (!res) {
            std::cerr << "HTTP request to /embed_tokens failed." << std::endl;
            return -1;
        }

        if (res->status != 200) {
            std::cerr << "HTTP request to /embed_tokens failed with status: " << res->status << std::endl;
            std::cerr << "Response body: " << res->body << std::endl;
            return -1;
        }

        // std::cout << "Received /embed_tokens response: " << res->body << std::endl;

        // 3. 解析 JSON 响应
        try {
            nlohmann::json res_json = nlohmann::json::parse(res->body);

            // 检查 status 字段
            if (res_json.at("status").get<std::string>() != "success") {
                std::cerr << "Server reported error: " << res_json.at("status").get<std::string>() << std::endl;
                if (res_json.contains("detail")) {
                    std::cerr << "Detail: " << res_json.at("detail").get<std::string>() << std::endl;
                }
                return -1;
            }

            // 获取输出数据 (从 int 转换回 unsigned short)
            std::vector<int> output_as_ints = res_json.at("output").get<std::vector<int>>();
            output_flat.clear();
            output_flat.reserve(output_as_ints.size());
            for (int val : output_as_ints) {
                output_flat.push_back(static_cast<unsigned short>(val)); // 转换回 unsigned short
            }
            output_shape = res_json.at("shape").get<std::vector<int>>();

        } catch (const nlohmann::json::exception& e) {
            std::cerr << "Error parsing JSON response from /embed_tokens: " << e.what() << std::endl;
            std::cerr << "Raw response: " << res->body << std::endl;
            return -1;
        }

        return 0;
    }

    // 对应 Python 服务的 /forward 接口
    // input_embeds_flat 现在是 const std::vector<unsigned short>&，既是输入也是输出
    int Forward(std::vector<unsigned short>& input_embeds_flat, // 修改为引用，允许修改
                 const std::vector<int>& shape,
                 bool is_causal = true) { // 移除 output_flat 和 output_shape 参数
        if (!cli) {
            std::cerr << "Client not initialized." << std::endl;
            return -1;
        }

        // 1. 构建 JSON 请求体，将 unsigned short 向量转换为 int 向量发送
        nlohmann::json req_json;
        std::vector<int> input_as_ints(input_embeds_flat.begin(), input_embeds_flat.end()); // 转换为 int
        req_json["input_embeds"] = input_as_ints;
        req_json["shape"] = shape;
        req_json["is_causal"] = is_causal;

        std::string req_body = req_json.dump();
        // std::cout << "Sending /forward request: " << req_body << std::endl;

        // 2. 发送 POST 请求
        auto res = cli->Post("/forward", req_body, "application/json");

        if (!res) {
            std::cerr << "HTTP request to /forward failed." << std::endl;
            return -1;
        }

        if (res->status != 200) {
            std::cerr << "HTTP request to /forward failed with status: " << res->status << std::endl;
            std::cerr << "Response body: " << res->body << std::endl;
            return -1;
        }

        // std::cout << "Received /forward response: " << res->body << std::endl;

        // 3. 解析 JSON 响应
        try {
            nlohmann::json res_json = nlohmann::json::parse(res->body);

            // 检查 status 字段
            if (res_json.at("status").get<std::string>() != "success") {
                std::cerr << "Server reported error: " << res_json.at("status").get<std::string>() << std::endl;
                if (res_json.contains("detail")) {
                    std::cerr << "Detail: " << res_json.at("detail").get<std::string>() << std::endl;
                }
                return -1;
            }

            // 获取输出数据 (从 int 转换回 unsigned short)
            std::vector<int> output_as_ints = res_json.at("output").get<std::vector<int>>();
            // 检查输出形状是否与输入一致
            std::vector<int> output_shape_from_server = res_json.at("shape").get<std::vector<int>>();
            if (output_shape_from_server != shape) {
                 std::cerr << "Warning: Output shape from server ["; 
                 for (const auto& s : output_shape_from_server) std::cerr << s << " "; 
                 std::cerr << "] does not match input shape ["; 
                 for (const auto& s : shape) std::cerr << s << " "; 
                 std::cerr << "]" << std::endl;
                 // 根据需要决定是否失败
                 // return -1;
            }
            // 将输出结果写回 input_embeds_flat
            input_embeds_flat.clear();
            input_embeds_flat.reserve(output_as_ints.size());
            for (int val : output_as_ints) {
                input_embeds_flat.push_back(static_cast<unsigned short>(val)); // 转换回 unsigned short
            }

        } catch (const nlohmann::json::exception& e) {
            std::cerr << "Error parsing JSON response from /forward: " << e.what() << std::endl;
            std::cerr << "Raw response: " << res->body << std::endl;
            return -1;
        }

        return 0;
    }

    // 对应 Python 服务的 /forward_step 接口
    // input_embeds_flat 现在是 const std::vector<unsigned short>&，既是输入也是输出
    int ForwardStep(std::vector<unsigned short>& input_embeds_flat, // 修改为引用，允许修改
                     const std::vector<int>& shape,
                     int position_id) { // 移除 output_flat 和 output_shape 参数
        if (!cli) {
            std::cerr << "Client not initialized." << std::endl;
            return -1;
        }

        // 1. 构建 JSON 请求体，将 unsigned short 向量转换为 int 向量发送
        nlohmann::json req_json;
        std::vector<int> input_as_ints(input_embeds_flat.begin(), input_embeds_flat.end()); // 转换为 int
        req_json["input_embeds"] = input_as_ints;
        req_json["shape"] = shape;
        req_json["position_id"] = position_id;

        std::string req_body = req_json.dump();
        // std::cout << "Sending /forward_step request: " << req_body << std::endl;

        // 2. 发送 POST 请求
        auto res = cli->Post("/forward_step", req_body, "application/json");

        if (!res) {
            std::cerr << "HTTP request to /forward_step failed." << std::endl;
            return -1;
        }

        if (res->status != 200) {
            std::cerr << "HTTP request to /forward_step failed with status: " << res->status << std::endl;
            std::cerr << "Response body: " << res->body << std::endl;
            return -1;
        }

        // std::cout << "Received /forward_step response: " << res->body << std::endl;

        // 3. 解析 JSON 响应
        try {
            nlohmann::json res_json = nlohmann::json::parse(res->body);

            // 检查 status 字段
            if (res_json.at("status").get<std::string>() != "success") {
                std::cerr << "Server reported error: " << res_json.at("status").get<std::string>() << std::endl;
                if (res_json.contains("detail")) {
                    std::cerr << "Detail: " << res_json.at("detail").get<std::string>() << std::endl;
                }
                return -1;
            }

            // 获取输出数据 (从 int 转换回 unsigned short)
            std::vector<int> output_as_ints = res_json.at("output").get<std::vector<int>>();
            // 检查输出形状是否与输入一致
            std::vector<int> output_shape_from_server = res_json.at("shape").get<std::vector<int>>();
            if (output_shape_from_server != shape) {
                 std::cerr << "Warning: Output shape from server [";
                 for (const auto& s : output_shape_from_server) std::cerr << s << " ";
                 std::cerr << "] does not match input shape [";
                 for (const auto& s : shape) std::cerr << s << " ";
                 std::cerr << "]" << std::endl;
                 // 根据需要决定是否失败
                 // return -1;
            }
            // 将输出结果写回 input_embeds_flat
            input_embeds_flat.clear();
            input_embeds_flat.reserve(output_as_ints.size());
            for (int val : output_as_ints) {
                input_embeds_flat.push_back(static_cast<unsigned short>(val)); // 转换回 unsigned short
            }

        } catch (const nlohmann::json::exception& e) {
            std::cerr << "Error parsing JSON response from /forward_step: " << e.what() << std::endl;
            std::cerr << "Raw response: " << res->body << std::endl;
            return -1;
        }

        return 0;
    }
};

