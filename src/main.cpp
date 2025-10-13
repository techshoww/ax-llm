#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>
#include <atomic>
#include <chrono> // For simulation delays
#include <random> // For simulation data
#include <opencv2/opencv.hpp>
#include "signal.h"
#include "runner/VoxCPM.hpp"
#include "runner/utils/slice_3d.h"
#include "runner/utils/wav.hpp"
#include "runner/utils/timer.hpp"
#include "cmdline.hpp"
#include "runner/utils/files.hpp"

static VoxCPM voxcpm;

// --- Shared State ---
WavBuffer g_wav_buffer;              // Shared buffer for tokens
std::mutex g_buffer_mutex;               // Mutex to protect the buffer
std::condition_variable g_buffer_cv;     // Condition variable for waiting/notifying
std::atomic<bool> g_llm_finished{false}; // Flag to signal LLM completion
std::atomic<bool> g_stop{false};
// --- Constants ---
const size_t MAX_BUFFER_SIZE = 100; // Optional: Limit buffer size to prevent unbounded growth

void __sigExit(int iSigNo)
{
    voxcpm.Stop();
    g_stop = true;
    return;
}

// void simulate_llm()
// {
//     std::vector<int> tokens;
//     readtxt("../../model_convert/llm_out_tokens.txt", tokens);

//     std::cout << "[LLM Thread] Starting to generate tokens...\n";

//     // Simulate generating a stream of tokens
//     for (int &token : tokens)
//     {
//         // Simulate time taken to generate a token
//         // std::this_thread::sleep_for(std::chrono::milliseconds(30));

//         {
//             // Acquire lock before modifying the shared buffer
//             std::lock_guard<std::mutex> lock(g_buffer_mutex);

//             // Optional: Backpressure - wait if buffer is full
//             // This prevents the LLM from running too far ahead.
//             // g_buffer_cv.wait(lock, [] { return g_wav_buffer.size() < MAX_BUFFER_SIZE; });

//             // Add the generated token(s) to the buffer
//             g_wav_buffer.push_back(token); // Add one token
//             // Or add a batch: for(...) g_wav_buffer.push_back(...);

//             std::cout << "[LLM Thread] Generated token " << g_wav_buffer.back()
//                       << " (Buffer size: " << g_wav_buffer.size() << ")\n";
//         } // Lock is automatically released here

//         // Notify the consumer (token2wav) that new data might be available
//         g_buffer_cv.notify_one();
//     }

//     // Signal that LLM generation is finished
//     g_llm_finished = true;
//     std::cout << "[LLM Thread] Finished generating tokens.\n";

//     // Final notify to wake up the consumer if it's waiting
//     g_buffer_cv.notify_all();
// }

void reset()
{
    g_llm_finished = false;
    WavBuffer().swap(g_wav_buffer);
}

int tts(const std::string &text, const std::string &prompt_text, const std::string &prompt_wav_path,
        float cfg_value=2.0, int inference_timesteps=10, int max_length=4096)
{
    std::vector<float> output;
    timer time_total;
    time_total.start();
    try
    {
        // Lambda to capture the LLM instance and shared resources
        // This makes it easy to pass them to the thread
        auto generate_thread_func = [&voxcpm,  &g_wav_buffer, &g_buffer_mutex, &g_buffer_cv, &g_llm_finished, &text, &prompt_text, &prompt_wav_path, &cfg_value, &inference_timesteps, &max_length]()
        {
            voxcpm.GenerateStreaming(g_wav_buffer, g_buffer_mutex, g_buffer_cv, g_llm_finished, text, prompt_text, prompt_wav_path, cfg_value, inference_timesteps, max_length);
        };

        // Start the LLM in a separate thread
        std::thread generate_thread(generate_thread_func);

        int token_offset = 0;
        int i = 0;
        std::vector<float> wav;
        while (true)
        {
            std::unique_lock<std::mutex> lock(g_buffer_mutex);

            // Wait until there are enough tokens OR LLM has finished
            // The lambda is the predicate that must be true for wait to stop waiting.
            g_buffer_cv.wait(lock, [&]
                             { return (g_wav_buffer.size() - token_offset >= 1) ||
                                      g_llm_finished.load() ||
                                      g_stop.load(); });

            if (g_stop)
            {
                lock.unlock();
                break;
            }
            // Check if we should process based on threshold or if LLM is finished
            else if (g_wav_buffer.size() - token_offset >= 1)
            {
                for(int i=token_offset; i<g_wav_buffer.size(); i++)
                {
                    wav.insert(wav.end(), g_wav_buffer[i].begin(), g_wav_buffer[i].end());
                }
                token_offset = g_wav_buffer.size();

                // --- End of Critical Section ---

                // Release the lock while processing, allowing LLM to produce more tokens
                lock.unlock();

                // --- Simulate Token2Wav Processing ---
                std::cout << "[Main Thread] Generated audio length " << wav.size() << " ...\n";
               
                // ALOGI("token2wav use time %.3f ms", t_t2v.cost());

                // TODO: 另起一个线程处理生成的音频
                output.insert(output.end(), wav.begin(), wav.end());
                std::string path = "output_" + std::to_string(i) + ".wav";

                saveVectorAsWavFloat(wav, path, 16000, 1);
                wav.clear();
                i++;
            }

            else if (g_llm_finished.load())
            {
                std::cout << "[Main/Token2Wav Thread] Buffer is empty and LLM finished. Exiting.\n";
                lock.unlock();
                break;
            }
            // Check exit condition: Buffer is empty and LLM is done
            else
            {
                // This else branch is technically not needed because the wait condition
                // ensures we only get here if one of the conditions is true.
                // But it's good practice to structure logic clearly.
                // In this specific loop, we will always process if we wake up.
                lock.unlock(); // Make sure to unlock if not processing
            }
        }

        // Wait for the LLM thread to finish
        if (generate_thread.joinable())
        {
            generate_thread.join();
        }

        if (g_stop)
        {
            WavBuffer().swap(g_wav_buffer);
            return 1;
        }

        
        // TODO: 另起一个线程处理生成的音频
        if(token_offset < g_wav_buffer.size())
        {
            for(int j=token_offset; j<g_wav_buffer.size(); j++)
            {
                wav.insert(wav.end(), g_wav_buffer[j].begin(), g_wav_buffer[j].end());
            }
            output.insert(output.end(), wav.begin(), wav.end());
            std::string path = "output_" + std::to_string(i) + ".wav";
            saveVectorAsWavFloat(wav, path, 16000, 1);
        }
        
        saveVectorAsWavFloat(output, "output.wav", 16000, 1);

        ALOGI("tts total use time: %.3f s", time_total.cost() / 1000);
        reset();
        std::cout << "\nVoice generation pipeline completed.\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error in pipeline: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int main(int argc, char *argv[])
{
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, __sigExit);
    VoxCPMConfig config;
    std::string text = "君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。";
    bool b_continue = true;

    cmdline::parser cmd;
    cmd.add<std::string>("text", 't', "text", true, text);
    cmd.add<std::string>("prompt_text", 0, "prompt text", false, "");
    cmd.add<std::string>("prompt_wav_path", 0, "prompt wav path", false, "");
    cmd.add<std::string>("dir_axmodels", 0, "axmodel directory", false, "axmodels");
    cmd.add<std::string>("dir_base_lm", 0, "base_lm directory", false, "base_lm-axmodels");
    cmd.add<std::string>("dir_residual_lm", 0, "residual_lm dierectory", false, "residual_lm-axmodels");
    cmd.add<std::string>("dir_feat_encoder", 0, "feat_encoder directory", false, "feat_encoder_encoder-axmodels");
    cmd.add<std::string>("dir_decoder_estimator", 0, "decoder estimator directory", false, "feat_decoder_estimator_decoder-axmodels");
    cmd.add<std::string>("url_tokenizer", 0, "tokenizer url", true, "http://127.0.0.1:12345");
    cmd.add<int>("n_timesteps", 'ts', "num of time steps", false, 10);
    cmd.add<float>("cfg_value", 'cv', "LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse", false, 2.0);
    cmd.add<bool>("continue", 0, "continuous dialogue", false, b_continue);

    cmd.parse_check(argc, argv);

    text = cmd.get<std::string>("text");
    std::string prompt_text = cmd.get<std::string>("prompt_text");
    std::string prompt_wav_path = cmd.get<std::string>("prompt_wav_path");
    config.dir_axmodels = cmd.get<std::string>("dir_axmodels");
    config.dir_base_lm = cmd.get<std::string>("dir_base_lm");
    config.dir_residual_lm = cmd.get<std::string>("dir_residual_lm");
    config.dir_feat_encoder = cmd.get<std::string>("dir_feat_encoder");
    config.dir_decoder_estimator = cmd.get<std::string>("dir_decoder_estimator");
    config.url_tokenizer = cmd.get<std::string>("url_tokenizer");
    int n_timesteps = cmd.get<int>("n_timesteps");
    float cfg_value = cmd.get<float>("cfg_value");
    b_continue = cmd.get<bool>("continue");

    if (!voxcpm.Init(config))
    {
        return -1;
    }

    if (text.size() > 0)
    {
        tts(text, prompt_text, prompt_wav_path, cfg_value, n_timesteps);
    }

    if (b_continue)
    {
        printf("Type \"q\" to exit, Ctrl+c to stop current running\n");
    }

    while (b_continue)
    {
        if (g_stop)
        {
            break;
        }

        printf("text >> ");
        fflush(stdout);
        std::getline(std::cin, text);
        if (text == "q")
        {
            break;
        }
        if (text == "")
        {
            continue;
        }

        fflush(stdout);

        tts(text, prompt_text, prompt_wav_path, cfg_value, n_timesteps);
    }

    voxcpm.Deinit();
    return 0;
}