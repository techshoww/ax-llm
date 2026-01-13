#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>
#include <atomic>
#include <chrono> // For simulation delays
#include <random> // For simulation data
#include "signal.h"
#include "runner/LLM.hpp"
#include "runner/Token2wav.hpp"
#include "runner/utils/slice_3d.h"
#include "runner/utils/wav.hpp"
#include "runner/utils/timer.hpp"
#include "cmdline.hpp"
#include "runner/utils/files.hpp"
#include <axcl.h>

static LLM lLaMa;
static Token2Wav lToken2Wav;

// --- Shared State ---
TokenBuffer g_token_buffer;              // Shared buffer for tokens
std::mutex g_buffer_mutex;               // Mutex to protect the buffer
std::condition_variable g_buffer_cv;     // Condition variable for waiting/notifying
std::atomic<bool> g_llm_finished{false}; // Flag to signal LLM completion
std::atomic<bool> g_stop{false};
// --- Constants ---
const size_t MAX_BUFFER_SIZE = 100; // Optional: Limit buffer size to prevent unbounded growth

void __sigExit(int iSigNo)
{
    lLaMa.Stop();
    g_stop = true;
    return;
}

void simulate_llm()
{
    std::vector<int> tokens;
    readtxt("../CosyVoice/llm_token_ids.txt", tokens);

    std::cout << "[LLM Thread] Starting to generate tokens...\n";

    // Simulate generating a stream of tokens
    for (int &token : tokens)
    {
        // Simulate time taken to generate a token
        // std::this_thread::sleep_for(std::chrono::milliseconds(30));

        {
            // Acquire lock before modifying the shared buffer
            std::lock_guard<std::mutex> lock(g_buffer_mutex);

            // Optional: Backpressure - wait if buffer is full
            // This prevents the LLM from running too far ahead.
            // g_buffer_cv.wait(lock, [] { return g_token_buffer.size() < MAX_BUFFER_SIZE; });

            // Add the generated token(s) to the buffer
            g_token_buffer.push_back(token); // Add one token
            // Or add a batch: for(...) g_token_buffer.push_back(...);

            std::cout << "[LLM Thread] Generated token " << g_token_buffer.back()
                      << " (Buffer size: " << g_token_buffer.size() << ")\n";
        } // Lock is automatically released here

        // Notify the consumer (token2wav) that new data might be available
        // g_buffer_cv.notify_one();
    }

    // Signal that LLM generation is finished
    g_llm_finished = true;
    std::cout << "[LLM Thread] Finished generating tokens.\n";

    // Final notify to wake up the consumer if it's waiting
    // g_buffer_cv.notify_all();
}

void reset()
{
    g_llm_finished = false;
    g_token_buffer.erase(g_token_buffer.begin(), g_token_buffer.end());
    lToken2Wav.reset();
}

int tts(
    // for llm
    std::string &text,
    std::vector<unsigned short> prompt_text_embeds,
    std::vector<unsigned short> prompt_speech_embeds,
    // for flow
    std::vector<float> prompt_feat,
    std::vector<float> prompt_speech_embeds_flow,
    std::vector<float> spk_embeds)
{
    std::vector<float> output;
    timer time_total;
    time_total.start();
    try
    {
        // Lambda to capture the LLM instance and shared resources
        // This makes it easy to pass them to the thread
        auto llm_thread_func = [&lLaMa, &text, &prompt_text_embeds, &prompt_speech_embeds, &g_token_buffer, &g_buffer_mutex, &g_buffer_cv, &g_llm_finished]()
        {
            lLaMa.Run(text, prompt_text_embeds, prompt_speech_embeds, g_token_buffer, g_buffer_mutex, g_buffer_cv, g_llm_finished);
        };

        // Start the LLM in a separate thread
        std::thread llm_thread(llm_thread_func);
        // simulate_llm();

        int token_offset = 0;
        int prompt_token_len = prompt_speech_embeds_flow.size() / lToken2Wav.flow_embed_size;
        if (prompt_token_len < 75)
        {
            ALOGE("Error, prompt speech token len %d < 75", prompt_token_len);
            return -1;
        }
        // int prompt_token_align_len = int(prompt_token_len / lToken2Wav.token_hop_len) * lToken2Wav.token_hop_len;
        int prompt_token_align_len = 75; // only support 75 now

        std::vector<float> prompt_speech_embeds_flow1;
        prompt_speech_embeds_flow1.insert(prompt_speech_embeds_flow1.begin(), prompt_speech_embeds_flow.begin(), prompt_speech_embeds_flow.begin() + prompt_token_align_len * lToken2Wav.flow_embed_size);

        std::vector<float> prompt_feat1;
        prompt_feat1.insert(prompt_feat1.begin(), prompt_feat.begin(), prompt_feat.begin() + prompt_token_align_len * 2 * 80);

        int promot_token_pad = 0;
        int this_token_hop_len;
        int i = 0;
        while (true)
        {
            // std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
            this_token_hop_len = (token_offset == 0) ? lToken2Wav.token_hop_len + promot_token_pad : lToken2Wav.token_hop_len;

            std::unique_lock<std::mutex> lock(g_buffer_mutex);

            // Wait until there are enough tokens OR LLM has finished
            // The lambda is the predicate that must be true for wait to stop waiting.
            g_buffer_cv.wait(lock, [&]
                             { return (g_token_buffer.size() - token_offset >= this_token_hop_len + lToken2Wav.pre_lookahead_len) ||
                                      g_llm_finished.load() ||
                                      g_stop.load(); });
            // g_buffer_cv.wait(lock, [&] {
            //     return  g_llm_finished.load() ||\
            //             g_stop.load();
            // });

            if (g_stop)
            {
                lock.unlock();
                break;
            }
            // Check if we should process based on threshold or if LLM is finished
            else if (g_token_buffer.size() - token_offset >= this_token_hop_len + lToken2Wav.pre_lookahead_len)
            {

                // Extract tokens to process
                std::vector<SpeechToken> token;
                int start = token_offset - std::min(int(token_offset / lToken2Wav.token_hop_len), lToken2Wav.max_infer_chunk_num - 1) * lToken2Wav.token_hop_len;
                int end = token_offset + this_token_hop_len + lToken2Wav.pre_lookahead_len;

                token.insert(token.end(), g_token_buffer.begin() + start, g_token_buffer.begin() + end);
                // --- End of Critical Section ---

                // Release the lock while processing, allowing LLM to produce more tokens
                lock.unlock();

                // --- Simulate Token2Wav Processing ---
                std::cout << "[Main/Token2Wav Thread] Processing batch of " << token.size() << " tokens...\n";
                // timer t_t2v;
                // t_t2v.start();
                auto speech = lToken2Wav.infer(token, prompt_speech_embeds_flow1, prompt_feat1, spk_embeds, token_offset, false);
                token_offset += this_token_hop_len;
                // ALOGI("token2wav use time %.3f ms", t_t2v.cost());

                // TODO: 另起一个线程处理生成的音频
                output.insert(output.end(), speech.begin(), speech.end());
                // std::string path = "output_" + std::to_string(i) + ".wav";
                // saveVectorAsWavFloat(speech, path, 24000, 1);
                i += 1;
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
        if (llm_thread.joinable())
        {
            llm_thread.join();
        }

        if (g_stop)
        {
            g_token_buffer.erase(g_token_buffer.begin(), g_token_buffer.end());
            return 1;
        }

        std::vector<SpeechToken> token;
        int start = g_token_buffer.size() - std::min(int(g_token_buffer.size() / lToken2Wav.token_hop_len), lToken2Wav.max_infer_chunk_num - 1) * lToken2Wav.token_hop_len;
        token.insert(token.end(), g_token_buffer.begin() + start, g_token_buffer.end());
        auto speech = lToken2Wav.infer(token, prompt_speech_embeds_flow1, prompt_feat1, spk_embeds, token_offset - start, true);
        // TODO: 另起一个线程处理生成的音频
        output.insert(output.end(), speech.begin(), speech.end());
        // std::string path = "output_" + std::to_string(i) + ".wav";
        // saveVectorAsWavFloat(speech, path, 24000, 1);
        saveVectorAsWavFloat(output, "output.wav", 24000, 1);

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
    LLMAttrType attr;
    std::string text = "君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。";
    bool b_continue = true;

    cmdline::parser cmd;
    cmd.add<std::string>("text", 't', "text", true, text);
    cmd.add<std::string>("token2wav_axmodel_dir", 0, "token2wav axmodel path template", false, "");
    cmd.add<std::string>("template_filename_axmodel", 0, "axmodel path template", false, attr.template_filename_axmodel);
    cmd.add<std::string>("filename_post_axmodel", 0, "post axmodel path", false, attr.filename_post_axmodel);
    cmd.add<std::string>("filename_decoder_axmodel", 0, "post axmodel path", false, attr.filename_decoder_axmodel);
    cmd.add<std::string>("filename_tokenizer_model", 0, "tokenizer model path", false, attr.filename_tokenizer_model);
    cmd.add<std::string>("filename_tokens_embed", 0, "tokens embed path", false, attr.filename_tokens_embed);
    cmd.add<std::string>("filename_llm_embed", 0, "tokens embed path", false, attr.filename_llm_embed);
    cmd.add<std::string>("filename_speech_embed", 0, "tokens embed path", false, attr.filename_speech_embed);
    cmd.add<std::string>("prompt_files", 0, "prompt files dir", false, "prompt_files");

    cmd.add<bool>("bos", 0, "", false, attr.b_bos);
    cmd.add<bool>("eos", 0, "", false, attr.b_eos);
    cmd.add<int>("axmodel_num", 0, "num of axmodel(for template)", false, attr.axmodel_num);
    cmd.add<int>("n_timesteps", 'ts', "num of time steps", false, 7);
    cmd.add<bool>("continue", 0, "continuous dialogue", false, b_continue);
    cmd.add<std::string>("devices", 0, "devices id,for example: \"0,1,2,3\" ", true, "0,1,2,3");
    cmd.parse_check(argc, argv);

    text = cmd.get<std::string>("text");

    attr.filename_tokenizer_model = cmd.get<std::string>("filename_tokenizer_model");
    attr.filename_tokens_embed = cmd.get<std::string>("filename_tokens_embed");
    attr.filename_llm_embed = cmd.get<std::string>("filename_llm_embed");
    attr.filename_speech_embed = cmd.get<std::string>("filename_speech_embed");
    attr.filename_post_axmodel = cmd.get<std::string>("filename_post_axmodel");
    attr.filename_decoder_axmodel = cmd.get<std::string>("filename_decoder_axmodel");
    attr.template_filename_axmodel = cmd.get<std::string>("template_filename_axmodel");

    attr.b_bos = cmd.get<bool>("bos");
    attr.b_eos = cmd.get<bool>("eos");
    attr.axmodel_num = cmd.get<int>("axmodel_num");
    std::string token2wav_axmodel_dir = cmd.get<std::string>("token2wav_axmodel_dir");
    int n_timesteps = cmd.get<int>("n_timesteps");
    std::string prompt_files = cmd.get<std::string>("prompt_files");

    b_continue = cmd.get<bool>("continue");

    auto devices_str = cmd.get<std::string>("devices");
    std::vector<int> devices;
    std::stringstream ss(devices_str);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        devices.push_back(std::stoi(item));
        ALOGI("device: %d", std::stoi(item));
    }

    // 分别给 Token2Wav和LLM分配devices
    lToken2Wav.devid = devices[ devices.size()-1 ];
    if(devices.size()>1)
    {
        attr.dev_ids.assign(devices.begin(), devices.end()-1);
    }else{
        attr.dev_ids.assign(devices.begin(), devices.end());
    }

    auto ret = axclInit(nullptr);
    if (0 != ret)
    {
        return ret;
    }

    for (auto &devid : devices)
    {
        if (axcl_Init(devid) != 0)
        {
            ALOGE("axcl_Init(%d) failed", devid);
            return -1;
        }
    }

    if (!lLaMa.Init(attr))
    {
        axclFinalize();
        return -1;
    }

    if (!lToken2Wav.Init(token2wav_axmodel_dir, n_timesteps))
    {
        lLaMa.Deinit();
        axclFinalize();
        return -1;
    }
    
    // for llm
    std::vector<int> prompt_text_token;
    std::vector<unsigned short> prompt_text_embeds;
    std::vector<int> prompt_speech_token;
    std::vector<unsigned short> prompt_speech_embeds;

    // for flow
    std::vector<float> prompt_feat;
    std::vector<float> prompt_speech_embeds_flow;
    std::vector<float> spk_embeds;

    readtxt(prompt_files + "/prompt_text.txt", prompt_text_token);
    readtxt(prompt_files + "/llm_prompt_speech_token.txt", prompt_speech_token);
    readtxt(prompt_files + "/prompt_speech_feat.txt", prompt_feat);
    readtxt<float>(prompt_files + "/flow_embedding.txt", spk_embeds);

    lLaMa.TextToken2Embeds(prompt_text_token, prompt_text_embeds);
    lLaMa.SpeechToken2Embeds(prompt_speech_token, prompt_speech_embeds);
    lToken2Wav.SpeechToken2Embeds(prompt_speech_token, prompt_speech_embeds_flow);

    if (text.size() > 0)
    {
        tts(
            // for llm
            text, prompt_text_embeds, prompt_speech_embeds,
            // for flow
            prompt_feat, prompt_speech_embeds_flow, spk_embeds);
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

        tts(
            // for llm
            text, prompt_text_embeds, prompt_speech_embeds,
            // for flow
            prompt_feat, prompt_speech_embeds_flow, spk_embeds);
    }

    lLaMa.Deinit();
    lToken2Wav.Deinit();
    for (auto &devid : devices)
        axcl_Exit(devid);
    axclFinalize();
    return 0;
}