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
#include "runner/LLM.hpp"
#include "runner/Token2wav.hpp"
#include "runner/utils/slice_3d.h"
#include "runner/utils/wav.hpp"
#include "runner/utils/timer.hpp"
#include "cmdline.hpp"
#include "runner/utils/files.hpp"

#include "runner/utils/httplib.h"
#include "runner/utils/json.hpp"

static httplib::Server svr;
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
    svr.stop();
    return;
}

void simulate_llm()
{
    std::vector<int> tokens;
    readtxt("../../model_convert/llm_out_tokens.txt", tokens);

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
        g_buffer_cv.notify_one();
    }

    // Signal that LLM generation is finished
    g_llm_finished = true;
    std::cout << "[LLM Thread] Finished generating tokens.\n";

    // Final notify to wake up the consumer if it's waiting
    g_buffer_cv.notify_all();
}

void reset()
{
    g_llm_finished = false;
    g_token_buffer.erase(g_token_buffer.begin(), g_token_buffer.end());
    lToken2Wav.reset();
}

std::string wav_file;
// std::mutex wav_file_mutex;

int tts(
    // for llm
    std::string text,
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
        prompt_speech_embeds_flow1.insert(prompt_speech_embeds_flow1.begin(), prompt_speech_embeds_flow.begin(), prompt_speech_embeds_flow.begin() + prompt_token_align_len * 512);

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
                std::string path = "output_" + std::to_string(i) + ".wav";

                saveVectorAsWavFloat(speech, path, 24000, 1);

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
        std::string path = "output_" + std::to_string(i) + ".wav";
        saveVectorAsWavFloat(speech, path, 24000, 1);
        saveVectorAsWavFloat(output, "output.wav", 24000, 1);

        {
            // std::lock_guard<std::mutex> lock(wav_file_mutex);
            wav_file = "output.wav";
        }

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
    // std::string text = "君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。";
    bool b_continue = true;

    cmdline::parser cmd;
    // cmd.add<std::string>("text", 't', "text", true, text);
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
    cmd.add<int>("n_timesteps", 0, "num of time steps", false, 7);
    cmd.add<bool>("continue", 0, "continuous dialogue", false, b_continue);

    cmd.parse_check(argc, argv);

    // text = cmd.get<std::string>("text");

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

    if (!lLaMa.Init(attr))
    {
        return -1;
    }

    if (!lToken2Wav.Init(token2wav_axmodel_dir, n_timesteps))
    {
        return -1;
    }
    ALOGI();
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

    std::atomic<bool> b_tts_runing = false;
    std::string text;
    svr.Post("/tts", [&](const httplib::Request &req, httplib::Response &res)
             {
        nlohmann::json json = nlohmann::json::parse(req.body);
        text = json["text"];
        ALOGI("tts text: %s", text.c_str());
        if(b_tts_runing)
        {
            ALOGE("tts is running");
            res.set_content("tts is running", "text/plain");
            res.status = 400;
            return;
        }

        std::function<void()> tts_func = [&text, &b_tts_runing, &prompt_text_embeds, &prompt_speech_embeds, &prompt_feat, &prompt_speech_embeds_flow, &spk_embeds]() {
            
            b_tts_runing = true;
            tts(
                // for llm
                text, prompt_text_embeds, prompt_speech_embeds,
                // for flow
                prompt_feat, prompt_speech_embeds_flow, spk_embeds
            );
            b_tts_runing = false;
        };
        std::thread tts_thread(tts_func);
        tts_thread.detach();
        res.set_content("ok", "text/plain"); });

    svr.Post("/stop", [&](const httplib::Request &req, httplib::Response &res)
             {
        lLaMa.Stop();
        g_stop = true;
        res.status = 200; });

    svr.Post("/timesteps", [&](const httplib::Request &req, httplib::Response &res)
             {
                nlohmann::json json = nlohmann::json::parse(req.body);
                int n_timesteps = json["timesteps"];
                ALOGI("timesteps: %d", n_timesteps);
                lToken2Wav.Init(token2wav_axmodel_dir, n_timesteps);
                res.set_content("ok", "text/plain"); });

    svr.Post("/prompt_files", [&](const httplib::Request &req, httplib::Response &res)
             {
                nlohmann::json json = nlohmann::json::parse(req.body);
                std::string prompt_files = json["prompt_files"];

                prompt_text_token.clear();
                prompt_text_embeds.clear();
                prompt_speech_token.clear();
                prompt_speech_embeds.clear();

                // for flow
                prompt_feat.clear();
                prompt_speech_embeds_flow.clear();
                spk_embeds.clear();

                readtxt(prompt_files + "/prompt_text.txt", prompt_text_token);
                readtxt(prompt_files + "/llm_prompt_speech_token.txt", prompt_speech_token);
                readtxt(prompt_files + "/prompt_speech_feat.txt", prompt_feat);
                readtxt<float>(prompt_files + "/flow_embedding.txt", spk_embeds);

                lLaMa.TextToken2Embeds(prompt_text_token, prompt_text_embeds);
                lLaMa.SpeechToken2Embeds(prompt_speech_token, prompt_speech_embeds);
                lToken2Wav.SpeechToken2Embeds(prompt_speech_token, prompt_speech_embeds_flow);


                res.set_content("ok", "text/plain"); });

    svr.Post("/get", [&](const httplib::Request &req, httplib::Response &res)
             {
                ALOGI("get wav files");
                 {
                // std::lock_guard<std::mutex> lock(wav_file_mutex);
                if(b_tts_runing)
                {
                    nlohmann::json json;
                    json["b_tts_runing"] = b_tts_runing.load();
                    res.set_content(json.dump(), "application/json");
                    res.status = 400;
                }
                else
                {
                    nlohmann::json json;
                    json["wav_file"] = wav_file;
                    json["b_tts_runing"] = b_tts_runing.load();
                    res.set_content(json.dump(), "application/json");
                    res.status = 200;
                }
            } });
    int port = 12346;
    ALOGI("api server start in host://0.0.0.0:%d", port);
    ALOGI("Post /tts to start tts thread, json body: {\"text\": \"your text\"}");
    ALOGI("Post /stop to stop tts thread");
    ALOGI("Post /get to get wav files path(files for stream audio)");
    svr.listen("0.0.0.0", port);
    ALOGI("api server stop");
    // if(text.size()>0)
    // {
    //     tts(
    //         // for llm
    //         text, prompt_text_embeds,prompt_speech_embeds,
    //         // for flow
    //         prompt_feat, prompt_speech_embeds_flow, spk_embeds
    //     );
    // }

    // if (b_continue)
    // {
    //     printf("Type \"q\" to exit, Ctrl+c to stop current running\n");
    // }

    // while (b_continue)
    // {
    //     if(g_stop)
    //     {
    //         break;
    //     }

    //     printf("text >> ");
    //     fflush(stdout);
    //     std::getline(std::cin, text);
    //     if (text == "q")
    //     {
    //         break;
    //     }
    //     if (text == "")
    //     {
    //         continue;
    //     }

    //     fflush(stdout);

    //    tts(
    //         // for llm
    //         text, prompt_text_embeds,prompt_speech_embeds,
    //         // for flow
    //         prompt_feat, prompt_speech_embeds_flow, spk_embeds
    //     );

    // }

    lLaMa.Deinit();
    lToken2Wav.Deinit();

    return 0;
}