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
#include "runner/Token2wav.hpp
#include "runner/utils/slice_3d.h"
#include "runner/utils/wav.hpp"
#include "cmdline.hpp"
#include "runner/utils/files.hpp"


static LLM lLaMa;
static Token2Wav lToken2Wav;

// --- Shared State ---
TokenBuffer g_token_buffer;              // Shared buffer for tokens
std::mutex g_buffer_mutex;               // Mutex to protect the buffer
std::condition_variable g_buffer_cv;     // Condition variable for waiting/notifying
std::atomic<bool> g_llm_finished{false}; // Flag to signal LLM completion

// --- Constants ---
const size_t MAX_BUFFER_SIZE = 100;     // Optional: Limit buffer size to prevent unbounded growth


void __sigExit(int iSigNo)
{
    lLaMa.Stop();
    return;
}

// void llm_running_callback(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve)
// {
//     fprintf(stdout, "%s", p_str);
//     fflush(stdout);
// }

int tts(
    // for llm
    std::string & text,
    std::vector<int> prompt_text_embeds,
    std::vector<unsigned short> prompt_speech_embeds,
    // for flow
    std::vector<float32> prompt_feat,
    std::vector<float32> prompt_speech_embeds_flow,
    std::vector<float32> spk_embeds
)
{
    std::vector <float> output;

    try {
        // Lambda to capture the LLM instance and shared resources
        // This makes it easy to pass them to the thread
        auto llm_thread_func = [&lLaMa, &text, &prompt_text_embeds, &prompt_speech_embeds,  &g_token_buffer, &g_buffer_mutex, &g_buffer_cv, &g_llm_finished]() {
            lLaMa.Run(text, prompt_text_embeds, prompt_speech_embeds, g_token_buffer, g_buffer_mutex, g_buffer_cv, g_llm_finished);
        };

        // Start the LLM in a separate thread
        std::thread llm_thread(llm_thread_func);

        int token_offset = 0;
        int prompt_token_len = prompt_speech_embeds_flow.size() / lToken2wav.flow_embed_size;
        int prompt_token_align_len = int(prompt_token_len / lToken2wav.token_hop_len) * lToken2wav.token_hop_len;
        auto prompt_speech_embeds_flow1 = slice_3d_last_dim_from<T>(prompt_speech_embeds_flow, 1, 1, prompt_speech_embeds_flow.size(), prompt_token_align_len * lToken2wav.flow_embed_size);
        auto prompt_feat1 = slice_3d_last_dim_from<T>(prompt_feat, 1, 1, prompt_feat.size(), prompt_token_align_len * 80 * 2);

        int promot_token_pad = 0;
        int this_token_hop_len;
        int i=0;
        while (true) {
            // std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
            this_token_hop_len = (token_offset == 0)? lToken2wav.token_hop_len + promot_token_pad : lToken2wav.token_hop_len;

            std::unique_lock<std::mutex> lock(g_buffer_mutex);

            // Wait until there are enough tokens OR LLM has finished
            // The lambda is the predicate that must be true for wait to stop waiting.
            g_buffer_cv.wait(lock, [] {
                return (g_token_buffer.size() - token_offset >= this_token_hop_len + lToken2Wav.pre_lookahead_len) || \
                        g_llm_finished.load() ;
            });


            // Check if we should process based on threshold or if LLM is finished
            if (g_token_buffer.size() >= this_token_hop_len + lToken2Wav.pre_lookahead_len ) {
                
                // Extract tokens to process
                std::vector<SpeechToken> token;
                int start = token_offset -  std::min( int(token_offset / lToken2Wav.token_hop_len), lToken2Wav.max_infer_chunk_num-1) * lToken2Wav.token_hop_len;
                int end = token_offset + this_token_hop_len + lToken2Wav.pre_lookahead_len;

                std::copy(g_token_buffer.begin() + start, g_token_buffer.begin() + end, token.begin());
                // --- End of Critical Section ---

                // Release the lock while processing, allowing LLM to produce more tokens
                lock.unlock();

                // --- Simulate Token2Wav Processing ---
                std::cout << "[Main/Token2Wav Thread] Processing batch of " << token.size() << " tokens...\n";
              
                audo speech = lToken2Wav.infer(token, prompt_speech_embeds_flow1, prompt_feat1, spk_embeds, token_offset, false);
                token_offset += this_token_hop_len;

                //TODO: 另起一个线程处理生成的音频
                output.insert(output.end(), speech.begin(), speech.end());
                std::string path = "output_"+std::to_string(i)+".wav";
                saveVectorAsWavFloat(speech, path, 24000, 1);
                i += 1;

            } 
            
            elif (g_llm_finished.load() ) {
                std::cout << "[Main/Token2Wav Thread] Buffer is empty and LLM finished. Exiting.\n";
                lock.unlock();
                break;
            }
            // Check exit condition: Buffer is empty and LLM is done
            else {
                // This else branch is technically not needed because the wait condition
                // ensures we only get here if one of the conditions is true.
                // But it's good practice to structure logic clearly.
                // In this specific loop, we will always process if we wake up.
                lock.unlock(); // Make sure to unlock if not processing
            }

        }

        // Wait for the LLM thread to finish
        if (llm_thread.joinable()) {
            llm_thread.join();
        }

        std::vector<SpeechToken> token;
        int start = g_token_buffer.size() - std::min( int(g_token_buffer.size() / lToken2Wav.token_hop_len), lToken2Wav.max_infer_chunk_num-1) * lToken2Wav.token_hop_len;
        std::copy(g_token_buffer.begin() + start, g_token_buffer.end(), token.begin());
        auto speech = lToken2Wav.infer(token, prompt_speech_embeds_flow1, prompt_feat1, spk_embeds, token_offset - start, true);
        //TODO: 另起一个线程处理生成的音频
        output.insert(output.end(), speech.begin(), speech.end());
        std::string path = "output_"+std::to_string(i)+".wav";
        saveVectorAsWavFloat(speech, path, 24000, 1);
        saveVectorAsWavFloat(output, "output.wav", 24000, 1);

        g_token_buffer.erase(g_token_buffer.begin(), g_token_buffer.end());
        std::cout << "\nVoice generation pipeline completed.\n";

    } catch (const std::exception& e) {
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
    std::string text;
    bool b_continue = true;

    cmdline::parser cmd;
    // cmd.add<std::string>("prompt", 'p', "prompt", true, prompt);
    // cmd.add<std::string>("image", 'i', "single image file or .txt file for images list", true);
    cmd.add<std::string>("template_filename_axmodel", 0, "axmodel path template", false, attr.template_filename_axmodel);
    cmd.add<std::string>("filename_post_axmodel", 0, "post axmodel path", false, attr.filename_post_axmodel);
    cmd.add<std::string>("filename_tokenizer_model", 0, "tokenizer model path", false, attr.filename_tokenizer_model);
    cmd.add<std::string>("filename_tokens_embed", 0, "tokens embed path", false, attr.filename_tokens_embed);
    
    cmd.add<bool>("bos", 0, "", false, attr.b_bos);
    cmd.add<bool>("eos", 0, "", false, attr.b_eos);
    cmd.add<int>("axmodel_num", 0, "num of axmodel(for template)", false, attr.axmodel_num);
    // cmd.add<int>("prefill_axmodel_num", 0, "num of axmodel(for template)", true, attr.prefill_axmodel_num);
    // cmd.add<int>("tokens_embed_num", 0, "tokens embed num", false, attr.tokens_embed_num);
    // cmd.add<int>("tokens_embed_size", 0, "tokens embed size", false, attr.tokens_embed_size);

    // cmd.add<bool>("use_topk", 0, "", false, attr.b_use_topk);
    // cmd.add<bool>("use_mmap_load_embed", 0, "it can save os memory", false, attr.b_use_mmap_load_embed);
    // cmd.add<bool>("dynamic_load_axmodel_layer", 0, "it can save cmm memory", false, attr.b_dynamic_load_axmodel_layer);

    // cmd.add<bool>("live_print", 0, "print in live if set true, else print in end", false);
    cmd.add<bool>("continue", 0, "continuous dialogue", false, b_continue);
    cmd.add<std::string>("post_config_path", 0, "post config path", false, attr.post_config_path);

    cmd.parse_check(argc, argv);

    // prompt = cmd.get<std::string>("prompt");
    // auto image_prompt = cmd.get<std::string>("image");
    // attr.tokenizer_type = (TokenizerType)cmd.get<int>("tokenizer_type");
    attr.filename_tokenizer_model = cmd.get<std::string>("filename_tokenizer_model");
    attr.filename_tokens_embed = cmd.get<std::string>("filename_tokens_embed");
    attr.filename_post_axmodel = cmd.get<std::string>("filename_post_axmodel");
    attr.template_filename_axmodel = cmd.get<std::string>("template_filename_axmodel");
    // attr.template_prefill_filename_axmodel = cmd.get<std::string>("template_prefill_filename_axmodel");
    // attr.prefill_axmodel_num = cmd.get<int>("prefill_axmodel_num");

    attr.b_bos = cmd.get<bool>("bos");
    attr.b_eos = cmd.get<bool>("eos");
    // attr.b_use_topk = cmd.get<bool>("use_topk");
    attr.axmodel_num = cmd.get<int>("axmodel_num");
    // attr.tokens_embed_num = cmd.get<int>("tokens_embed_num");
    // attr.tokens_embed_size = cmd.get<int>("tokens_embed_size");

    // attr.b_use_mmap_load_embed = cmd.get<bool>("use_mmap_load_embed");
    // attr.b_dynamic_load_axmodel_layer = cmd.get<bool>("dynamic_load_axmodel_layer");
    attr.post_config_path = cmd.get<std::string>("post_config_path");

    // bool b_live_print = cmd.get<bool>("live_print");
    // if (b_live_print)
    // {
    //     attr.runing_callback = llm_running_callback;
    //     attr.reserve = 0;
    // }

    b_continue = cmd.get<bool>("continue");

    if (!lLaMa.Init(attr))
    {
        return -1;
    }

    if (!lToken2Wav.Init("../model_convert/token2wav-axmodels/"))
    {
        return -1;
    }
    // for llm
    std::vector<int> prompt_text_token;
    std::vector<unsigned short> prompt_text_embeds;
    std::vector<unsigned short> prompt_speech_embeds;

    // for flow
    std::vector<float32> prompt_feat;
    std::vector<float32> prompt_speech_embeds_flow;
    std::vector<float32> spk_embeds;

    lLaMa.Token2Embeds(prompt_text_token, prompt_text_embeds);

    //
    if (b_continue)
    {
        printf("Type \"q\" to exit, Ctrl+c to stop current running\n");
    }

    while (b_continue)
    {
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
            text, prompt_text_embeds,prompt_speech_embeds,
            // for flow
            prompt_feat, prompt_speech_embeds_flow, spk_embeds
        );


    }

    lLaMa.Deinit();
    lToken2Wav.Deinit();

    return 0;
}