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
#include "runner/Token2wav.hpp

#include "cmdline.hpp"

#include <opencv2/opencv.hpp>

#include "runner/utils/image_processor.hpp"

#include "runner/utils/files.hpp"


static LLM lLaMa;
static Token2Wav lToken2Wav;

// --- Shared State ---
TokenBuffer g_token_buffer;              // Shared buffer for tokens
std::mutex g_buffer_mutex;               // Mutex to protect the buffer
std::condition_variable g_buffer_cv;     // Condition variable for waiting/notifying
std::atomic<bool> g_llm_finished{false}; // Flag to signal LLM completion

// --- Constants ---
const size_t PROCESSING_THRESHOLD = 10; // Minimum tokens needed to trigger processing
const size_t MAX_BUFFER_SIZE = 100;     // Optional: Limit buffer size to prevent unbounded growth

// --- Simulated Modules ---
// Simulates the LLM generating tokens and adding them to the buffer.

// Simulates the token2wav processing tokens from the buffer.
void run_token2wav() {
    std::cout << "[Main/Token2Wav Thread] Starting to process tokens...\n";

    while (true) {
        std::unique_lock<std::mutex> lock(g_buffer_mutex);

        // Wait until there are enough tokens OR LLM has finished
        // The lambda is the predicate that must be true for wait to stop waiting.
        g_buffer_cv.wait(lock, [] {
            return g_token_buffer.size() >= PROCESSING_THRESHOLD || g_llm_finished.load();
        });

        // Check exit condition: Buffer is empty and LLM is done
        if (g_token_buffer.empty() && g_llm_finished.load()) {
            std::cout << "[Main/Token2Wav Thread] Buffer is empty and LLM finished. Exiting.\n";
            break;
        }

        // Check if we should process based on threshold or if LLM is finished
        if (g_token_buffer.size() >= PROCESSING_THRESHOLD || (g_llm_finished.load() && !g_token_buffer.empty())) {
            
            // --- Critical Section: Accessing and Modifying the Buffer ---
            size_t tokens_to_process = g_token_buffer.size(); // Process all if LLM finished
            if (!g_llm_finished.load()) {
                // While LLM is running, process only up to a batch size or what's available
                tokens_to_process = std::min(tokens_to_process, PROCESSING_THRESHOLD);
            }

            // Extract tokens to process
            std::vector<SpeechToken> batch(tokens_to_process);
            for (size_t i = 0; i < tokens_to_process; ++i) {
                batch[i] = g_token_buffer.front();
                g_token_buffer.pop_front();
            }
            // --- End of Critical Section ---

            // Release the lock while processing, allowing LLM to produce more tokens
            lock.unlock();

            // --- Simulate Token2Wav Processing ---
            std::cout << "[Main/Token2Wav Thread] Processing batch of " << batch.size() << " tokens...\n";
            // ... (Your actual token-to-wav conversion logic would go here) ...
            // Simulate processing time
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            std::cout << "[Main/Token2Wav Thread] Finished processing batch.\n";
            // --- End of Simulation ---

            // Re-lock if needed afterwards (not needed in this loop structure)
            // std::lock_guard<std::mutex> lock_again(g_buffer_mutex);

        } else {
            // This else branch is technically not needed because the wait condition
            // ensures we only get here if one of the conditions is true.
            // But it's good practice to structure logic clearly.
            // In this specific loop, we will always process if we wake up.
            lock.unlock(); // Make sure to unlock if not processing
        }
    }
}
void __sigExit(int iSigNo)
{
    lLaMa.Stop();
    return;
}

void llm_running_callback(int *p_token, int n_token, const char *p_str, float token_per_sec, void *reserve)
{
    fprintf(stdout, "%s", p_str);
    fflush(stdout);
}

int tts(
    // for llm
    std::string & text,
    std::vector<int> prompt_text_token;
    std::vector<unsigned short> prompt_speech_embeds;
    // for flow
    std::vector<float32> prompt_feat;
    std::vector<float32> prompt_speech_embeds_flow;
    std::vector<float32> spk_embeds;
)
{
    lLaMa.Token2Embeds(prompt_text_token, prompt_text_embeds);

    lLaMa.Run(text, prompt_text_embeds, prompt_speech_embeds, 
                g_token_buffer,
                g_buffer_mutex,
                g_buffer_cv,
                g_llm_finished
            );
    

    while (true) {
        std::unique_lock<std::mutex> lock(g_buffer_mutex);

        // Wait until there are enough tokens OR LLM has finished
        // The lambda is the predicate that must be true for wait to stop waiting.
        g_buffer_cv.wait(lock, [] {
            return g_token_buffer.size() >= PROCESSING_THRESHOLD || g_llm_finished.load();
        });

        // Check exit condition: Buffer is empty and LLM is done
        if (g_token_buffer.empty() && g_llm_finished.load()) {
            std::cout << "[Main/Token2Wav Thread] Buffer is empty and LLM finished. Exiting.\n";
            break;
        }

        // Check if we should process based on threshold or if LLM is finished
        if (g_token_buffer.size() >= PROCESSING_THRESHOLD || (g_llm_finished.load() && !g_token_buffer.empty())) {
            
            // --- Critical Section: Accessing and Modifying the Buffer ---
            size_t tokens_to_process = g_token_buffer.size(); // Process all if LLM finished
            if (!g_llm_finished.load()) {
                // While LLM is running, process only up to a batch size or what's available
                tokens_to_process = std::min(tokens_to_process, PROCESSING_THRESHOLD);
            }

            // Extract tokens to process
            std::vector<SpeechToken> batch(tokens_to_process);
            for (size_t i = 0; i < tokens_to_process; ++i) {
                batch[i] = g_token_buffer.front();
                g_token_buffer.pop_front();
            }
            // --- End of Critical Section ---

            // Release the lock while processing, allowing LLM to produce more tokens
            lock.unlock();

            // --- Simulate Token2Wav Processing ---
            std::cout << "[Main/Token2Wav Thread] Processing batch of " << batch.size() << " tokens...\n";
            // ... (Your actual token-to-wav conversion logic would go here) ...
            // Simulate processing time
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            std::cout << "[Main/Token2Wav Thread] Finished processing batch.\n";
            // --- End of Simulation ---

            // Re-lock if needed afterwards (not needed in this loop structure)
            // std::lock_guard<std::mutex> lock_again(g_buffer_mutex);

        } else {
            // This else branch is technically not needed because the wait condition
            // ensures we only get here if one of the conditions is true.
            // But it's good practice to structure logic clearly.
            // In this specific loop, we will always process if we wake up.
            lock.unlock(); // Make sure to unlock if not processing
        }
    }

}



int main(int argc, char *argv[])
{
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, __sigExit);
    LLMAttrType attr;
    std::string prompt = "Hi";
    bool b_continue = true;

    cmdline::parser cmd;
    // cmd.add<std::string>("prompt", 'p', "prompt", true, prompt);
    // cmd.add<std::string>("image", 'i', "single image file or .txt file for images list", true);
    cmd.add<std::string>("template_filename_axmodel", 0, "axmodel path template", false, attr.template_filename_axmodel);
    cmd.add<std::string>("filename_post_axmodel", 0, "post axmodel path", false, attr.filename_post_axmodel);
    cmd.add<std::string>("filename_tokenizer_model", 0, "tokenizer model path", false, attr.filename_tokenizer_model);
    cmd.add<std::string>("filename_tokens_embed", 0, "tokens embed path", false, attr.filename_tokens_embed);

    cmd.add<std::string>("filename_image_encoder_axmodedl", 0, "vpm encoder axmodel path", false, attr.filename_image_encoder_axmodedl);

    cmd.add<bool>("bos", 0, "", false, attr.b_bos);
    cmd.add<bool>("eos", 0, "", false, attr.b_eos);
    cmd.add<int>("axmodel_num", 0, "num of axmodel(for template)", false, attr.axmodel_num);
    // cmd.add<int>("prefill_axmodel_num", 0, "num of axmodel(for template)", true, attr.prefill_axmodel_num);
    cmd.add<int>("tokens_embed_num", 0, "tokens embed num", false, attr.tokens_embed_num);
    cmd.add<int>("tokens_embed_size", 0, "tokens embed size", false, attr.tokens_embed_size);

    cmd.add<bool>("use_topk", 0, "", false, attr.b_use_topk);
    cmd.add<bool>("use_mmap_load_embed", 0, "it can save os memory", false, attr.b_use_mmap_load_embed);
    cmd.add<bool>("dynamic_load_axmodel_layer", 0, "it can save cmm memory", false, attr.b_dynamic_load_axmodel_layer);

    cmd.add<bool>("live_print", 0, "print in live if set true, else print in end", false);
	cmd.add<bool>("video", 0, "inputs are video", false);
    cmd.add<bool>("continue", 0, "continuous dialogue", false, b_continue);
    cmd.add<int>("img_width", 'w', "image width", true);
    cmd.add<int>("img_height", 'h', "image height", true);
    cmd.add<int>("img_token_id", 0, "image token id", false, 151655); 
    cmd.add<int>("video_token_id", 0, "video token id", false, 151656);
    cmd.add<int>("vision_start_token_id", 0, "vision_start_token_id", false, 151652);
    
    cmd.add<int>("temporal_patch_size", 0, "temporal_patch_size", false, 2);
    cmd.add<int>("tokens_per_second", 0, "tokens_per_second", false, 2);
    cmd.add<int>("spatial_merge_size", 0, "spatial_merge_size", false, 2);
    cmd.add<int>("patch_size", 0, "patch size", false, 14);
    cmd.add<int>("fps", 0, "fps", false, 1);

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

    attr.filename_image_encoder_axmodedl = cmd.get<std::string>("filename_image_encoder_axmodedl");
    attr.b_bos = cmd.get<bool>("bos");
    attr.b_eos = cmd.get<bool>("eos");
    attr.b_use_topk = cmd.get<bool>("use_topk");
    attr.axmodel_num = cmd.get<int>("axmodel_num");
    attr.tokens_embed_num = cmd.get<int>("tokens_embed_num");
    attr.tokens_embed_size = cmd.get<int>("tokens_embed_size");

    attr.b_use_mmap_load_embed = cmd.get<bool>("use_mmap_load_embed");
    attr.b_dynamic_load_axmodel_layer = cmd.get<bool>("dynamic_load_axmodel_layer");
    attr.post_config_path = cmd.get<std::string>("post_config_path");

    bool b_live_print = cmd.get<bool>("live_print");
    if (b_live_print)
    {
        attr.runing_callback = llm_running_callback;
        attr.reserve = 0;
    }

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
    //
    if (b_continue)
    {
        printf("Type \"q\" to exit, Ctrl+c to stop current running\n");
    }

    while (b_continue)
    {
        printf("prompt >> ");
        fflush(stdout);
        std::getline(std::cin, prompt);
        if (prompt == "q")
        {
            break;
        }
        if (prompt == "")
        {
            continue;
        }

        printf("image >> ");
        fflush(stdout);
        std::string image_prompt;
        std::getline(std::cin, image_prompt);
        std::string output;
        if (image_prompt == "")
        {
            lLaMa.Encode(prompt_data, position_ids, config, prompt_complete(prompt, attr.tokenizer_type));
            output = lLaMa.Run(prompt_data, position_ids);
        }
        else
        {
            auto src = ReadImages(image_prompt);
            if (src.empty())
            {
                // output = lLaMa.Run(prompt);
                ALOGE("image prompt(%s) not found", image_prompt.c_str());
                // continue;
                lLaMa.Encode(prompt_data, position_ids, config, prompt_complete(prompt, attr.tokenizer_type));
                output = lLaMa.Run(prompt_data, position_ids);
            }
            else
            {
                lLaMa.Encode(src, b_video, img_embed, config);
                lLaMa.Encode(img_embed, prompt_data, position_ids, config, prompt_complete(prompt, attr.tokenizer_type));
                output = lLaMa.Run(prompt_data, position_ids);
            }
        }

        if (!b_live_print)
            printf("%s\n", output.c_str());
    }

    lLaMa.Deinit();

    return 0;
}