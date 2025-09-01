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
    std::vector<unsigned short> prompt_text_embeds,
    std::vector<unsigned short> prompt_speech_embeds,
    // for flow
    std::vector<float> prompt_feat,
    std::vector<float> prompt_speech_embeds_flow,
    std::vector<float> spk_embeds
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
        int prompt_token_len = prompt_speech_embeds_flow.size() / lToken2Wav.flow_embed_size;
        int prompt_token_align_len = int(prompt_token_len / lToken2Wav.token_hop_len) * lToken2Wav.token_hop_len;
        ALOGI("prompt_token_len %d prompt_token_align_len %d",prompt_token_len, prompt_token_align_len);
        // auto prompt_speech_embeds_flow1 = slice_3d_last_dim_from<float>(prompt_speech_embeds_flow, 1, 1, prompt_speech_embeds_flow.size(), prompt_token_align_len * lToken2Wav.flow_embed_size);
        std::vector<float> prompt_speech_embeds_flow1;
        // memcpy(prompt_speech_embeds_flow1.data(), prompt_speech_embeds_flow.data(), prompt_speech_embeds_flow1.size()*sizeof(float));
        prompt_speech_embeds_flow1.insert(prompt_speech_embeds_flow1.begin(), prompt_speech_embeds_flow.begin(), prompt_speech_embeds_flow.begin()+prompt_token_align_len * 512);
        // auto prompt_feat1 = slice_3d_last_dim_from<float>(prompt_feat, 1, 1, prompt_feat.size(), prompt_token_align_len * 80 * 2);
        ALOGI("prompt_feat size %d", prompt_feat.size());
        std::vector<float> prompt_feat1;
        prompt_feat1.insert(prompt_feat1.begin(), prompt_feat.begin(), prompt_feat.begin()+prompt_token_align_len*2*80);
        ALOGI("prompt_feat size %d", prompt_feat1.size());

        int promot_token_pad = 0;
        int this_token_hop_len;
        int i=0;
        while (true) {
            // std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
            this_token_hop_len = (token_offset == 0)? lToken2Wav.token_hop_len + promot_token_pad : lToken2Wav.token_hop_len;

            std::unique_lock<std::mutex> lock(g_buffer_mutex);

            // Wait until there are enough tokens OR LLM has finished
            // The lambda is the predicate that must be true for wait to stop waiting.
            g_buffer_cv.wait(lock, [&] {
                return (g_token_buffer.size() - token_offset >= this_token_hop_len + lToken2Wav.pre_lookahead_len) || \
                        g_llm_finished.load() ;
            });

            ALOGI("token2wav proc");
            // Check if we should process based on threshold or if LLM is finished
            if (g_token_buffer.size() >= this_token_hop_len + lToken2Wav.pre_lookahead_len ) {
                
                ALOGI("token2wav proc");
                // Extract tokens to process
                std::vector<SpeechToken> token;
                int start = token_offset -  std::min( int(token_offset / lToken2Wav.token_hop_len), lToken2Wav.max_infer_chunk_num-1) * lToken2Wav.token_hop_len;
                int end = token_offset + this_token_hop_len + lToken2Wav.pre_lookahead_len;
                ALOGI("token_offset %d, this_token_hop_len:%dm pre_lookahead_len:%d", token_offset , this_token_hop_len , lToken2Wav.pre_lookahead_len);
                ALOGI("start:%d, end:%d, g_token_buffer.size():%d",start, end,g_token_buffer.size());
                // std::copy(g_token_buffer.begin() + start, g_token_buffer.begin() + end, token.begin());
                token.insert(token.end(), g_token_buffer.begin()+start, g_token_buffer.begin()+end);
                // --- End of Critical Section ---

                // Release the lock while processing, allowing LLM to produce more tokens
                lock.unlock();

                // --- Simulate Token2Wav Processing ---
                std::cout << "[Main/Token2Wav Thread] Processing batch of " << token.size() << " tokens...\n";
                ALOGI("token size:%d", token.size());
                auto speech = lToken2Wav.infer(token, prompt_speech_embeds_flow1, prompt_feat1, spk_embeds, token_offset, false);
                token_offset += this_token_hop_len;

                //TODO: 另起一个线程处理生成的音频
                output.insert(output.end(), speech.begin(), speech.end());
                std::string path = "output_"+std::to_string(i)+".wav";
                ALOGI("speech size:%d", speech.size());
                saveVectorAsWavFloat(speech, path, 24000, 1);
                i += 1;

            } 
            
            else if (g_llm_finished.load() ) {
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
    
    cmd.add<bool>("bos", 0, "", false, attr.b_bos);
    cmd.add<bool>("eos", 0, "", false, attr.b_eos);
    cmd.add<int>("axmodel_num", 0, "num of axmodel(for template)", false, attr.axmodel_num);
    cmd.add<bool>("continue", 0, "continuous dialogue", false, b_continue);

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
   

    b_continue = cmd.get<bool>("continue");

    if (!lLaMa.Init(attr))
    {
        return -1;
    }

    if (!lToken2Wav.Init(token2wav_axmodel_dir))
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
    ALOGI();
    readtxt("prompt_text_1_15.txt", prompt_text_token);
    readtxt("llm_prompt_speech_token_1_87.txt", prompt_speech_token);
    readtxt("prompt_speech_feat_1_174_80.txt", prompt_feat);
    readtxt("flow_embedding_1_192.txt", spk_embeds);    
    ALOGI("prompt_text_token.size:%d",prompt_text_token.size());
    lLaMa.TextToken2Embeds(prompt_text_token, prompt_text_embeds);
    ALOGI("prompt_speech_token.size:%d",prompt_speech_token.size());
    lLaMa.SpeechToken2Embeds(prompt_speech_token, prompt_speech_embeds);
    ALOGI("prompt_speech_token.size:%d",prompt_speech_token.size());
    lToken2Wav.SpeechToken2Embeds(prompt_speech_token, prompt_speech_embeds_flow);
    ALOGI();
    tts(
            // for llm
            text, prompt_text_embeds,prompt_speech_embeds,
            // for flow
            prompt_feat, prompt_speech_embeds_flow, spk_embeds
        );
    ALOGI();
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