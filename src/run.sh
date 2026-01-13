LLM_DIR=../CosyVoice/CosyVoice-BlankEN-Ax650-C64-P256-CTX512/
TOKEN2WAV_DIR=../CosyVoice/token2wav-axmodels/

# rm output*.wav
./build_650/main \
--template_filename_axmodel "${LLM_DIR}/qwen2_p64_l%d_together.axmodel" \
--token2wav_axmodel_dir $TOKEN2WAV_DIR \
--n_timesteps 10 \
--axmodel_num 24 \
--bos 0 --eos 0 \
--filename_tokenizer_model "http://10.122.86.184:12345" \
--filename_post_axmodel "${LLM_DIR}/qwen2_post.axmodel" \
--filename_decoder_axmodel "${LLM_DIR}/llm_decoder.axmodel" \
--filename_tokens_embed "${LLM_DIR}/model.embed_tokens.weight.bfloat16.bin" \
--filename_llm_embed "${LLM_DIR}/llm.speech_embedding.float16.bin" \
--filename_speech_embed "${LLM_DIR}/llm.speech_embedding.float16.bin" \
--continue 0 \
--prompt_files prompt_files \
--devices "0,1,2" \
--text "高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。"


chmod 777 output*.wav
