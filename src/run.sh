LLM_DIR=../../model_convert/CosyVoice-BlankEN-Ax650-prefill_512/
TOKEN2WAV_DIR=../../model_convert/token2wav-axmodels/

../build/install/bin/main \
--template_filename_axmodel "${LLM_DIR}/qwen2_p128_l%d_together.axmodel" \
--token2wav_axmodel_dir $TOKEN2WAV_DIR \
--axmodel_num 24 \
--bos 0 --eos 0 \
--filename_tokenizer_model "http://10.122.86.184:12345" \
--filename_post_axmodel "${LLM_DIR}/qwen2_post.axmodel" \
--filename_decoder_axmodel "${LLM_DIR}/llm_decoder.axmodel" \
--filename_tokens_embed "${LLM_DIR}/model.embed_tokens.weight.bfloat16.bin" \
--filename_llm_embed "${LLM_DIR}/llm.llm_embedding.float16.bin" \
--filename_speech_embed "${LLM_DIR}/llm.speech_embedding.float16.bin" \
--continue 1 \
--text "君不见黄河之水天上来，奔流到海不复回。君不见高堂明镜悲白发，朝如青丝暮成雪。"