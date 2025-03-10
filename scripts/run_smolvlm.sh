./main \
--template_filename_axmodel "../SmolVLM-256M-Instruct/smolvlm-256m-ax650/llama_p128_l%d_together.axmodel" \
--axmodel_num 30 \
--filename_vpm_resampler_axmodedl "../SmolVLM-256M-Instruct/smolvlm-256m-ax650/SmolVLM-256M-Instruct_vision_nhwc.axmodel" \
--tokenizer_type 2 \
--bos 0 --eos 0 \
--use_mmap_load_embed 1 \
--filename_tokenizer_model "http://localhost:8080" \
--filename_post_axmodel "../SmolVLM-256M-Instruct/smolvlm-256m-ax650/llama_post.axmodel" \
--use_topk 0 \
--filename_tokens_embed "../SmolVLM-256M-Instruct/smolvlm-256m-ax650/model.embed_tokens.weight.bfloat16.bin" \
--tokens_embed_num 49280 \
--tokens_embed_size 576 \
--live_print 1 \
--continue 1 \
--img_width 512 \
--img_height 512 \
--img_token_id 49190 \
--post_config_path post_config.json \
--prompt "$1" --image "$2"
