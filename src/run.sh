BASE=../../../huggingface/VoxCPM/
export LD_LIBRARY_PATH=../build/onnxruntime-aarch64-none-gnu-1.16.0/lib/:$LD_LIBRARY_PATH

rm output*.wav
../build/install/bin/main \
--dir_axmodels $BASE/axmodels \
--dir_base_lm  $BASE/base_lm-axmodels \
--dir_residual_lm $BASE/residual_lm-axmodels \
--dir_feat_encoder $BASE/feat_encoder_encoder-axmodels \
--dir_decoder_estimator $BASE/feat_decoder_estimator_decoder-axmodels \
--url_tokenizer "http://127.0.0.1:9999" \
--n_timesteps 10 \
--continue 0 \
--prompt_wav_path en_man1.mp3 \
--prompt_text "Because he has zero capacity to respond to the two and a half hour" \
--text "Streaming text to speech is easy with VoxCPM!"

chmod 777 output*.wav
