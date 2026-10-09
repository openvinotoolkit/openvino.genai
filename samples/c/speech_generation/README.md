# Text to speech in C

Download an exported model such as [OpenVINO/Kokoro-82M-int8-ov](https://huggingface.co/OpenVINO/Kokoro-82M-int8-ov). Keep `config.json`, `openvino_model.xml`, `openvino_model.bin`, the selected `voices/*.bin` file, and the English lexicon files in `data/` together in the model directory.

Build this sample with `ENABLE_SAMPLES=ON`, then run:

```sh
text2speech_c MODEL_DIR "Hello from OpenVINO" MODEL_DIR/voices/af_heart.bin output_audio.wav --device CPU
```

To generate two waveforms in one call with the same speaker embedding (optionally with `--speed`, `--language` and `--device` flags):

```sh
text2speech_c --batch MODEL_DIR "Hello from OpenVINO" "A second sentence" \
  MODEL_DIR/voices/af_heart.bin first.wav second.wav --speed 1.1 --language en-us
```

The sample reads a float32 speaker embedding and writes a mono 16-bit PCM WAV file. Kokoro requires an embedding. Pass `-` in its place to use the default voice of a model that provides one, such as SpeechT5. The returned waveform tensor is owned by the caller and must be released with `ov_tensor_free`.

For SpeechT5, [llmware/speech-t5-tts-ov](https://huggingface.co/llmware/speech-t5-tts-ov) provides an exported model with `openvino_tokenizer.xml` and `openvino_tokenizer.bin`, which are required for text input.
