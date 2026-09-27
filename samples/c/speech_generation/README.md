# Text to speech in C

Download an exported model such as [OpenVINO/Kokoro-82M-int8-ov](https://huggingface.co/OpenVINO/Kokoro-82M-int8-ov). Keep `config.json`, `openvino_model.xml`, `openvino_model.bin`, the selected `voices/*.bin` file, and the English lexicon files in `data/` together in the model directory.

Build this sample with `ENABLE_SAMPLES=ON`, then run:

```sh
text2speech_c MODEL_DIR "Hello from OpenVINO" MODEL_DIR/voices/af_heart.bin output_audio.wav CPU
```

The sample reads a float32 speaker embedding and writes a mono 16-bit PCM WAV file. Kokoro requires an embedding. Pass `-` in its place to use the default voice of a model that provides one, such as SpeechT5. The returned waveform tensor is owned by the caller and must be released with `ov_tensor_free`.
