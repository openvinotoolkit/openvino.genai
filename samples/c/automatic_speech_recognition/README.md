# Automatic speech recognition C sample

This sample uses the GenAI `ASRPipeline` C API to transcribe a WAV file and
optionally print segment and word timestamps. It uses the
existing Whisper sample's WAV loader and error handling.

## Build and run

Build with `ENABLE_SAMPLES=ON`, or configure this directory against an installed
OpenVINO GenAI package:

```sh
cmake -S . -B build -DOpenVINOGenAI_DIR=/path/to/runtime/cmake
cmake --build build
optimum-cli export openvino --model openai/whisper-tiny --weight-format fp16 whisper-tiny
./build/automatic_speech_recognition_c whisper-tiny audio.wav CPU
```

The device is optional and defaults to CPU. The shared WAV loader accepts mono PCM WAV audio and
resamples it to 16 kHz when needed.

By default, the sample uses automatic language selection and disables timestamps,
so the same command works across supported ASR exports, including SenseVoiceSmall.
See the [supported speech recognition models](../../../site/docs/supported-models/index.mdx#speech-recognition-models)
for the current list.

To reproduce the C++ Whisper example with English and word timestamps:

```sh
./build/automatic_speech_recognition_c whisper-tiny audio.wav --language en --timestamps --word-timestamps
```

Use `--max-new-tokens N` to bound transcription length.

Language values depend on the model:

- Multilingual Whisper and SenseVoiceSmall use codes such as `en`; SenseVoiceSmall also accepts `auto`.
- Qwen3-ASR uses names such as `English` and `Japanese`.
- Fun-ASR inserts the supplied value into a Chinese-language transcription instruction.
  Its model examples use labels such as `英文` and `日文`; `English` may work, but
  language names should not be assumed equivalent across models.

Whisper supports segment and word timestamps. Word timestamps require the
`word_timestamps` constructor property and appropriate `alignment_heads` in the
model's generation configuration; the sample enables the property when
`--word-timestamps` is supplied. Qwen3-ASR word timestamps require an
`ASRForcedAligner`, available in C++ and Python but currently unavailable through
the C API. Fun-ASR and SenseVoiceSmall ignore the word timestamp flag. Segment
timestamps selected by `--timestamps` are supported only for Whisper.

The pipeline selects the model architecture automatically. Configuration can be
retrieved from the pipeline, changed using `ov_genai_asr_generation_config_*`,
and passed to `ov_genai_asr_pipeline_generate`. Passing `NULL` for the configuration
uses the configuration stored in the pipeline, including updates made through
`ov_genai_asr_pipeline_set_generation_config`. Creating a standalone configuration
uses model-independent defaults; retrieve the pipeline configuration or load its
JSON file when model-specific settings are needed. Base generation settings are
independent copies. To change individual settings, get a copy, modify it, and
apply it with `ov_genai_asr_generation_config_set_generation_config`, which
replaces every base field, including generation limits and EOS/stop token IDs.

Text and language getters support a two-call buffer-size query, including the
null terminator. Segment and word getters take a transcription index followed
by a timestamp index. Returned configurations, results, metrics, and timestamp
handles are owned by the caller and must be released with their matching free
functions. Timestamp handles remain valid after freeing the result or pipeline.

Developer regression-test instructions are in
[tests/cpp/README_ASR_C_API.md](../../../tests/cpp/README_ASR_C_API.md).
