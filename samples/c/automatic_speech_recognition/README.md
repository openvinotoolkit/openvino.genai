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
so the same command works with Whisper, Qwen3-ASR, and Fun-ASR exports.

To reproduce the C++ Whisper example with English and word timestamps:

```sh
./build/automatic_speech_recognition_c whisper-tiny audio.wav --language '<|en|>' --timestamps --word-timestamps
```

Use `--max-new-tokens N` to bound transcription length.

For Qwen3-ASR or Fun-ASR, pass `--language English` to select English explicitly.
Whisper supports segment and word timestamps. Qwen3-ASR word timestamps require a
forced aligner; passing a preconstructed aligner is currently a C++ API feature.

The pipeline selects the model architecture automatically. Configuration can be
retrieved from the pipeline, changed using `ov_genai_asr_generation_config_*`,
and passed to `ov_genai_asr_pipeline_generate`. Passing `NULL` for the configuration
uses the pipeline defaults. Base generation settings are independent copies;
apply edits with `ov_genai_asr_generation_config_set_generation_config`.

Text and language getters support a two-call buffer-size query, including the
null terminator. Segment and word getters take a transcription index followed
by a timestamp index. Returned configurations, results, metrics, and timestamp
handles are owned by the caller and must be released with their matching free
functions. Timestamp handles remain valid after freeing the result or pipeline.

## C API regression tests

Build `tests_asr_c_api` with `ENABLE_TESTS=ON`. It contains model-free unit tests
and optional real-model C/C++ comparisons. Supply all three exported model paths
and mono 16 kHz PCM test audio to run the complete suite without skips:

```sh
ASR_TEST_WHISPER_MODEL=/models/whisper-tiny \
ASR_TEST_QWEN_MODEL=/models/qwen3-asr \
ASR_TEST_FUN_MODEL=/models/fun-asr \
ASR_TEST_WAV=/audio/test.wav \
./build/tests/cpp/tests_asr_c_api
```

Random fixture models need a `generation_config.json` with a finite
`max_new_tokens` (for example, 16) so default-config inference terminates.
The suite compares default and per-call configurations, stored configuration,
transcriptions, scores, languages, timestamps, metrics, and retained results.
The Whisper case additionally compares both legacy Whisper C and C++ pipelines.
The sample pytest suite compares the C executable against Python for all three
architectures and verifies Whisper output against the other language samples.
