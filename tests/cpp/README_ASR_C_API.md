# ASR C API regression tests

These developer tests require a source build; `tests_asr_c_api` is not included
in the installed sample archive.

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
