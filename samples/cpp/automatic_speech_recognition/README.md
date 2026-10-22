# Automatic speech recognition samples

These samples demonstrate speech recognition using `ov::genai::ASRPipeline`, with WAV audio files as input. They keep configuration options minimal to encourage users to explore and customize the source code, for example, by experimenting with GPU inference.

## Download and convert the models and tokenizers

Install [../../export-requirements.txt](../../export-requirements.txt) to convert a model.

```sh
pip install -U --upgrade-strategy eager -r ../../export-requirements.txt
```
These dependencies are needed only for model export, not for running already-exported OpenVINO models.

### Whisper

```sh
optimum-cli export openvino --trust-remote-code --model openai/whisper-base whisper-base
```

### Qwen3-ASR and Forced Aligner

Qwen3-ASR performs speech transcription. The Qwen3 Forced Aligner generates word-level timestamps for an existing transcript, which can be produced by Qwen3-ASR or by another ASR model. Export the models into separate directories.

```sh
pip install qwen-asr

optimum-cli export openvino --trust-remote-code --model Qwen/Qwen3-ASR-0.6B \
    --task automatic-speech-recognition-with-past qwen3-asr-ov

optimum-cli export openvino --trust-remote-code --model Qwen/Qwen3-ForcedAligner-0.6B \
    --task automatic-speech-recognition-with-past qwen3-forced-aligner-ov
```

### Fun-ASR-Nano

```sh
pip install funasr "transformers==4.57.6"

optimum-cli export openvino --trust-remote-code \
    --model FunAudioLLM/Fun-ASR-Nano-2512 fun-asr-nano-ov
```

## Prepare audio

Prepare audio file in wav format with sampling rate 16k Hz.

Download the example audio file: https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav

## Run the samples

Follow [Get Started with Samples](https://docs.openvino.ai/2026/get-started/learn-openvino/openvino-samples/get-started-demos.html) to run the samples.

### Automatic speech recognition

```sh
automatic_speech_recognition whisper-base how_are_you_doing_today.wav
```

Output:

```text
 How are you doing today?
timestamps: [0.00, 2.00] text:  How are you doing today?
```

### Standalone Forced-aligner word timestamps

```sh
forced_aligner_timestamps_generation fun-asr-nano-ov qwen3-forced-aligner-ov zh_sample.wav --language Chinese
```

Output:

```text
Transcription: 甚至出现交易几乎停滞的情况。

Word timestamps:
[0.40, 0.72]: 甚
[0.72, 0.96]: 至
[0.96, 1.12]: 出
[1.12, 1.52]: 现
[1.52, 1.76]: 交
[1.76, 2.00]: 易
[2.00, 2.24]: 几
[2.24, 2.48]: 乎
[2.48, 2.72]: 停
[2.72, 2.88]: 滞
[2.88, 3.04]: 的
[3.04, 3.36]: 情
[3.36, 3.68]: 况
```

Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-recognition-models) for more details.

## ASR pipeline usage

```c++
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"

ov::genai::ASRPipeline pipeline(model_dir, "CPU");
// Pipeline expects normalized audio with Sample Rate of 16kHz
ov::genai::RawSpeechInput raw_speech = read_wav("how_are_you_doing_today.wav");
auto result = pipeline.generate(raw_speech);
//  How are you doing today?
```

### Transcription

All supported ASR models can transcribe audio without specifying a language. Multilingual Whisper and Qwen3-ASR also detect and report the input language, whereas Fun-ASR-Nano does not. Therefore, in the standalone forced-aligner sample, `--language` is optional with Qwen3-ASR but required with Fun-ASR-Nano.

```c++
ov::genai::RawSpeechInput raw_speech = read_wav("how_are_you_doing_today.wav");
auto result = pipeline.generate(raw_speech);
//  How are you doing today?

raw_speech = read_wav("fr_sample.wav");
result = pipeline.generate(raw_speech);
//  Il s'agit d'une entité très complexe qui consiste...
```

If the language is known, pass it to `generate()`:

```c++
ov::genai::RawSpeechInput raw_speech = read_wav("how_are_you_doing_today.wav");
auto result = pipeline.generate(raw_speech, ov::genai::language("en"));
//  How are you doing today?

raw_speech = read_wav("fr_sample.wav");
result = pipeline.generate(raw_speech, ov::genai::language("fr"));
//  Il s'agit d'une entité très complexe qui consiste...
```

Whisper uses language codes such as `"en"` and `"zh"`, while Qwen3-ASR uses language names such as `"English"` and `"Chinese"`. The documented language hints for Fun-ASR-Nano are `"中文"`, `"英文"`, and `"日文"`.

### Timestamps prediction

Whisper can predict timestamps for recognized speech segments using `return_timestamps`. To generate word-level timestamps, pass `ov::genai::word_timestamps(true)` both when constructing the Whisper pipeline and when calling `generate()`.

```c++
ov::genai::RawSpeechInput raw_speech = read_wav("how_are_you_doing_today.wav");
auto result = pipeline.generate(raw_speech, ov::genai::return_timestamps(true));

std::cout << std::fixed << std::setprecision(2);
for (auto& chunk : (*result.chunks)[0]) {
    std::cout << "timestamps: [" << chunk.start_ts << ", " << chunk.end_ts << "] text: " << chunk.text << "\n";
}
// timestamps: [0.00, 2.00] text:  How are you doing today?
```

Qwen3-ASR can produce word-level timestamps if integrated with its forced aligner into ASRPipeline:

```c++
#include "openvino/genai/automatic_speech_recognition/forced_aligner.hpp"

auto aligner = std::make_shared<ov::genai::ASRForcedAligner>(forced_aligner_model_dir, "CPU");
ov::genai::ASRPipeline pipe(qwen3_asr_model_dir, "CPU", ov::genai::forced_aligner(aligner));

auto raw_speech = read_wav("fr_audio.wav");
auto result = pipe.generate(raw_speech, ov::genai::word_timestamps(true));

std::cout << "Detected language: " << result.languages[0] << "\n";
std::cout << "Transcription: " << result.texts[0] << "\n";
std::cout << "\nWord timestamps:\n";

std::cout << std::fixed << std::setprecision(2);
for (const auto& word : (*result.words)[0]) {
    std::cout << "[" << word.start_ts << ", " << word.end_ts << "]: " << word.text << "\n";
}
```

Output:

```text
Detected language: French
Transcription: Les premiers cas de cette maladie saisonnière ont été déclarés fin juillet.

Word timestamps:
[1.04, 1.12]: Les
[1.12, 1.44]: premiers
[1.44, 1.52]: cas
[1.52, 1.60]: de
[1.60, 1.84]: cette
[1.84, 2.16]: maladie
[2.16, 2.80]: saisonnière
[2.88, 2.96]: ont
[2.96, 3.12]: été
[3.28, 3.76]: déclarés
[3.84, 4.00]: fin
[4.00, 4.32]: juillet
```

### Whisper-specific features

#### Translation

Whisper normally performs transcription, producing text in the language of the source audio. To translate non-English speech into English, set the task to `"translate"`:

```c++
ov::genai::RawSpeechInput raw_speech = read_wav("fr_sample.wav");
auto result = pipeline.generate(raw_speech, ov::genai::task("translate"));
//  It is a very complex entity that consists...
```

#### Long-form audio transcription

Whisper processes audio in windows of up to 30 seconds. For longer recordings, the pipeline uses a sequential chunking algorithm with a sliding window to transcribe successive portions of the audio.

#### Initial prompt and hotwords

Whisper supports the `initial_prompt` and `hotwords` generation arguments:

- `initial_prompt`: initial prompt tokens passed as a previous transcription (after `<|startofprev|>` token) to the first processing window
- `hotwords`: hotwords tokens passed as a previous transcription (after `<|startofprev|>` token) to the all processing windows

These arguments help guide transcription and maintain consistent writing style. The supplied text does not need to be an actual earlier transcription. For example, an initial prompt can help recognize an uncommon name:

```c++
auto result = pipeline.generate(raw_speech);
//  He has gone and gone for good answered Paul Icrom who...

result = pipeline.generate(raw_speech, ov::genai::initial_prompt("Polychrome"));
//  He has gone and gone for good answered Polychrome who...
```

### Troubleshooting

#### Empty or incorrect transcription

If the output is empty or unintelligible, check the input audio and its sample rate. `ASRPipeline` expects normalized audio sampled at 16 kHz.

Example output:
```
----------------
```

You can use FFmpeg to convert an existing recording.
