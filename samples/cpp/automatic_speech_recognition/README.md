# Automatic speech recognition sample

This example showcases inference of speech recognition models. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `ov::genai::ASRPipeline` and uses audio file in wav format as an input source.

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

It's not required to install [../../export-requirements.txt](../../export-requirements.txt) for deployment if the model has already been exported.

```sh
pip install --upgrade-strategy eager -r ../../requirements.txt
optimum-cli export openvino --trust-remote-code --model openai/whisper-base whisper-base
```

## Prepare audio file

Prepare audio file in wav format with sampling rate 16k Hz.

You can download example audio file: https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav

## Run

Follow [Get Started with Samples](https://docs.openvino.ai/2026/get-started/learn-openvino/openvino-samples/get-started-demos.html) to run the sample.

`automatic_speech_recognition whisper-base how_are_you_doing_today.wav`

Output:
```
 How are you doing today?
timestamps: [0, 2] text:  How are you doing today?
```

Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-recognition-models-whisper-based) for more details.

# ASR pipeline usage
## Qwen3-ASR continuous batching

`automatic_speech_recognition_cb` uses `ContinuousBatchingPipeline::add_request()`
and `step()` to process multiple WAV files. Use a Qwen3-ASR split export containing
`openvino_audio_encoder_model.xml`, `openvino_text_embeddings_model.xml`,
`openvino_language_model.xml`, their weights, and the tokenizer/configuration files.
The legacy encoder/decoder export is not supported by this sample.

For Qwen3-ASR, when `apply_chat_template=true`, plain prompt text is treated as
ASR system context, not a user instruction. Audio tags are removed from that text
and represented as structured audio items in the user message. The model's stored
chat template formats these messages before audio placeholders are expanded.
Use an empty prompt for transcription without context. This interpretation is
specific to Qwen3-ASR, not shared with Qwen3-Omni. With `apply_chat_template=false`,
the caller supplies the complete formatted prompt; text is not moved to a system
message, but audio placeholders are still expanded.

```sh
automatic_speech_recognition_cb Qwen3-ASR-0.6B-split CPU first.wav second.wav
```

Each WAV file must have a 16 kHz sample rate and fit within one audio chunk
(at most 1200 seconds). The sample submits all files before stepping the decoder,
limits each output to 256 generated tokens, and prints raw model output including
language metadata. Audio encoding happens during submission; decoder inference is
continuously batched. No ASRPipeline chunk merging or timestamp prediction is performed.

### Input prompt examples

The following examples apply to Qwen3-ASR split models. Audio indices are zero-based
within each `add_request()` call; they are not request IDs. `audio0` and `audio1`
denote 1-D float32 PCM tensors sampled at 16 kHz.

#### Automatic formatting (`apply_chat_template=true`)

Pass context as plain text and let the model's stored chat template supply the
system/user/assistant wrappers:

```cpp
auto config = pipeline.get_config();
config.max_new_tokens = 256;
config.apply_chat_template = true;
auto handle = pipeline.add_request(
    0, "OpenVINO, Qwen",
    ov::genai::audios(std::vector<ov::Tensor>{audio0}),
    ov::genai::generation_config(config));
```

With the supplied Qwen3-ASR template, the messages before formatting are:

```json
[
  {"role": "system", "content": "OpenVINO, Qwen"},
  {"role": "user", "content": [{"type": "audio"}]}
]
```

| Prompt string | Supplied audio | System context | Audio order in the user message |
| --- | --- | --- | --- |
| `""` | `audio0` | Empty | `audio0` |
| `"OpenVINO, Qwen"` | `audio0` | `OpenVINO, Qwen` | `audio0` |
| `"<ov_genai_audio_0>"` | `audio0` | Empty | `audio0` |
| `"Names: <ov_genai_audio_0>Qwen"` | `audio0` | `Names: Qwen` | `audio0` |
| `"A<ov_genai_audio_0>B"` | `audio0` | `AB` | `audio0` |
| `""` | `audio0, audio1` | Empty | `audio0, audio1` |
| `"<ov_genai_audio_1><ov_genai_audio_0>"` | `audio0, audio1` | Empty | `audio1, audio0` |
| `"<ov_genai_audio_0><ov_genai_audio_0>"` | `audio0` | Empty | `audio0, audio0` |
| `"<ov_genai_audio_1>"` | `audio0, audio1` | Empty | Only `audio1`; `audio0` is not inserted |

Text on either side of an audio tag is concatenated exactly, without inserting or
trimming whitespace. Text position relative to audio does not survive automatic
formatting: all text becomes system context, and all referenced audio becomes user
content. For example, `"Translate to French"` is passed as system context; it does
not select a translation task. Context can provide names or vocabulary, but does
not guarantee particular transcription wording.

Instead of indexed tags, callers may use one native tag per supplied audio:

```text
<|audio_start|><|audio_pad|><|audio_end|>
```

Native tags bind to audio in input order. Two native tags require two supplied
audio tensors. With no tags, all supplied audio is inserted in input order. With
indexed tags, only referenced inputs are inserted. Multiple audio items remain
part of one generation request; use separate requests for separate transcriptions.

#### Explicit formatting (`apply_chat_template=false`)

The current C++ sample uses this mode. Supply the complete prompt, including role
wrappers and the assistant prefix. No text is extracted into system context:

```cpp
config.apply_chat_template = false;
const std::string prompt =
    "<|im_start|>system\nOpenVINO, Qwen<|im_end|>\n"
    "<|im_start|>user\n<ov_genai_audio_0><|im_end|>\n"
    "<|im_start|>assistant\n";
auto handle = pipeline.add_request(
    0, prompt,
    ov::genai::audios(std::vector<ov::Tensor>{audio0}),
    ov::genai::generation_config(config));
```

The indexed tag can be replaced by the native tag above. The same audio ordering,
repetition, and count rules apply in both modes. Audio placeholders are expanded
to the encoder-derived length in both modes; callers should not repeat
`<|audio_pad|>` manually.

For a forced English-language prefix, append `"language English<asr_text>"` after
the assistant newline in the explicit prompt. This supplies an output prefix;
it does not translate the recording. CB returns newly generated tokens, not the
prefix supplied in the prompt, and does not parse ASR language metadata.

#### Invalid or misleading inputs

| Input | Behavior |
| --- | --- |
| `<ov_genai_audio_1>` with only `audio0` | Error: audio index is outside the supplied range. |
| An indexed or native audio tag with no supplied audio | Error: no matching audio input. |
| Native tags mixed with indexed audio tags | Error: tag styles cannot be mixed. |
| One native tag with two supplied audio tensors | Error: native tag count must equal input count. |
| A complete formatted prompt with automatic formatting enabled | Role delimiters are not recognized as existing messages; text is treated as system context. Use `false` for formatted prompts. |
| Plain text or an empty prompt with audio and automatic formatting disabled | Audio markers are inserted, but no role wrappers are added. This is not a complete Qwen3-ASR transcription prompt. |
| No audio and no audio tags | No audio is inserted. Automatic formatting can still produce a text-only prompt, but this is not a transcription request. An empty explicit prompt is rejected. |

An empty audio tensor retains its input index but contributes zero audio embedding
tokens. It is not a substitute for a recording. Supplying images or videos is not
supported by Qwen3-ASR.

## ASRPipeline example

```c++
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"

ov::genai::ASRPipeline pipeline(model_dir, "CPU");
// Pipeline expects normalized audio with Sample Rate of 16kHz
ov::genai::RawSpeechInput raw_speech = read_wav("how_are_you_doing_today.wav");
auto result = pipeline.generate(raw_speech);
//  How are you doing today?
```

### Transcription

ASR pipeline predicts the language of the source audio automatically.

```c++
ov::genai::RawSpeechInput raw_speech = read_wav("how_are_you_doing_today.wav");
auto result = pipeline.generate(raw_speech);
//  How are you doing today?

raw_speech = read_wav("fr_sample.wav");
result = pipeline.generate(raw_speech);
//  Il s'agit d'une entité très complexe qui consiste...
```

If the source audio language is known in advance, it can be specified as an argument to `generate` method:

```c++
ov::genai::RawSpeechInput raw_speech = read_wav("how_are_you_doing_today.wav");
auto result = pipeline.generate(raw_speech, ov::genai::language("<|en|>"));
//  How are you doing today?

raw_speech = read_wav("fr_sample.wav");
result = pipeline.generate(raw_speech, ov::genai::language("<|fr|>"));
//  Il s'agit d'une entité très complexe qui consiste...
```

### Timestamps prediction

The model can predict timestamps. For sentence-level timestamps, pass the `return_timestamps` argument:

```C++
ov::genai::RawSpeechInput raw_speech = read_wav("how_are_you_doing_today.wav");
auto result = pipeline.generate(raw_speech, ov::genai::return_timestamps(true));

std::cout << std::setprecision(2);
for (auto& chunk : *result.chunks) {
    std::cout << "timestamps: [" << chunk.start_ts << ", " << chunk.end_ts << "] text: " << chunk.text << "\n";
}
// timestamps: [0, 2] text:  How are you doing today?
```

## Whisper-specific features

### Translation

By default, Whisper performs the task of speech transcription, where the source audio language is the same as the target text language. To perform speech translation, where the target text is in English, set the task to "translate":

```c++
ov::genai::RawSpeechInput raw_speech = read_wav("fr_sample.wav");
auto result = pipeline.generate(raw_speech, ov::genai::task("translate"));
//  It is a very complex entity that consists...
```

### Long-Form audio Transcription

The Whisper model is designed to work on audio samples of up to 30s in duration. ASR pipeline uses sequential chunking algorithm to transcribe audio samples of arbitrary length.
Sequential chunking algorithm uses a "sliding window", transcribing 30-second slices one after the other.

### Initial prompt and hotwords

ASR pipeline has `initial_prompt` and `hotwords` generate arguments:
* `initial_prompt`: initial prompt tokens passed as a previous transcription (after `<|startofprev|>` token) to the first processing window
* `hotwords`: hotwords tokens passed as a previous transcription (after `<|startofprev|>` token) to the all processing windows

The Whisper model can use that context to better understand the speech and maintain a consistent writing style. However, prompts do not need to be genuine transcripts from prior audio segments. Such prompts can be used to steer the model to use particular spellings or styles:

```c++
auto result = pipeline.generate(raw_speech);
//  He has gone and gone for good answered Paul Icrom who...

result = pipeline.generate(raw_speech, ov::genai::initial_prompt("Polychrome"));
//  He has gone and gone for good answered Polychrome who...
```

### Troubleshooting

#### Empty or rubbish output

Example output:
```
----------------
```

To resolve this ensure that audio data has 16k Hz sampling rate
