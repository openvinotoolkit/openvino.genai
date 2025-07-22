# Whisper automatic speech recognition sample

This example showcases inference of speech recognition Whisper Models. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `ov::genai::WhisperPipeline` and uses audio file in wav format as an input source.

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

Follow [Get Started with Samples](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/get-started-demos.html) to run the sample.

`whisper_speech_recognition whisper-base how_are_you_doing_today.wav`

Output:
```
 How are you doing today?
timestamps: [0, 2] text:  How are you doing today?
```

Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-recognition-models-whisper-based) for more details.

# Whisper pipeline usage

```c++
#include "openvino/genai/whisper_pipeline.hpp"

ov::genai::WhisperPipeline pipeline(model_dir, "CPU");
// Pipeline expects normalized audio with Sample Rate of 16kHz
ov::genai::RawSpeechInput raw_speech = read_wav("how_are_you_doing_today.wav");
auto result = pipeline.generate(raw_speech);
//  How are you doing today?
```

### Transcription

Whisper pipeline predicts the language of the source audio automatically.

```c++
ov::genai::RawSpeechInput raw_speech = read_wav("how_are_you_doing_today.wav");
auto result = pipeline.generate(raw_speech);
//  How are you doing today?

raw_speech = read_wav("fr_sample.wav");
result = pipeline.generate(raw_speech);
//  Il s'agit d'une entité très complexe qui consiste...
```

If the source audio languange is know in advance, it can be specified as an argument to `generate` method:

```c++
ov::genai::RawSpeechInput raw_speech = read_wav("how_are_you_doing_today.wav");
auto result = pipeline.generate(raw_speech, ov::genai::language("<|en|>"));
//  How are you doing today?

raw_speech = read_wav("fr_sample.wav");
result = pipeline.generate(raw_speech, ov::genai::language("<|fr|>"));
//  Il s'agit d'une entité très complexe qui consiste...
```

### Translation

By default, Whisper performs the task of speech transcription, where the source audio language is the same as the target text language. To perform speech translation, where the target text is in English, set the task to "translate":

```c++
ov::genai::RawSpeechInput raw_speech = read_wav("fr_sample.wav");
auto result = pipeline.generate(raw_speech, ov::genai::task("translate"));
//  It is a very complex entity that consists...
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

### Long-Form audio Transcription

The Whisper model is designed to work on audio samples of up to 30s in duration. Whisper pipeline uses sequential chunking algorithm to transcribe audio samples of arbitrary length.
Sequential chunking algorithm uses a "sliding window", transcribing 30-second slices one after the other.

### Initial prompt and hotwords

Whisper pipeline has `initial_prompt` and `hotwords` generate arguments:
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
