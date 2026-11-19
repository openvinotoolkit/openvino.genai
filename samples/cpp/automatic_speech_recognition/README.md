# Automatic speech recognition sample

This example showcases inference of speech recognition models. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `ov::genai::ASRPipeline` and uses audio file in wav format as an input source.

## Download and convert the model and tokenizers

It's not required to install [../../export-requirements.txt](../../export-requirements.txt) for deployment if the model has already been exported.

```sh
pip install -r ../../requirements.txt
```

Some newer models may require the latest Optimum Intel. If needed, run:

```sh
pip install "optimum-intel @ git+https://github.com/huggingface/optimum-intel.git@main"
```

Then, run the export with Optimum CLI:

```sh
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

## Streaming ASR sample

[`asr_streaming.cpp`](asr_streaming.cpp) transcribes audio while it is being spoken with `ov::genai::ASRStreamingSession` (Qwen3-ASR, Voxtral Realtime). The WAV file is fed at real-time speed, as a live microphone would deliver it:

```c++
ov::genai::ASRStreamingSession session(pipeline, streaming_config);
for (const auto& piece : audio_pieces) {
    session.push(piece);                 // only buffers audio
    while (auto result = session.read())  // runs inference once a full chunk is buffered
        print(*result);
}
session.end();                           // close input
while (auto result = session.read())      // remaining audio; the last result has is_final == true
    print(*result);
```

```sh
asr_streaming <MODEL_DIR> <WAV_FILE> [--device DEVICE] [--chunk-sec SEC] [--step-ms MS] [--window-chunks N]
              [--unfixed-chunks N] [--unfixed-tokens N] [--streams N] [--attention-backend PA|SDPA]
```

For example, four concurrent streams whose inference the PagedAttention backend batches:

```sh
asr_streaming Qwen3-ASR-0.6B recording.wav --attention-backend PA --streams 4
```

Each result exposes:

- `committed_text`: all stable text so far. It only grows.
- `new_committed_text`: stable text added by this result.
- `partial_text`: unstable tail that may change in the next result. Empty in the final result.
- `is_final`: true for the last result after `end()`.

The sample prints one timeline line per result, then compares latency with offline `generate()` over the same audio:

```
[stream 0] partial audio   2.0-  4.0 s | read   1187 ms | latency   1187 ms | +"Folks, you know, if you watch the show, you" [ know I spend most.]
...
[stream 0] final   audio  20.0- 20.0 s | read      0 ms | latency    885 ms | +" using traditional push knife." []

=== Latency: time from end of speech to text ===
streaming, every result: p50    885 ms | p95   1182 ms | max   1182 ms
streaming, final result:    885 ms
offline generate()     :   4251 ms

=== Compute: total inference time per stream ===
streaming, sum of read():   8853 ms | 0.44 s per 1 s of audio
offline generate()      :   4251 ms | 0.21 s per 1 s of audio
```

- `read`: time spent in `read()`, which runs inference on one chunk.
- `latency`: time from the moment the last sample of the chunk was spoken until `read()` returned its result.
- Offline `generate()` can only start after the speech has ended, so its whole duration is latency. Streaming runs most inference while the speech goes on.
- Compute above 1.0 s per 1 s of audio means streaming can't keep up with a live source, and its latency grows with every chunk. Re-decoding models such as Qwen3-ASR spend more compute than offline `generate()`, because every chunk re-transcribes the audio window.

# ASR pipeline usage

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

## Continuous batching with Qwen3-ASR

Qwen3-ASR model supports `ContinuousBatchingPipeline` API. The model's chat template is designed for transcription: it discards user text and assistant turns, so questions about earlier audio, translation instructions, and multi-turn audio conversations are not supported.

```cpp
ov::genai::ContinuousBatchingPipeline pipe(model_dir, ov::genai::SchedulerConfig{}, "CPU");
const std::vector<std::string> wav_file_paths = {"audio_1.wav", "audio_2.wav"};
auto config = pipe.get_config();
std::vector<ov::genai::GenerationHandle> handles;

for (const auto& path : wav_file_paths) {
    auto waveform = utils::audio::read_wav(path);
    ov::Tensor audio(ov::element::f32, {waveform.size()});
    std::copy(waveform.begin(), waveform.end(), audio.data<float>());

    const std::vector<ov::Tensor> audios{audio};
    auto handle = pipe.add_request(handles.size(), "", ov::genai::audios(audios),
                                   ov::genai::generation_config(config));
    handles.push_back(handle);
}

while (pipe.has_non_finished_requests()) {
    pipe.step();
}

for (const auto& handle : handles) {
    std::cout << pipe.get_tokenizer().decode(handle->read_all().front().generated_ids) << '\n';
}
```

### Troubleshooting

#### Empty or rubbish output

Example output:
```
----------------
```

To resolve this ensure that audio data has 16k Hz sampling rate
