# Automatic speech recognition sample

This example showcases inference of speech recognition models. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `openvino_genai.ASRPipeline` and uses audio file in wav format as an input source.

## Download and convert the model and tokenizers

Install [../../export-requirements.txt](../../export-requirements.txt) to convert a model.

```sh
pip install -r ../../export-requirements.txt
```

Some newer models may require the latest Optimum Intel. If needed, run:

```sh
pip install "optimum-intel @ git+https://github.com/huggingface/optimum-intel.git@main"
```

Then, run the export with Optimum CLI:

```sh
optimum-cli export openvino --trust-remote-code --model openai/whisper-base whisper-base
```

Alternatively, you can do it in Python code:

```python
from optimum.exporters.openvino.convert import export_tokenizer
from optimum.intel import OVModelForSpeechSeq2Seq
from transformers import AutoTokenizer

output_dir = "whisper-base"

model = OVModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base", export=True, trust_remote_code=True)
model.save_pretrained(output_dir)

tokenizer = AutoTokenizer.from_pretrained("openai/whisper-base")
export_tokenizer(tokenizer, output_dir)
```

## Prepare audio file

Prepare audio file in wav format with sampling rate 16k Hz.

You can download example audio file: https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav

Or you can use the [`recorder.py`](recorder.py) script. The script records 5 seconds of audio from the microphone.

To install `PyAudio` dependency follow the [installation instructions](https://pypi.org/project/PyAudio/).

To run the script:
```
python recorder.py
```

## Streaming ASR sample

[`asr_streaming.py`](asr_streaming.py) transcribes audio while it is being spoken with `ASRStreamingSession` (Qwen3-ASR, Voxtral Realtime). Audio comes from the microphone, or from a WAV file fed at real-time speed:

```python
session = openvino_genai.ASRStreamingSession(pipe, config)
for block in audio_blocks:
    session.push(block)                       # only buffers audio
    while (result := session.read()) is not None:  # runs inference once a full chunk is buffered
        print(result.new_committed_text)
session.end()                                 # close input
while (result := session.read()) is not None:  # remaining audio; the last result has is_final == True
    print(result.new_committed_text)
```

### 1. Stream from a WAV file

```sh
python asr_streaming.py /path/to/exported_model_dir --wav /path/to/audio.wav
```

Four concurrent streams whose inference the PagedAttention backend batches:

```sh
python asr_streaming.py /path/to/exported_model_dir --wav /path/to/audio.wav --attention-backend PA --streams 4
```

### 2. Stream from a microphone

```sh
python asr_streaming.py /path/to/exported_model_dir --duration 10
```

Streaming options: `--chunk-size-sec` (audio per inference), `--unfixed-token-num`, `--unfixed-chunk-num`, `--window-chunk-num` and `--block-ms` (size of pushed audio blocks).

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

## Run

Install [deployment-requirements.txt](../../deployment-requirements.txt) via `pip install -r ../../deployment-requirements.txt` and then, run the existing file-based sample using its positional arguments:

`python automatic_speech_recognition.py whisper-base how_are_you_doing_today.wav`

Output:
```
 How are you doing today?
timestamps: [0.00, 2.00] text:  How are you doing today?
```

Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-recognition-models-whisper-based) for more details.

# ASR pipeline usage

```python
import openvino_genai
import librosa

def read_wav(filepath):
    raw_speech, samplerate = librosa.load(filepath, sr=16000)
    return raw_speech.tolist()

pipe = openvino_genai.ASRPipeline(model_dir, "CPU")
# Pipeline expects normalized audio with Sample Rate of 16kHz
raw_speech = read_wav('how_are_you_doing_today.wav')
result = pipe.generate(raw_speech)
#  How are you doing today?
```

### Transcription

ASR pipeline predicts the language of the source audio automatically.

```python
raw_speech = read_wav('how_are_you_doing_today.wav')
result = pipe.generate(raw_speech)
#  How are you doing today?

raw_speech = read_wav('fr_sample.wav')
result = pipe.generate(raw_speech)
#  Il s'agit d'une entité très complexe qui consiste...
```

If the source audio language is known in advance, it can be specified as an argument to `generate` method:

```python
raw_speech = read_wav("how_are_you_doing_today.wav")
result = pipe.generate(raw_speech, language="<|en|>")
#  How are you doing today?

raw_speech = read_wav("fr_sample.wav")
result = pipe.generate(raw_speech, language="<|fr|>")
#  Il s'agit d'une entité très complexe qui consiste...
```

### Timestamps prediction

The model can predict timestamps. For sentence-level timestamps, pass the `return_timestamps` argument:

```python
raw_speech = read_wav("how_are_you_doing_today.wav")
result = pipe.generate(raw_speech, return_timestamps=True)

for chunk in result.chunks:
    print(f"timestamps: [{chunk.start_ts:.2f}, {chunk.end_ts:.2f}] text: {chunk.text}")
# timestamps: [0.00, 2.00] text:  How are you doing today?
```

## Whisper-specific features

### Translation

By default, Whisper performs the task of speech transcription, where the source audio language is the same as the target text language. To perform speech translation, where the target text is in English, set the task to "translate":

```python
raw_speech = read_wav("fr_sample.wav")
result = pipe.generate(raw_speech, task="translate")
# It is a very complex entity that consists...
```

### Long-Form audio Transcription

The Whisper model is designed to work on audio samples of up to 30s in duration. Whisper pipeline uses sequential chunking algorithm to transcribe audio samples of arbitrary length.
Sequential chunking algorithm uses a "sliding window", transcribing 30-second slices one after the other.

### Initial prompt and hotwords

Whisper pipeline has `initial_prompt` and `hotwords` generate arguments:
* `initial_prompt`: initial prompt tokens passed as a previous transcription (after `<|startofprev|>` token) to the first processing window
* `hotwords`: hotwords tokens passed as a previous transcription (after `<|startofprev|>` token) to the all processing windows

The Whisper model can use that context to better understand the speech and maintain a consistent writing style. However, prompts do not need to be genuine transcripts from prior audio segments. Such prompts can be used to steer the model to use particular spellings or styles:

```python
result = pipe.generate(raw_speech)
#  He has gone and gone for good answered Paul Icrom who...

result = pipe.generate(raw_speech, initial_prompt="Polychrome")
#  He has gone and gone for good answered Polychrome who...
```

## Continuous batching with Qwen3-ASR

Qwen3-ASR model supports `ContinuousBatchingPipeline` API. The model's chat template is designed for transcription: it discards user text and assistant turns, so questions about earlier audio, translation instructions, and multi-turn audio conversations are not supported.

```python
import librosa
import openvino as ov
import openvino_genai as genai

wav_file_paths = ["audio_1.wav", "audio_2.wav"]
pipe = genai.ContinuousBatchingPipeline(model_dir, genai.SchedulerConfig(), "CPU")
config = genai.GenerationConfig()
handles = []
for request_id, path in enumerate(wav_file_paths):
    samples, _ = librosa.load(path, sr=16000)
    handle = pipe.add_request(
        request_id, "", images=[], videos=[], audios=[ov.Tensor(samples)],
        generation_config=config,
    )
    handles.append(handle)
while pipe.has_non_finished_requests():
    pipe.step()
for handle in handles:
    print(pipe.get_tokenizer().decode(handle.read_all()[0].generated_ids))
```

### Troubleshooting

#### Empty or rubbish output

Example output:
```
----------------
```

To resolve this ensure that audio data has a 16k Hz sampling rate. You can use the recorder.py provided to record or use FFmpeg to convert the audio to the required format.
