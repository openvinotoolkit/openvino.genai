# Automatic speech recognition sample

This example showcases inference of speech recognition models. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `openvino_genai.ASRPipeline` and uses audio file in wav format as an input source.

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

Install [../../export-requirements.txt](../../export-requirements.txt) to convert a model.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
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

## Run

Install [deployment-requirements.txt](../../deployment-requirements.txt) via `pip install -r ../../deployment-requirements.txt` and then, run a sample:

`python whisper_speech_recognition.py whisper-base how_are_you_doing_today.wav`

Output:
```
 How are you doing today?
timestamps: [0.00, 2.00] text:  How are you doing today?
```

Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-recognition-models-whisper-based) for more details.

# Voxtral Realtime streaming

Voxtral Realtime uses `ASRPipeline.create_stream()` to transcribe audio incrementally. The streaming model export is
provided by [optimum-intel PR #2075](https://github.com/huggingface/optimum-intel/pull/2075); use an Optimum Intel
version containing that support.

Install the export and runtime dependencies:

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
pip install -r ../../deployment-requirements.txt
pip install --upgrade git+https://github.com/kbalka/optimum-intel.git@voxtral_streaming_asr
```

Export the model to OpenVINO:

```sh
optimum-cli export openvino --trust-remote-code --model mistralai/Voxtral-Mini-4B-Realtime-2602 voxtral-realtime-2602
```

Run the streaming example with a mono or multichannel audio file. Audio is resampled to mono 16 kHz before streaming:

```sh
python voxtral_streaming.py voxtral-realtime-2602 audio.wav
```

The script submits 100 ms audio frames from a producer thread while inference runs and prints transcript deltas as they
arrive. It also reports the generated-token count, interval tokens per second, and average tokens per second after each
transcript update, so changes in throughput are visible during generation. The optional third argument selects the
inference device; it defaults to `CPU`.

Each update also shows the combined retained audio and text KV-cache size in MiB, measured from the cache tensors
after sliding-window cropping. This is not total process or peak inference memory: convolution caches, inference
request buffers, model weights and temporary allocations are excluded.
Each update and the final summary also show audio and text KV windows as `retained/max (percent%)`,
in cache positions after cropping, for example `audio window: 750/750 (100.0%)` and
`text window: 256/256 (100.0%)`. Maxima come from the loaded model configuration, not hardcoded defaults.
These are audio-aligned cache positions, not word counts. Window occupancy is available without `--profile`.

Cache updates reuse stream-owned tensor buffers once their capacity is sufficient. When a sliding window evicts
positions, retained keys are rephased in parallel across batch/head slices and retained values are copied in
contiguous head-sized blocks. Window sizes and RoPE-relative positions are unchanged; cache maintenance still
touches the retained window on each eviction, so bounded cache memory does not imply constant-cost decoding.

The next line breaks down interval milliseconds per generated step into `audio infer`, `text infer`, `audio KV`,
and `text KV`. Inference timings cover the respective `infer()` calls. KV timings cover retained tensor allocation,
copying, sliding-window cropping and key rephasing. They exclude convolution cache copies, feature extraction,
tensor setup, sampling and callback/output overhead, so their sum is not the full step latency. The first interval
includes prefill; the final summary reports whole-stream averages, including steps after the last transcript update.

Add `--profile` to enable OpenVINO operation profiling inside both inference requests:

```sh
python voxtral_streaming.py voxtral-realtime-2602 audio.wav --profile
```

Each update lists the eight most expensive operation groups for audio and text inference using the latest plugin
counters in milliseconds, grouped by node type and execution type. CPU counters are running averages across node
executions, not interval times. They must not be accumulated, differenced or divided by generated-token counts.
The final summary shows the final plugin snapshot, not an independently measured whole-stream average. This can
distinguish attention, concatenation, reorder and matrix multiplication when the device plugin reports them
separately. Fused kernels cannot be split into their original operations by these counters. Plugin real-time
counters can overlap and omit runtime overhead, so their sum need not equal `infer()` wall time.
Profiling and counter collection add overhead; profiling is disabled by default.

The three most expensive executed Broadcast nodes per component are also listed with their full node names,
execution types, compiled-graph input/output partial shapes and consumer names. `?` denotes a dynamic dimension,
not a measured tensor size. Plugin nodes without an exact compiled-graph name match are marked with
`graph shapes unavailable`. For example, `[1]` to `[?,8,4,?,128]` describes scalar expansion to the shape used for
four-way expansion of KV heads before attention; consumer names locate the associated attention layer. This identifies the
graph operation, but does not measure its instantaneous runtime shape or isolate all downstream costs.

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

### Troubleshooting

#### Empty or rubbish output

Example output:
```
----------------
```

To resolve this ensure that audio data has a 16k Hz sampling rate. You can use the recorder.py provided to record or use FFmpeg to convert the audio to the required format.
