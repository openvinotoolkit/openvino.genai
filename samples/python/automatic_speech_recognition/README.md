# Automatic speech recognition samples

These samples demonstrate speech transcription and word-level timestamps using OpenVINO GenAI's `ASRPipeline`. They cover Whisper speech recognition and Qwen3-ASR with its companion forced aligner.

## Prepare the models

Install the export dependencies to convert models to OpenVINO format. The `--upgrade-strategy eager` option, together with `-U`, allows pip to upgrade dependencies even when their installed versions satisfy the requirements, which can be necessary for Optimum Intel.

```sh
pip install -U --upgrade-strategy eager -r ../../export-requirements.txt
```

Install additional libraries only for the model families you want to export. These packages are needed for both Optimum CLI and Python exports, but not for running models already converted to OpenVINO format.

For **Qwen3-ASR or its forced aligner**:

```sh
pip install qwen-asr "transformers==4.57.6"
```

For **Fun-ASR-Nano**:

```sh
pip install funasr "transformers==4.57.6"
```


### Export the models using Optimum CLI

#### Whisper

Export the Whisper model:

```sh
optimum-cli export openvino --trust-remote-code --model openai/whisper-base whisper-base
```

#### Qwen3-ASR and Forced Aligner

Qwen3-ASR performs speech transcription, while its companion Qwen3 Forced Aligner provides word-level timestamps. To use word timestamps, export both models into separate directories. For standalone alignment of an existing transcript, only the forced-aligner model is needed at runtime.

```sh
optimum-cli export openvino --trust-remote-code --model Qwen/Qwen3-ASR-0.6B \
    --task automatic-speech-recognition-with-past qwen3-asr-ov

optimum-cli export openvino --trust-remote-code --model Qwen/Qwen3-ForcedAligner-0.6B \
    --task automatic-speech-recognition-with-past qwen3-forced-aligner-ov
```

#### FunASR

Export Fun-ASR-Nano:

```sh
optimum-cli export openvino --trust-remote-code \
    --model FunAudioLLM/Fun-ASR-Nano-2512 fun-asr-nano-ov
```

### Export the models using Python

The same models can be exported programmatically with `main_export()`:

```python
from optimum.exporters.openvino import main_export

# Whisper
main_export(
    model_name_or_path="openai/whisper-base",
    output="whisper-base",
    trust_remote_code=True,
    convert_tokenizer=True,
)

# Qwen3-ASR
main_export(
    model_name_or_path="Qwen/Qwen3-ASR-0.6B",
    output="qwen3-asr-ov",
    task="automatic-speech-recognition-with-past",
    trust_remote_code=True,
    convert_tokenizer=True,
)

# Qwen3 Forced Aligner
main_export(
    model_name_or_path="Qwen/Qwen3-ForcedAligner-0.6B",
    output="qwen3-forced-aligner-ov",
    task="automatic-speech-recognition-with-past",
    trust_remote_code=True,
    convert_tokenizer=True,
)

# Fun-ASR-Nano
main_export(
    model_name_or_path="FunAudioLLM/Fun-ASR-Nano-2512",
    output="fun-asr-nano-ov",
    task="automatic-speech-recognition-with-past",
    trust_remote_code=True,
    convert_tokenizer=True,
)
```

For whisper we can also use this snippet:

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

The `convert_tokenizer=True` option exports the tokenizer and detokenizer to OpenVINO format. `main_export()` saves the converted models and tokenizer files directly to the specified output directories, without requiring separate `save_pretrained()` or `export_tokenizer()` calls.

For more architectures and conversion options, see [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/) and [Model Preparation](https://openvinotoolkit.github.io/openvino.genai/docs/guides/model-preparation/convert-to-openvino/).

## Prepare audio

The Python samples load audio with `librosa`, resampling it to 16 kHz and downmixing multichannel audio to mono. Supported input formats depend on the installed audio decoders.

You can use the example recording [how_are_you_doing_today.wav](https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav).

Alternatively, install [PyAudio](https://pypi.org/project/PyAudio/) and record your own audio using [`recorder.py`](recorder.py):

```sh
python recorder.py
```

The recorder saves a five-second recording to `output.wav`.

## Run the samples

Install the runtime dependencies:

```sh
pip install -r ../../deployment-requirements.txt
```

The samples run on `CPU` by default. You can optionally specify another device, such as `GPU`, as the last argument.

### Automatic speech recognition

The `automatic_speech_recognition.py` sample demonstrates speech transcription with Whisper, including timestamps for recognized text and individual words.

```sh
python automatic_speech_recognition.py whisper-base how_are_you_doing_today.wav
```

Example output (excerpt):

```text
 How are you doing today?
timestamps: [0.00, 2.00] text:  How are you doing today?
[0.00, 0.58]:  How
[0.58, 0.68]:  are
[0.68, 0.80]:  you
[0.80, 1.04]:  doing
[1.04, 1.40]:  today?
```

The sample also prints word-level timestamps.

### Forced-aligner word timestamps

The `forced_aligner_timestamps_generation.py` sample uses Qwen3-ASR to transcribe audio and automatically detect its language. It then uses the companion Qwen3 Forced Aligner to generate word-level timestamps.

Provide the exported Qwen3-ASR and forced-aligner model directories:

```sh
python forced_aligner_timestamps_generation.py qwen3-asr-ov qwen3-forced-aligner-ov fr_audio.wav
```

Example output:

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


## ASR pipeline usage

The `ASRPipeline` API provides a common interface for speech recognition models. The following example loads an exported ASR model and transcribes audio:

```python
import openvino_genai
import librosa

def read_wav(filepath):
    raw_speech, samplerate = librosa.load(filepath, sr=16000)
    return raw_speech.tolist()

model_dir = "whisper-base"  # Or another supported exported ASR model
pipe = openvino_genai.ASRPipeline(model_dir, "CPU")

# Pipeline expects normalized audio with a sample rate of 16 kHz
raw_speech = read_wav("how_are_you_doing_today.wav")
result = pipe.generate(raw_speech)
print(result.texts[0])
# How are you doing today?
```

### Transcription

Multilingual Whisper and Qwen3-ASR models can automatically detect the language of the input audio.

```python
result = pipe.generate(raw_speech)
# How are you doing today?

raw_speech = read_wav('fr_sample.wav')
result = pipe.generate(raw_speech)
#  Il s'agit d'une entité très complexe qui consiste...
```

If the language is known in advance, it can be specified using the `language` argument to `generate` method:

```python
result = pipe.generate(raw_speech, language="en")
# How are you doing today?

raw_speech = read_wav("fr_sample.wav")
result = pipe.generate(raw_speech, language="fr")
# Il s'agit d'une entité très complexe qui consiste...
```

Whisper uses language codes such as `"en"` and `"zh"`, while Qwen3-ASR and Fun-ASR-Nano use language names such as `"English"` and `"Chinese"`.

Language identification depends on the model architecture. For example, Fun-ASR-Nano supports transcription without a specified language but does not report an automatically detected language.

### Timestamps prediction

Whisper can predict timestamps for recognized speech segments. Enable this with `return_timestamps=True`:

```python
raw_speech = read_wav("how_are_you_doing_today.wav")
result = pipe.generate(raw_speech, return_timestamps=True)

for chunk in result.chunks[0]:
    print(f"timestamps: [{chunk.start_ts:.2f}, {chunk.end_ts:.2f}] text: {chunk.text}")
# timestamps: [0.00, 2.00] text:  How are you doing today?
```

For word-level timestamps, the `word_timestamps` property must also be enabled when constructing the Whisper pipeline:

```python
pipe = openvino_genai.ASRPipeline(model_dir, "CPU", word_timestamps=True)
result = pipe.generate(raw_speech, word_timestamps=True)

for word in result.words[0]:
    print(f"[{word.start_ts:.2f}, {word.end_ts:.2f}]: {word.text}")
```

For Qwen3-ASR, word-level timestamps are generated using the companion Qwen3 Forced Aligner, as demonstrated in `forced_aligner_timestamps_generation.py`.

The `ASRForcedAligner` API can also generate word-level timestamps for a supplied transcription:

```python
aligner = openvino_genai.ASRForcedAligner(aligner_model_dir, "CPU")
raw_speech = read_wav("BAC009S0764W0121.wav")

words = aligner.align(
    raw_speech,
    "甚至出现交易几乎停滞的情况。",
    language="Chinese",
)

for word in words:
    print(f"[{word.start_ts:.2f}, {word.end_ts:.2f}]: {word.text}")
```

Example output:

```text
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

### Whisper-specific features

#### Translation

Whisper normally performs transcription, producing text in the language of the source audio. To translate non-English speech into English, set the task to `"translate"`:

```python
raw_speech = read_wav("fr_sample.wav")
result = pipe.generate(raw_speech, task="translate")
# It is a very complex entity that consists...
```

#### Long-form audio transcription

Whisper processes audio in windows of up to 30 seconds. For longer recordings, the pipeline uses a sequential chunking algorithm with a sliding window to transcribe successive portions of the audio.

#### Initial prompt and hotwords

Whisper supports the `initial_prompt` and `hotwords` generation arguments:

- `initial_prompt`: provides context as a previous transcription for the first processing window.
- `hotwords`: provides context for subsequent processing windows as well.

These arguments help guide transcription and maintain consistent spelling and style. The supplied text does not need to be an actual earlier transcription. For example, an initial prompt can help recognize an uncommon name:

```python
result = pipe.generate(raw_speech, initial_prompt="Polychrome")
```

### Troubleshooting

#### Empty or incorrect transcription

If the output is empty or unintelligible, check the input audio and its sample rate. `ASRPipeline` expects normalized audio sampled at 16 kHz.

The `read_wav()` helper above uses `librosa` to resample audio to 16 kHz. You can also use `recorder.py` to create a compatible recording or FFmpeg to convert an existing recording.

For more information, see the [Automatic Speech Recognition guide](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/speech-recognition/) and [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/).


