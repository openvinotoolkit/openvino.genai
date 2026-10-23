# Automatic speech recognition samples

These samples demonstrate speech recognition using `openvino_genai.ASRPipeline`, with WAV audio files as input. They keep configuration options minimal to encourage users to explore and customize the source code, for example, by experimenting with GPU inference.

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
Alternatively, you can export the model using Python:

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

Alternatively, use [`recorder.py`](recorder.py) to record five seconds of audio from your microphone.

To install `PyAudio` dependency follow the [installation instructions](https://pypi.org/project/PyAudio/).

## Run the samples

Install [deployment-requirements.txt](../../deployment-requirements.txt) via `pip install -r ../../deployment-requirements.txt` and then choose one of the below samples to run.

### Automatic speech recognition

```sh
python automatic_speech_recognition.py whisper-base how_are_you_doing_today.wav
```

Output:

```text
 How are you doing today?
timestamps: [0.00, 2.00] text:  How are you doing today?
```

### Standalone Forced-aligner word timestamps

```sh
python forced_aligner_timestamps_generation.py fun-asr-nano-ov qwen3-forced-aligner-ov zh_sample.wav --language Chinese
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

All supported ASR models can transcribe audio without specifying a language. Multilingual Whisper and Qwen3-ASR also detect and report the input language, whereas Fun-ASR-Nano does not. Therefore, in the standalone forced-aligner sample, `--language` is optional with Qwen3-ASR but required with Fun-ASR-Nano.

```python
raw_speech = read_wav('how_are_you_doing_today.wav')
result = pipe.generate(raw_speech)
# How are you doing today?

raw_speech = read_wav('fr_sample.wav')
result = pipe.generate(raw_speech)
#  Il s'agit d'une entité très complexe qui consiste...
```

If the language is known, pass it to `generate()`:

```python
raw_speech = read_wav("how_are_you_doing_today.wav")
result = pipe.generate(raw_speech, language="en")
# How are you doing today?

raw_speech = read_wav("fr_sample.wav")
result = pipe.generate(raw_speech, language="fr")
# Il s'agit d'une entité très complexe qui consiste...
```

Whisper uses language codes such as `"en"` and `"zh"`, while Qwen3-ASR uses language names such as `"English"` and `"Chinese"`. The documented language hints for Fun-ASR-Nano are `"中文"`, `"英文"`, and `"日文"`.


### Timestamps prediction

Whisper can predict timestamps for recognized speech segments using `return_timestamps=True`. To generate word-level timestamps, pass `word_timestamps=True` both when constructing the Whisper pipeline and when calling `generate()`.

```python
raw_speech = read_wav("how_are_you_doing_today.wav")
result = pipe.generate(raw_speech, return_timestamps=True)

for chunk in result.chunks[0]:
    print(f"timestamps: [{chunk.start_ts:.2f}, {chunk.end_ts:.2f}] text: {chunk.text}")
# timestamps: [0.00, 2.00] text:  How are you doing today?
```

Qwen3-ASR can produce word-level timestamps if integrated with its forced aligner into ASRPipeline:

```python
aligner = openvino_genai.ASRForcedAligner(forced_aligner_model_dir, "CPU")
pipe = openvino_genai.ASRPipeline(qwen3_asr_model_dir, "CPU", forced_aligner=aligner)

raw_speech = read_wav("fr_audio.wav")
result = pipe.generate(raw_speech, word_timestamps=True)

print(f"Detected language: {result.languages[0]}")
print(f"Transcription: {result.texts[0]}")
print("\nWord timestamps:")

for word in result.words[0]:
    print(f"[{word.start_ts:.2f}, {word.end_ts:.2f}]: {word.text}")
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

```python
raw_speech = read_wav("fr_sample.wav")
result = pipe.generate(raw_speech, task="translate")
# It is a very complex entity that consists...
```

#### Long-form audio transcription

Whisper processes audio in windows of up to 30 seconds. For longer recordings, the pipeline uses a sequential chunking algorithm with a sliding window to transcribe successive portions of the audio.

#### Initial prompt and hotwords

Whisper supports the `initial_prompt` and `hotwords` generation arguments:

- `initial_prompt`: initial prompt tokens passed as a previous transcription (after `<|startofprev|>` token) to the first processing window
- `hotwords`: hotwords tokens passed as a previous transcription (after `<|startofprev|>` token) to the all processing windows

These arguments help guide transcription and maintain consistent writing style. The supplied text does not need to be an actual earlier transcription. For example, an initial prompt can help recognize an uncommon name:

```python
result = pipe.generate(raw_speech)
#  He has gone and gone for good answered Paul Icrom who...

result = pipe.generate(raw_speech, initial_prompt="Polychrome")
#  He has gone and gone for good answered Polychrome who...
```

### Troubleshooting

#### Empty or incorrect transcription

If the output is empty or unintelligible, check the input audio and its sample rate. `ASRPipeline` expects normalized audio sampled at 16 kHz.

Example output:
```
----------------
```

You can also use `recorder.py` to create a compatible recording or FFmpeg to convert an existing recording.