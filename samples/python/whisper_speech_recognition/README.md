# Whisper automatic speech recognition sample

This example showcases inference of speech recognition Whisper Models. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `openvino_genai.WhisperPipeline` and uses audio file in wav format as an input source.

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

Download example audio file: https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav

Or you can use the [`recorder.py`](recorder.py) script. The script records 5 seconds of audio from the microphone. 

To install `PyAudio` dependency follow the [installation instructions](https://pypi.org/project/PyAudio/).

To run the script:
```
python recorder.py
```

## Run the Whisper model

Install [deployment-requirements.txt](../../deployment-requirements.txt) via `pip install -r ../../deployment-requirements.txt` and then, run a sample:

`python whisper_speech_recognition.py whisper-base how_are_you_doing_today.wav`

Output:
```
 How are you doing today?
timestamps: [0, 2] text:  How are you doing today?
```

Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-recognition-models-whisper-based) for more details.

# Whisper pipeline usage

```python
import openvino_genai
import librosa

def read_wav(filepath):
    raw_speech, samplerate = librosa.load(filepath, sr=16000)
    return raw_speech.tolist()

pipe = openvino_genai.WhisperPipeline(model_dir, "CPU")
# Pipeline expects normalized audio with Sample Rate of 16kHz
raw_speech = read_wav('how_are_you_doing_today.wav')
result = pipe.generate(raw_speech)
#  How are you doing today?
```

### Transcription

Whisper pipeline predicts the language of the source audio automatically.

```python
raw_speech = read_wav('how_are_you_doing_today.wav')
result = pipe.generate(raw_speech)
#  How are you doing today?

raw_speech = read_wav('fr_sample.wav')
result = pipe.generate(raw_speech)
#  Il s'agit d'une entité très complexe qui consiste...
```

If the source audio languange is know in advance, it can be specified as an argument to `generate` method:

```python
raw_speech = read_wav("how_are_you_doing_today.wav")
result = pipe.generate(raw_speech, language="<|en|>")
#  How are you doing today?

raw_speech = read_wav("fr_sample.wav")
result = pipe.generate(raw_speech, language="<|fr|>")
#  Il s'agit d'une entité très complexe qui consiste...
```

### Translation

By default, Whisper performs the task of speech transcription, where the source audio language is the same as the target text language. To perform speech translation, where the target text is in English, set the task to "translate":

```python
raw_speech = read_wav("fr_sample.wav")
result = pipe.generate(raw_speech, task="translate")
# It is a very complex entity that consists...
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
