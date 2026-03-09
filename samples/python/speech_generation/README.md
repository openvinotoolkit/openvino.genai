# Text-to-speech pipeline samples

This folder provides multiple Python samples around `openvino_genai.Text2SpeechPipeline`:

- `text2speech.py`: minimal end-to-end text → audio example
- `kokoro_generate_from_tokens.py`: Kokoro token-based generation via `generate_from_tokens`

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

Install [../../export-requirements.txt](../../export-requirements.txt) to convert a model.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
```

Then, run the export with Optimum CLI:

```sh
optimum-cli export openvino --model microsoft/speecht5_tts --model-kwargs "{\"vocoder\": \"microsoft/speecht5_hifigan\"}" speecht5_tts
```

Alternatively, you can do it in Python code:

```python
from optimum.exporters.openvino.convert import export_tokenizer
from optimum.intel import OVModelForTextToSpeechSeq2Seq
from transformers import AutoTokenizer

output_dir = "speecht5_tts"

model = OVModelForTextToSpeechSeq2Seq.from_pretrained("microsoft/speecht5_tts", vocoder="microsoft/speecht5_hifigan", export=True)
model.save_pretrained(output_dir)

tokenizer = AutoTokenizer.from_pretrained("microsoft/speecht5_tts")
export_tokenizer(tokenizer, output_dir)
```

**Note:** OpenVINO GenAI speech generation supports multiple backends.
For `SpeechT5 TTS`, model export requires a vocoder via `--model-kwargs` in JSON format.
For `Kokoro`, use a directory containing `openvino_model.xml/.bin` and `config.json`.

## Prepare speaker embedding file

To generate speech using the SpeechT5 TTS model, you can specify a target voice by providing a speaker embedding file.
This file must contain 512 32-bit floating-point values that represent the voice characteristics of the target speaker.
The model will use these characteristics to synthesize the input text in the specified voice.

If no speaker embedding is provided, the model will default to a built-in speaker for speech generation.

You can generate a speaker embedding using the [`create_speaker_embedding.py`](create_speaker_embedding.py) script.
This script records 5 seconds of audio from your microphone and extracts a speaker embedding vector from the recording.

To run the script:

```
python create_speaker_embedding.py
```

## Run text-to-speech samples

Install [deployment-requirements.txt](../../deployment-requirements.txt)
via `pip install -r ../../deployment-requirements.txt` and then run one of the following samples.

### 1) Minimal E2E sample (`text2speech.py`)

SpeechT5 example (with speaker embedding):

`python text2speech.py --speaker_embedding_file_path speaker_embedding.bin speecht5_tts "Hello OpenVINO GenAI"`

Kokoro example (plain text input):

`python text2speech.py --voice af_heart --language en-us  Kokoro-82M "Hello from Kokoro in OpenVINO GenAI"`

### 2) Kokoro token sample (`kokoro_generate_from_tokens.py`)

**Note:** `generate_from_tokens` is supported only by Kokoro backend. SpeechT5 backend does not support this API.

This sample demonstrates the intended application flow: run Python Misaki G2P first, map Misaki tokens into
`openvino_genai.SpeechToken`, then call `generate_from_tokens`.

Install Misaki (if not already installed):

`pip install misaki`

Single sequence token synthesis using `generate_from_tokens`:

`python kokoro_generate_from_tokens.py Kokoro-82M "Hello from Kokoro via Misaki tokens"`

All samples generate WAV output files.

Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-generation-models) for more details.

# Text-to-speech pipeline usage

```python
import openvino_genai

pipe = openvino_genai.Text2SpeechPipeline(model_dir, device)
result = pipe.generate("Hello OpenVINO GenAI", speaker_embedding)

# Kokoro voice-based text generation (speaker embedding not required)
result = pipe.generate("Hello from Kokoro", None, voice="af_heart", language="en-us")

# Kokoro generation from token
# NOTE: generate_from_tokens is supported only by Kokoro backend.
tokens = [
    openvino_genai.SpeechToken(phonemes="həlˈoʊ", whitespace=True, text="Hello"),
    openvino_genai.SpeechToken(phonemes="wˈɝld", whitespace=False, text="world"),
]
result = pipe.generate_from_tokens(tokens, None, voice="af_heart", language="en-us")
speech = result.speeches[0]
# speech tensor contains the waveform of the spoken phrase
```
