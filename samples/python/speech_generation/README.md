# Text-to-speech pipeline sample

This example demonstrates how to use the openvino_genai.Text2SpeechPipeline in Python to convert input text into speech.
You can specify a target voice using a speaker embedding vector that captures the desired voice characteristics.
Additionally, you can choose the inference device (e.g., CPU, GPU) to control where the model runs.

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

**Note:** Currently, text-to-speech in OpenVINO GenAI supports the `SpeechT5 TTS` model. 
When exporting the model, you must specify a vocoder using the `--model-kwargs` option in JSON format.

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

## Run Text-to-speech sample

Install [deployment-requirements.txt](../../deployment-requirements.txt)
via `pip install -r ../../deployment-requirements.txt` and then, run a sample:

`python text2speech.py --speaker_embedding_file_path speaker_embedding.bin speecht5_tts "Hello OpenVINO GenAI"`

It generates `output_audio.wav` file containing the phrase `Hello OpenVINO GenAI` spoken in the target voice.

Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-generation-models) for more details.

# Text-to-speech pipeline usage

```python
import openvino_genai

pipe = openvino_genai.Text2SpeechPipeline(model_dir, device)
result = pipe.generate("Hello OpenVINO GenAI", speaker_embedding)
speech = result.speeches[0]
# speech tensor contains the waveform of the spoken phrase 
```
