# Text-to-speech pipeline sample

This example demonstrates how to use the openvino_genai.Text2SpeechPipeline in Python to convert input text into speech.
You can specify a target voice using a speaker embedding vector that captures the desired voice characteristics.
Additionally, you can choose the inference device (e.g., CPU, GPU) to control where the model runs.

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

Install [../../export-requirements.txt](../../export-requirements.txt) to convert a model.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
optimum-cli export openvino --model microsoft/speecht5_tts --model-kwargs "{\"vocoder\": \"microsoft/speecht5_hifigan\"}" speecht5_tts
```

**Note:** OpenVINO GenAI speech generation supports multiple backends.
For `SpeechT5 TTS`, model export requires a vocoder via `--model-kwargs` in JSON format.
For `Kokoro`, use a directory containing `openvino_model.xml/.bin` and `config.json`.

## Prepare speaker embedding file

To generate speech using the SpeechT5 TTS model, you can specify a target voice by providing a speaker embedding file.
This file must contain 512 32-bit floating-point values that represent the voice characteristics of the target speaker.
The model will use these characteristics to synthesize the input text in the specified voice.

If no speaker embedding is provided, the model will default to a built-in speaker for speech generation.

You can generate a speaker embedding using
the [`create_speaker_embedding.py`](../../python/speech_generation/create_speaker_embedding.py) script.
This script records 5 seconds of audio from your microphone and extracts a speaker embedding vector from the recording.

To run the script:

```
python create_speaker_embedding.py
```

## Run Text-to-speech sample

Follow [Get Started with Samples](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/get-started-demos.html)
to run the sample.

SpeechT5 example:

`text-to-speech speecht5_tts "Hello OpenVINO GenAI" speaker_embedding.bin --speech_model_type speecht5_tts`

Kokoro example (voice-based, no speaker embedding file):

`text-to-speech standalone_python_ov/Kokoro-82M "Hello from Kokoro in OpenVINO GenAI" --speech_model_type kokoro --voice af_heart --language en-us`

It generates `output_audio.wav` file containing the phrase `Hello OpenVINO GenAI` spoken in the target voice.

Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-generation-models) for more details.

# Text-to-speech pipeline usage

```c++
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

ov::genai::Text2SpeechPipeline pipe(models_path, device);
gen_speech = pipe.generate(prompt, speaker_embedding, ov::AnyMap{{"speech_model_type", "speecht5_tts"}});

// Kokoro voice-based generation (speaker embedding not required)
gen_speech = pipe.generate(prompt,
						   ov::Tensor(),
						   ov::AnyMap{{"speech_model_type", "kokoro"},
									  {"voice", "af_heart"},
									  {"language", "en-us"}});

auto speech = gen_speech.speeches[0];
// speech tensor contains the waveform of the spoken phrase
```
