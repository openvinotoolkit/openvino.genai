# Text-to-speech pipeline sample

This example demonstrates how to use the `Text2SpeechPipeline` from `openvino-genai-node` to convert input text into speech. The application accepts a text string, runs TTS inference, and writes the output to a WAV file using the `node-wav` package.

You can specify a target voice using a speaker embedding binary file that captures the desired voice characteristics. Additionally, you can choose the inference device (e.g., CPU, GPU) to control where the model runs.

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

It's not required to install [../../export-requirements.txt](../../export-requirements.txt) for deployment if the model has already been exported.

```sh
pip install --upgrade-strategy eager -r <GENAI_ROOT_DIR>/samples/export-requirements.txt
```

Then, run the export with Optimum CLI:

```sh
optimum-cli export openvino --model microsoft/speecht5_tts --model-kwargs "{\"vocoder\": \"microsoft/speecht5_hifigan\"}" speecht5_tts
```

**Note:** Currently, text-to-speech in OpenVINO GenAI supports the `SpeechT5 TTS` model.
When exporting the model, you must specify a vocoder using the `--model-kwargs` option in JSON format.

## Prepare speaker embedding file (optional)

To generate speech using the SpeechT5 TTS model, you can specify a target voice by providing a speaker embedding file.
This file must contain 512 32-bit floating-point values that represent the voice characteristics of the target speaker.
The model will use these characteristics to synthesize the input text in the specified voice.

If no speaker embedding is provided, the model will default to a built-in speaker for speech generation.

You can generate a speaker embedding using the Python [`create_speaker_embedding.py`](../../python/speech_generation/create_speaker_embedding.py) script from the Python samples.

## Run

From the `samples/js` directory, install dependencies (if not already done):

```bash
npm install
```

If you use the master branch, you may need to [build openvino-genai-node from source](../../../src/js/README.md#build-bindings) first.

Run the sample:

```bash
node speech_generation/text2speech.js speecht5_tts "Hello OpenVINO GenAI"
```

With a speaker embedding:

```bash
node speech_generation/text2speech.js speecht5_tts "Hello OpenVINO GenAI" --speaker_embedding speaker_embedding.bin
```

Optional positional argument for device (default: CPU):

```bash
node speech_generation/text2speech.js speecht5_tts "Hello OpenVINO GenAI" GPU
```

Custom output file path:

```bash
node speech_generation/text2speech.js speecht5_tts "Hello OpenVINO GenAI" --output my_audio.wav
```

Output:

```
[Info] Text successfully converted to audio file "output_audio.wav".

=== Performance Summary ===
Throughput              : 123.45 samples/sec.
Total Generation Time   : 1.234 sec.
```

Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-generation-models) for more details.

# Text-to-speech pipeline usage

```javascript
import { readFile, writeFile } from 'node:fs/promises';
import { encode } from 'node-wav';
import { Text2SpeechPipeline } from 'openvino-genai-node';

const pipeline = await Text2SpeechPipeline(modelDir, "CPU");
const result = await pipeline.generate("Hello OpenVINO GenAI");
// result.speeches[0] is an OpenVINO Tensor with the waveform at 16 kHz
const wavData = encode([result.speeches[0].data], { sampleRate: 16000 });
await writeFile("output_audio.wav", wavData);
```
