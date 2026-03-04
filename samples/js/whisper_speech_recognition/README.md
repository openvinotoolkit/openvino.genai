# Whisper automatic speech recognition sample (JavaScript)

This example showcases inference of speech recognition Whisper models using the JavaScript/Node.js API. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `WhisperPipeline` from `openvino-genai-node` and uses a WAV audio file as input. Audio is decoded via **node-wav** and converted to 16 kHz mono Float32.

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

The sample uses **node-wav** to decode audio, so pass a WAV file; it will be converted to 16 kHz mono automatically.

Download example WAV: https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav

Or record from the microphone (16 kHz mono WAV):

- **JavaScript:** `node whisper_speech_recognition/recorder.js` (optional: `--duration 5 --output output.wav`). Uses the **naudiodon** npm package (PortAudio bindings), so no system tools are required; run `npm install` in `samples/js` first.
- **Python:** [recorder.py](../../python/whisper_speech_recognition/recorder.py) (requires `pip install pyaudio`).

## Run the Whisper model

From the `samples/js` directory, install dependencies (if not already done):

```bash
npm install
```

If you use the master branch, you may need to [build openvino-genai-node from source](../../src/js/README.md#build-bindings) first.

Run the sample:

```bash
node whisper_speech_recognition/whisper_speech_recognition.js whisper-base how_are_you_doing_today.wav
```

Optional third argument is the device (default: CPU):

```bash
node whisper_speech_recognition/whisper_speech_recognition.js whisper-base how_are_you_doing_today.wav GPU
```

Expected output:

```
 How are you doing today?
timestamps: [0.00, 2.00] text:  How are you doing today?
[0.00, 0.xx]:
[0.xx, 0.xx]: How
...
```

Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-recognition-models-whisper-based) for more details.

## Whisper pipeline usage (JavaScript)

```javascript
import { WhisperPipeline } from 'openvino-genai-node';
import { readFileSync } from 'node:fs';

// Helper: read WAV as Float32Array at 16 kHz (see whisper_speech_recognition.js for full readWav)
function readWav(filepath) {
  // ... parse WAV, normalize to [-1, 1], resample to 16 kHz if needed
  return float32Samples;
}

const pipeline = await WhisperPipeline(modelDir, "CPU");
const rawSpeech = readWav('how_are_you_doing_today.wav');
const result = await pipeline.generate(rawSpeech);
console.log(result.texts[0]);
//  How are you doing today?
```

### Transcription

Specify language in the generation config for better accuracy:

```javascript
const generationConfig = { language: "<|en|>", task: "transcribe" };
const result = await pipeline.generate(rawSpeech, { generationConfig });
```

### Translation

To translate non-English speech to English:

```javascript
const generationConfig = { task: "translate" };
const result = await pipeline.generate(rawSpeech, { generationConfig });
```

### Timestamps prediction

Segment-level timestamps:

```javascript
const generationConfig = { return_timestamps: true, language: "<|en|>", task: "transcribe" };
const result = await pipeline.generate(rawSpeech, { generationConfig });
for (const chunk of result.chunks ?? []) {
  console.log(`timestamps: [${chunk.startTs.toFixed(2)}, ${chunk.endTs.toFixed(2)}] text: ${chunk.text}`);
}
```

### Word-level timestamps

Pass `word_timestamps: true` in the pipeline constructor, then in the generation config:

```javascript
const pipeline = await WhisperPipeline(modelDir, "CPU", { word_timestamps: true });
const generationConfig = { return_timestamps: true, word_timestamps: true, language: "<|en|>", task: "transcribe" };
const result = await pipeline.generate(rawSpeech, { generationConfig });
for (const w of result.words ?? []) {
  console.log(`[${w.startTs.toFixed(2)}, ${w.endTs.toFixed(2)}]: ${w.word}`);
}
```

### Initial prompt and hotwords

```javascript
let result = await pipeline.generate(rawSpeech);
//  He has gone and gone for good answered Paul Icrom who...

const generationConfig = { initial_prompt: "Polychrome" };
result = await pipeline.generate(rawSpeech, { generationConfig });
//  He has gone and gone for good answered Polychrome who...
```

### Troubleshooting

#### Empty or rubbish output

Ensure the input is a valid WAV file. The sample's `readAudio` helper converts it to 16 kHz mono before inference.

For non-WAV sources (MP3, M4A, FLAC), convert to WAV first with your preferred tool.

#### NPU device

For NPU, pass `STATIC_PIPELINE: true` in the pipeline properties:

```javascript
const pipeline = await WhisperPipeline(modelDir, "NPU", { word_timestamps: true, STATIC_PIPELINE: true });
```
