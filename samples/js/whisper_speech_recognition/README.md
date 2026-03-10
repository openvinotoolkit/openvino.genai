# Whisper automatic speech recognition sample (JavaScript)

This example showcases inference of speech recognition Whisper Models. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `WhisperPipeline` and uses audio file in wav format as an input source. Audio conversion is performed by a custom helper in `wav_utils.js` (PCM16 mono/stereo at 16 kHz) to align numerical behavior with the C++ and Python sample paths.

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

It's not required to install [../../export-requirements.txt](../../export-requirements.txt) for deployment if the model has already been exported.

```sh
pip install --upgrade-strategy eager -r <GENAI_ROOT_DIR>/samples/requirements.txt
optimum-cli export openvino --trust-remote-code --model openai/whisper-base whisper-base
```

## Prepare audio file

Prepare audio file in wav format with sampling rate 16k Hz.

You can download example audio file: https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav

## Run

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

Output:

```
 How are you doing today?
timestamps: [0.00, 2.00] text:  How are you doing today?
[0.00, 0.xx]:
[0.xx, 0.xx]: How
...
```

Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#speech-recognition-models-whisper-based) for more details.

# Whisper pipeline usage

```javascript
import { WhisperPipeline } from 'openvino-genai-node';
import { readFileSync } from 'node:fs';
import { decode } from 'node-wav';

const pipeline = await WhisperPipeline(modelDir, "CPU");
const rawSpeechBuffer = readFileSync(audioFilePath);
const rawSpeech = decode(rawSpeechBuffer).channelData[0];
const result = await pipeline.generate(rawSpeech);
console.log(result.texts[0]);
//  How are you doing today?
```

### Transcription

Whisper pipeline predicts the language of the source audio automatically.

If the source audio language is known in advance, it can be specified in generation config:

```javascript
const generationConfig = { language: "<|en|>", task: "transcribe" };
const result = await pipeline.generate(rawSpeech, { generationConfig });
```

### Translation

By default, Whisper performs the task of speech transcription, where the source audio language is the same as the target text language. To perform speech translation, where the target text is in English, set the task to "translate":

```javascript
const generationConfig = { task: "translate" };
const result = await pipeline.generate(rawSpeech, { generationConfig });
```

### Timestamps prediction

The model can predict timestamps. For sentence-level timestamps, pass the `return_timestamps` argument:

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

Whisper pipeline has `initial_prompt` and `hotwords` generate arguments:
* `initial_prompt`: initial prompt tokens passed as a previous transcription (after `<|startofprev|>` token) to the first processing window
* `hotwords`: hotwords tokens passed as a previous transcription (after `<|startofprev|>` token) to the all processing windows

The Whisper model can use that context to better understand the speech and maintain a consistent writing style. However, prompts do not need to be genuine transcripts from prior audio segments. Such prompts can be used to steer the model to use particular spellings or styles:

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
