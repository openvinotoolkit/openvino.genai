# Automatic speech recognition sample (JavaScript)

This example showcases inference of speech recognition models. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `ASRPipeline` and uses an audio file in wav format as an input source. Audio conversion is performed by a custom helper in `wav_utils.js` (PCM16 mono/stereo at 16 kHz) to align numerical behavior with the C++ and Python sample paths.

`ASRPipeline` dispatches to the model implementation based on the model's `config.json` (for example Whisper or Qwen3-ASR).

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
node automatic_speech_recognition/automatic_speech_recognition.js whisper-base how_are_you_doing_today.wav
```

Optional third argument is the device (default: CPU):

```bash
node automatic_speech_recognition/automatic_speech_recognition.js whisper-base how_are_you_doing_today.wav GPU
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

# ASR pipeline usage

```javascript
import { ASRPipeline } from 'openvino-genai-node';
import { readFileSync } from 'node:fs';
import { decode } from 'node-wav';

const pipeline = await ASRPipeline(modelDir, "CPU");
const rawSpeechBuffer = readFileSync(audioFilePath);
const rawSpeech = decode(rawSpeechBuffer).channelData[0];
const result = await pipeline.generate(rawSpeech);
console.log(result.texts[0]);
//  How are you doing today?
```

### Transcription

Multilingual models predict the language of the source audio automatically.

If the source audio language is known in advance, it can be specified in the generation config:

```javascript
// In the form of "<|en|>" for Whisper models, "English" for Qwen3-ASR models.
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

The model can predict timestamps. For sentence-level timestamps, pass the `return_timestamps` argument. The result exposes `chunks` grouped per input, so use `chunks[0]` for a single audio input:

```javascript
const generationConfig = { return_timestamps: true, language: "<|en|>", task: "transcribe" };
const result = await pipeline.generate(rawSpeech, { generationConfig });
for (const chunk of result.chunks?.[0] ?? []) {
  console.log(`timestamps: [${chunk.startTs.toFixed(2)}, ${chunk.endTs.toFixed(2)}] text: ${chunk.text}`);
}
```

### Word-level timestamps

Word-level timestamps are supported by Whisper models. Pass `word_timestamps: true` in the pipeline constructor, then in the generation config:

```javascript
const pipeline = await ASRPipeline(modelDir, "CPU", { word_timestamps: true });
const generationConfig = { return_timestamps: true, word_timestamps: true, language: "<|en|>", task: "transcribe" };
const result = await pipeline.generate(rawSpeech, { generationConfig });
for (const word of result.words?.[0] ?? []) {
  console.log(`[${word.startTs.toFixed(2)}, ${word.endTs.toFixed(2)}]: ${word.text}`);
}
```

### Initial prompt and hotwords

The ASR pipeline has `initial_prompt` and `hotwords` generate arguments for Whisper models:
* `initial_prompt`: initial prompt tokens passed as a previous transcription (after `<|startofprev|>` token) to the first processing window
* `hotwords`: hotwords tokens passed as a previous transcription (after `<|startofprev|>` token) to all processing windows
