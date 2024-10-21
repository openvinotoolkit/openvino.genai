# Whisper automatic speech recognition sample

This example showcases inference of speech recognition Whisper Models. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `ov::genai::WhisperPipeline` and uses audio file in wav format as an input source.

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

It's not required to install [../../requirements.txt](../../requirements.txt) for deployment if the model has already been exported.

```sh
pip install --upgrade-strategy eager -r ../../requirements.txt
optimum-cli export openvino --trust-remote-code --model openai/whisper-base whisper-base
```

## Prepare audio file

Prepare audio file in wav format with sampling rate 16k Hz.

## Run

`whisper_speech_recognition whisper-base sample.wav`

Output: text transcription of `sample.wav`

Models can be downloaded from [OpenAI HuggingFace](https://huggingface.co/openai).

See [SUPPORTED_MODELS.md](../../../src/docs/SUPPORTED_MODELS.md#whisper-models) for the list of supported models.

### Troubleshooting

#### Empty or rubbish output

Example output:
```
----------------
```

To resolve this ensure that audio data has 16k Hz sampling rate
