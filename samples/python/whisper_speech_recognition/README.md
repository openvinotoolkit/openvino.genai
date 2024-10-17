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

You can prepare an audio file in WAV format with a sampling rate of 16k Hz using the `recorder.py` script. The script will record 5 seconds of audio from the microphone. 

To install dependencies:
```
pip install pyaudio
```
To run the script:
```
python recorder.py
```

## Run the Whisper model

`whisper_speech_recognition whisper-base sample.wav`

Output: text transcription of `sample.wav`

Models can be downloaded from [OpenAI HuggingFace](https://huggingface.co/openai).

Supported Models:
[openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | [openai/whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en) | [openai/whisper-base](https://huggingface.co/openai/whisper-base) | [openai/whisper-base.en](https://huggingface.co/openai/whisper-base.en) | [openai/whisper-small](https://huggingface.co/openai/whisper-small) | [openai/whisper-small.en](https://huggingface.co/openai/whisper-small.en) | [openai/whisper-medium](https://huggingface.co/openai/whisper-medium) | [openai/whisper-medium.en](https://huggingface.co/openai/whisper-medium.en) | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)

### Troubleshooting

#### Empty or rubbish output

Example output:
```
----------------
```

To resolve this ensure that audio data has a 16k Hz sampling rate. You can use the recorder.py provided to record or use FFmpeg to convert the audio to the required format. 
