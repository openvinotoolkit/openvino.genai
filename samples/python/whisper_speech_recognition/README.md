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

Download example audio file:
`https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav`

Or you can use the [`recorder.py`](recorder.py) script. The script records 5 seconds of audio from the microphone. 

To install `PyAudio` dependency follow the [installation instructions](https://pypi.org/project/PyAudio/).

To run the script:
```
python recorder.py
```

## Run the Whisper model

`whisper_speech_recognition whisper-base how_are_you_doing_today.wav`

Output:
```
 How are you doing today?
timestamps: [0, 2] text:  How are you doing today?
```

See [SUPPORTED_MODELS.md](../../../src/docs/SUPPORTED_MODELS.md#whisper-models) for the list of supported models.

### Troubleshooting

#### Empty or rubbish output

Example output:
```
----------------
```

To resolve this ensure that audio data has a 16k Hz sampling rate. You can use the recorder.py provided to record or use FFmpeg to convert the audio to the required format. 
