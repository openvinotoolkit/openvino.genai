# Whisper Automatic Speech Recognition C Sample

## Table of Contents

1. [Download OpenVINO GenAI](#download-openvino-genai)
2. [Build Samples](#build-samples)
3. [Download and Convert the Model](#download-and-convert-the-model)
4. [Prepare Audio File](#prepare-audio-file)
5. [Sample Description](#sample-description)
6. [Troubleshooting](#troubleshooting)
7. [Support and Contribution](#support-and-contribution)

## Download OpenVINO GenAI

Download and extract [OpenVINO GenAI Archive](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?PACKAGE=OPENVINO_GENAI&VERSION=NIGHTLY&OP_SYSTEM=WINDOWS&DISTRIBUTION=ARCHIVE) Visit the OpenVINO Download Page.

## Build Samples

Set up the environment and build the samples Linux and macOS:

```sh
source <INSTALL_DIR>/setupvars.sh
./<INSTALL_DIR>/samples/c/build_samples.sh
```

Windows Command Prompt:

```sh
<INSTALL_DIR>\setupvars.bat
<INSTALL_DIR>\samples\c\build_samples_msvc.bat
```

Windows PowerShell:

```sh
.<INSTALL_DIR>\setupvars.ps1
.<INSTALL_DIR>\samples\c\build_samples.ps1
```

## Download and Convert the Model

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

Install [../../export-requirements.txt](../../export-requirements.txt) if model conversion is required.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
optimum-cli export openvino --trust-remote-code --model openai/whisper-tiny whisper-tiny
```

If a converted model in OpenVINO IR format is available in the [OpenVINO optimized models](https://huggingface.co/OpenVINO) collection on Hugging Face, you can download it directly via huggingface-cli.

For example:

```sh
pip install huggingface-hub
huggingface-cli download OpenVINO/whisper-tiny-int8-ov --local-dir whisper-tiny-int8-ov
```

## Prepare audio file

Prepare audio file in wav format with sampling rate 16k Hz.

You can download example audio file: https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav

## Sample Description

This example showcases inference of speech recognition Whisper Models using the OpenVINO GenAI C API. The sample features `ov_genai_whisper_pipeline` and uses audio files in WAV format as input.

### Run Command

```sh
./whisper_speech_recognition_c <MODEL_DIR> "<WAV_FILE_PATH>" [DEVICE]
```

### Parameters

- `MODEL_DIR`: Path to the converted Whisper model directory
- `WAV_FILE_PATH`: Path to the WAV audio file (use quotes if path contains spaces)
- `DEVICE`: Optional - device to run inference on (default: "CPU")

### Example Usage

```sh
./whisper_speech_recognition_c whisper-tiny how_are_you_doing_today.wav
```

### Expected Output

```text
 How are you doing today?
timestamps: [0.00, 2.00] text:  How are you doing today?
```

The sample will:

1. Load the WAV audio file and validate its format
2. Automatically resample to 16kHz if needed
3. Perform speech-to-text transcription
4. Output the full transcription
5. Display word-level timestamps for each text chunk

## Troubleshooting

### Empty or Incorrect Output

If you get empty or incorrect transcription results:

- Ensure your audio file is in WAV format
- Check that the audio contains clear speech

### Model Loading Errors

If the model fails to load:

- Verify the model path exists and contains valid Whisper model files
- Ensure the model was properly converted to OpenVINO IR format
- Check that the specified device (CPU, GPU, etc.) is available on your system

### Audio File Errors

The sample provides detailed error messages for common audio file issues:

- File not found
- Permission denied
- Invalid WAV format
- Unsupported audio encoding (only PCM is supported)
- Multi-channel audio (only mono is supported)


## Support and Contribution
- For troubleshooting, consult the [OpenVINO documentation](https://docs.openvino.ai).
- To report issues or contribute, visit the [GitHub repository](https://github.com/openvinotoolkit/openvino.genai).
