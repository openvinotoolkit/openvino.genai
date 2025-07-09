# Whisper Speech Recognition C Sample

This sample demonstrates how to use the OpenVINO GenAI Whisper Pipeline C API for automatic speech recognition.

## Features

- **C API Usage**: Shows how to use the WhisperPipeline C API for speech recognition
- **Configuration**: Demonstrates setting language, task, and other parameters
- **Timestamps**: Shows how to retrieve timestamp information for transcribed segments
- **Memory Management**: Proper allocation and deallocation of resources
- **Error Handling**: Comprehensive error checking using status codes

## Building

The sample is built as part of the OpenVINO GenAI build process:

```bash
mkdir build && cd build
cmake ..
make whisper_speech_recognition
```

## Usage

```bash
./whisper_speech_recognition <MODEL_DIR> [language] [task]
```

### Arguments

- `MODEL_DIR`: Path to the directory containing the Whisper model files
- `language`: Optional language code (e.g., "en", "fr", "de") - default: auto-detect
- `task`: Optional task ("transcribe" or "translate") - default: "transcribe"

### Examples

```bash
# Basic usage with auto-detection
./whisper_speech_recognition /path/to/whisper/model

# Specify language and task
./whisper_speech_recognition /path/to/whisper/model en transcribe

# Translation task
./whisper_speech_recognition /path/to/whisper/model fr translate
```

## Sample Output

```
Creating Whisper pipeline...
Creating generation config...
Setting task to: transcribe
Running speech recognition on sample audio...
Transcription: The quick brown fox jumps over the lazy dog.

Detailed Results (1 texts):
  [0] Score: 0.9534, Text: The quick brown fox jumps over the lazy dog.

Timestamp Information (3 chunks):
  [0] 0.00s - 1.50s: The quick brown fox
  [1] 1.50s - 2.80s: jumps over the lazy
  [2] 2.80s - 4.00s: dog.

Speech recognition completed successfully!
```

## API Functions Used

This sample demonstrates the following C API functions:

### Pipeline Management
- `ov_genai_whisper_pipeline_create()` - Create pipeline
- `ov_genai_whisper_pipeline_free()` - Free pipeline
- `ov_genai_whisper_pipeline_generate()` - Perform speech recognition

### Configuration
- `ov_genai_whisper_generation_config_create()` - Create config
- `ov_genai_whisper_generation_config_free()` - Free config
- `ov_genai_whisper_generation_config_set_language()` - Set language
- `ov_genai_whisper_generation_config_set_task()` - Set task
- `ov_genai_whisper_generation_config_set_return_timestamps()` - Enable timestamps

### Results Processing
- `ov_genai_whisper_decoded_results_get_string()` - Get full transcription
- `ov_genai_whisper_decoded_results_get_texts_count()` - Get number of texts
- `ov_genai_whisper_decoded_results_get_text_at()` - Get specific text
- `ov_genai_whisper_decoded_results_get_score_at()` - Get confidence score
- `ov_genai_whisper_decoded_results_has_chunks()` - Check for timestamps
- `ov_genai_whisper_decoded_results_get_chunks_count()` - Get number of chunks
- `ov_genai_whisper_decoded_results_get_chunk_at()` - Get specific chunk

### Chunk Processing
- `ov_genai_whisper_decoded_result_chunk_get_start_ts()` - Get start timestamp
- `ov_genai_whisper_decoded_result_chunk_get_end_ts()` - Get end timestamp
- `ov_genai_whisper_decoded_result_chunk_get_text()` - Get chunk text
- `ov_genai_whisper_decoded_result_chunk_free()` - Free chunk

## Notes

- The sample uses synthetic audio (sine wave) for demonstration purposes
- In a real application, you would load audio from a file or stream
- Audio must be normalized to the range [-1, 1] and have a 16kHz sample rate
- All allocated memory is properly freed using the appropriate free functions
- Error handling is implemented using status codes and the CHECK_STATUS macro