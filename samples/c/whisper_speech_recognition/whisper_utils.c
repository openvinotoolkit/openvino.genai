// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper_utils.h"
#include <string.h>
#include <math.h>
#include <errno.h>

void print_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("\nRequired:\n");
    printf("  -m, --model            Path to Whisper model directory\n");
    printf("\nOptional:\n");
    printf("  -i, --input            Path to audio file (WAV format). If not specified, uses synthetic audio\n");
    printf("  -d, --device           Device to run inference on (default: %s)\n", DEFAULT_DEVICE);
    printf("  -l, --language         Language code (e.g., 'en', 'fr', 'de'). Empty for auto-detect (default: auto-detect)\n");
    printf("  -t, --task             Task: 'transcribe' or 'translate' (default: %s)\n", DEFAULT_TASK);
    printf("  --initial_prompt       Initial prompt to guide transcription\n");
    printf("  --timestamps           Return timestamps for each segment\n");
    printf("  -h, --help             Print this help message\n");
    printf("\nSynthetic audio options (when no input file specified):\n");
    printf("  --duration             Duration of synthetic audio in seconds (default: %.1f)\n", DEFAULT_DURATION);
    printf("\nExamples:\n");
    printf("  # Transcribe an audio file\n");
    printf("  %s -m /path/to/whisper/model -i audio.wav\n", program_name);
    printf("\n  # Translate French audio to English\n");
    printf("  %s -m /path/to/whisper/model -i french_audio.wav -l fr -t translate\n", program_name);
    printf("\n  # Use synthetic audio\n");
    printf("  %s -m /path/to/whisper/model\n", program_name);
}

int parse_arguments(int argc, char* argv[], Options* options) {
    // Initialize with defaults
    options->model_path = NULL;
    options->audio_path = NULL;
    options->device = DEFAULT_DEVICE;
    options->language = DEFAULT_LANGUAGE;
    options->task = DEFAULT_TASK;
    options->initial_prompt = NULL;
    options->return_timestamps = false;
    options->use_synthetic_audio = true;
    options->sample_rate = DEFAULT_SAMPLE_RATE;
    options->duration = DEFAULT_DURATION;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (i + 1 < argc) {
                options->model_path = argv[++i];
            } else {
                fprintf(stderr, "Error: --model requires an argument\n");
                return -1;
            }
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
            if (i + 1 < argc) {
                options->audio_path = argv[++i];
                options->use_synthetic_audio = false;
            } else {
                fprintf(stderr, "Error: --input requires an argument\n");
                return -1;
            }
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--device") == 0) {
            if (i + 1 < argc) {
                options->device = argv[++i];
            } else {
                fprintf(stderr, "Error: --device requires an argument\n");
                return -1;
            }
        } else if (strcmp(argv[i], "-l") == 0 || strcmp(argv[i], "--language") == 0) {
            if (i + 1 < argc) {
                options->language = argv[++i];
            } else {
                fprintf(stderr, "Error: --language requires an argument\n");
                return -1;
            }
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--task") == 0) {
            if (i + 1 < argc) {
                options->task = argv[++i];
                if (strcmp(options->task, "transcribe") != 0 && strcmp(options->task, "translate") != 0) {
                    fprintf(stderr, "Error: --task must be 'transcribe' or 'translate'\n");
                    return -1;
                }
            } else {
                fprintf(stderr, "Error: --task requires an argument\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--initial_prompt") == 0) {
            if (i + 1 < argc) {
                options->initial_prompt = argv[++i];
            } else {
                fprintf(stderr, "Error: --initial_prompt requires an argument\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--timestamps") == 0) {
            options->return_timestamps = true;
        } else if (strcmp(argv[i], "--duration") == 0) {
            if (i + 1 < argc) {
                options->duration = (float)atof(argv[++i]);
            } else {
                fprintf(stderr, "Error: --duration requires an argument\n");
                return -1;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Error: Unknown option %s\n", argv[i]);
            fprintf(stderr, "Use -h or --help for usage information\n");
            return -1;
        }
    }

    // Validate required arguments
    if (options->model_path == NULL) {
        fprintf(stderr, "Error: Model path is required. Use -m or --model option\n");
        fprintf(stderr, "Use -h or --help for usage information\n");
        return -1;
    }

    return 1;
}

int load_wav_file(const char* filename, float** audio_data, size_t* audio_length, float* sample_rate) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open audio file '%s'. ", filename);
        if (errno == ENOENT) {
            fprintf(stderr, "File does not exist.\n");
        } else if (errno == EACCES) {
            fprintf(stderr, "Permission denied.\n");
        } else {
            fprintf(stderr, "Error code: %d\n", errno);
        }
        return -1;
    }

    WAVHeader header;
    if (fread(&header, sizeof(WAVHeader), 1, file) != 1) {
        fprintf(stderr, "Error: Cannot read WAV header\n");
        fclose(file);
        return -1;
    }

    // Basic WAV validation
    if (strncmp(header.chunk_id, "RIFF", 4) != 0 || strncmp(header.format, "WAVE", 4) != 0) {
        fprintf(stderr, "Error: Invalid WAV file format\n");
        fclose(file);
        return -1;
    }

    if (header.audio_format != 1) { // PCM
        fprintf(stderr, "Error: Only PCM WAV files are supported\n");
        fclose(file);
        return -1;
    }

    if (header.num_channels != 1) {
        fprintf(stderr, "Error: Only mono audio is supported (found %d channels)\n", header.num_channels);
        fclose(file);
        return -1;
    }

    *sample_rate = (float)header.sample_rate;
    size_t num_samples = header.subchunk2_size / (header.bits_per_sample / 8);
    *audio_length = num_samples;

    // Allocate memory for audio data
    *audio_data = (float*)malloc(num_samples * sizeof(float));
    if (!*audio_data) {
        fprintf(stderr, "Error: Cannot allocate memory for audio data\n");
        fclose(file);
        return -1;
    }

    // Read and convert audio data to float
    if (header.bits_per_sample == 16) {
        int16_t* temp_buffer = (int16_t*)malloc(num_samples * sizeof(int16_t));
        if (!temp_buffer) {
            fprintf(stderr, "Error: Cannot allocate temporary buffer\n");
            free(*audio_data);
            fclose(file);
            return -1;
        }

        if (fread(temp_buffer, sizeof(int16_t), num_samples, file) != num_samples) {
            fprintf(stderr, "Error: Cannot read audio data\n");
            free(temp_buffer);
            free(*audio_data);
            fclose(file);
            return -1;
        }

        // Convert 16-bit PCM to float [-1, 1]
        for (size_t i = 0; i < num_samples; i++) {
            (*audio_data)[i] = temp_buffer[i] / 32768.0f;
        }

        free(temp_buffer);
    } else if (header.bits_per_sample == 32) {
        if (fread(*audio_data, sizeof(float), num_samples, file) != num_samples) {
            fprintf(stderr, "Error: Cannot read audio data\n");
            free(*audio_data);
            fclose(file);
            return -1;
        }
    } else {
        fprintf(stderr, "Error: Unsupported bit depth: %d\n", header.bits_per_sample);
        free(*audio_data);
        fclose(file);
        return -1;
    }

    fclose(file);
    return 0;
}

void generate_synthetic_audio(float* audio, size_t length, float frequency, float sample_rate) {
    for (size_t i = 0; i < length; i++) {
        audio[i] = 0.5f * sinf(2.0f * M_PI * frequency * (float)i / sample_rate);
    }
}

float* resample_audio(const float* input, size_t input_length, float input_rate, float target_rate, size_t* output_length) {
    if (input_rate == target_rate) {
        *output_length = input_length;
        float* output = (float*)malloc(input_length * sizeof(float));
        if (output) {
            memcpy(output, input, input_length * sizeof(float));
        }
        return output;
    }

    float ratio = input_rate / target_rate;
    *output_length = (size_t)(input_length / ratio);
    float* output = (float*)malloc(*output_length * sizeof(float));
    
    if (!output) {
        return NULL;
    }

    for (size_t i = 0; i < *output_length; i++) {
        float src_idx = i * ratio;
        size_t idx0 = (size_t)src_idx;
        size_t idx1 = idx0 + 1;
        
        if (idx1 >= input_length) {
            output[i] = input[input_length - 1];
        } else {
            float frac = src_idx - idx0;
            output[i] = input[idx0] * (1.0f - frac) + input[idx1] * frac;
        }
    }

    return output;
}