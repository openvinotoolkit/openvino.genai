// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper_utils.h"

#include <errno.h>
#include <string.h>


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

    if (header.audio_format != 1) {  // PCM
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


float* resample_audio(const float* input,
                      size_t input_length,
                      float input_rate,
                      float target_rate,
                      size_t* output_length) {
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
