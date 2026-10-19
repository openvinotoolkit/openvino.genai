// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_utils.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>

int load_wav_file(const char* filename, float** audio_data, size_t* audio_length, float* sample_rate) {
    if (!filename || !audio_data || !audio_length || !sample_rate)
        return -1;

    *audio_data = NULL;
    *audio_length = 0;
    *sample_rate = 0.0f;

    drwav wav;
    if (!drwav_init_file(&wav, filename, NULL)) {
        fprintf(stderr, "Error: Cannot open WAV file '%s'. ", filename);
        if (errno == ENOENT) {
            fprintf(stderr, "File does not exist.\n");
        } else if (errno == EACCES) {
            fprintf(stderr, "Permission denied.\n");
        } else {
            fprintf(stderr, "Error code: %d\n", errno);
        }
        return -1;
    }

    if (wav.channels != 1) {
        fprintf(stderr, "Error: Only mono audio is supported (found %u channels)\n", wav.channels);
        drwav_uninit(&wav);
        return -1;
    }

    *sample_rate = (float)wav.sampleRate;
    if (wav.totalPCMFrameCount > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "Error: WAV frame count exceeds the supported allocation size\n");
        drwav_uninit(&wav);
        return -1;
    }
    *audio_length = (size_t)wav.totalPCMFrameCount;
    *audio_data = (float*)malloc(*audio_length * sizeof(float));
    if (!*audio_data) {
        fprintf(stderr, "Error: Cannot allocate memory for audio data\n");
        drwav_uninit(&wav);
        return -1;
    }

    {
        const drwav_uint64 expected_frames = wav.totalPCMFrameCount;
        const drwav_uint64 frames_read = drwav_read_pcm_frames_f32(&wav, expected_frames, *audio_data);
        drwav_uninit(&wav);
        if (frames_read != expected_frames) {
            fprintf(stderr, "Error: Cannot read audio data\n");
            free(*audio_data);
            *audio_data = NULL;
            *audio_length = 0;
            return -1;
        }
    }

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

    {
        const float ratio = input_rate / target_rate;
        *output_length = (size_t)(input_length / ratio);
    }
    float* output = (float*)malloc(*output_length * sizeof(float));

    if (!output) {
        return NULL;
    }

    {
        const float ratio = input_rate / target_rate;
        for (size_t i = 0; i < *output_length; i++) {
            const float src_idx = i * ratio;
            const size_t idx0 = (size_t)src_idx;
            const size_t idx1 = idx0 + 1;

            if (idx1 >= input_length) {
                output[i] = input[input_length - 1];
            } else {
                const float frac = src_idx - idx0;
                output[i] = input[idx0] * (1.0f - frac) + input[idx1] * frac;
            }
        }
    }

    return output;
}

int load_wav_as_tensor(const char* filename, float expected_sample_rate, ov_tensor_t** tensor) {
    float* audio_data = NULL;
    size_t audio_length = 0;
    float file_sample_rate = 0.0f;
    ov_shape_t shape = {0};
    ov_tensor_t* out = NULL;
    void* data = NULL;

    if (!tensor)
        return 0;

    if (load_wav_file(filename, &audio_data, &audio_length, &file_sample_rate) != 0)
        return 0;

    if (file_sample_rate != expected_sample_rate) {
        free(audio_data);
        return 0;
    }

    const int64_t dims[] = {(int64_t)audio_length};
    if (ov_shape_create(1, dims, &shape) != OK) {
        free(audio_data);
        return 0;
    }

    if (ov_tensor_create(F32, shape, &out) != OK) {
        ov_shape_free(&shape);
        free(audio_data);
        return 0;
    }

    if (ov_tensor_data(out, &data) != OK || !data) {
        ov_shape_free(&shape);
        ov_tensor_free(out);
        free(audio_data);
        return 0;
    }

    memcpy(data, audio_data, audio_length * sizeof(float));

    ov_shape_free(&shape);
    free(audio_data);
    *tensor = out;
    return 1;
}

int save_wav_file(const char* path, const float* samples, size_t count, uint32_t sample_rate) {
    drwav wav;
    drwav_data_format format;
    drwav_uint64 frames_written = 0;

    if (!path || !samples || count == 0 || sample_rate == 0)
        return 0;

    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = 1;
    format.sampleRate = sample_rate;
    format.bitsPerSample = 32;

    if (!drwav_init_file_write(&wav, path, &format, NULL)) {
        return 0;
    }

    frames_written = drwav_write_pcm_frames(&wav, (drwav_uint64)count, samples);
    if (frames_written != (drwav_uint64)count) {
        drwav_uninit(&wav);
        return 0;
    }

    drwav_uninit(&wav);
    return 1;
}
