// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "text2speech_sample_utils.h"

#include <stdlib.h>
#include <string.h>

// Minimal WAV header structure
// This removes struct padding to ensure the Wav Header matches the exact binary layout
// required by the WAV specification, ensuring the output file is valid.
#pragma pack(push, 1)
typedef struct {
    char chunk_id[4];
    uint32_t chunk_size;
    char format[4];
    char subchunk1_id[4];
    uint32_t subchunk1_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char subchunk2_id[4];
    uint32_t subchunk2_size;
} WavHeader;
#pragma pack(pop)

void save_to_wav(const float* waveform_ptr, size_t waveform_size, const char* file_path) {
    uint32_t sample_rate = 16000;
    uint16_t num_channels = 1;
    uint16_t bits_per_sample = 32;  // IEEE Float

    WavHeader header;
    memcpy(header.chunk_id, "RIFF", 4);
    header.chunk_size = 36 + waveform_size * sizeof(float);
    memcpy(header.format, "WAVE", 4);
    memcpy(header.subchunk1_id, "fmt ", 4);
    header.subchunk1_size = 16;
    header.audio_format = 3;  // IEEE Float
    header.num_channels = num_channels;
    header.sample_rate = sample_rate;
    header.byte_rate = sample_rate * num_channels * sizeof(float);
    header.block_align = num_channels * sizeof(float);
    header.bits_per_sample = bits_per_sample;
    memcpy(header.subchunk2_id, "data", 4);
    header.subchunk2_size = waveform_size * sizeof(float);

    FILE* file = fopen(file_path, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open file for writing: %s\n", file_path);
        return;
    }
    if (fwrite(&header, sizeof(WavHeader), 1, file) != 1) {
        fprintf(stderr, "Failed to write WAV header to file: %s\n", file_path);
        fclose(file);
        return;
    }
    if (fwrite(waveform_ptr, sizeof(float), waveform_size, file) != waveform_size) {
        fprintf(stderr, "Failed to write waveform data to file: %s\n", file_path);
        fclose(file);
        return;
    }
    fclose(file);
}

ov_tensor_t* read_speaker_embedding(const char* file_path) {
    FILE* file = fopen(file_path, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open speaker embedding file: %s\n", file_path);
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (file_size != 512 * sizeof(float)) {
        fprintf(stderr, "Speaker embedding file must be 512 floats (2048 bytes).\n");
        fclose(file);
        return NULL;
    }

    float* data = (float*)malloc(file_size);
    if (fread(data, 1, file_size, file) != (size_t)file_size) {
        fprintf(stderr, "Failed to read speaker embedding data.\n");
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);

    ov_tensor_t* tensor = NULL;
    ov_shape_t shape;
    int64_t dims[] = {1, 512};
    ov_shape_create(2, dims, &shape);

    // Create tensor from host ptr
    if (ov_tensor_create(F32, shape, &tensor) != 0) {
        fprintf(stderr, "Failed to create speaker embedding tensor.\n");
        ov_shape_free(&shape);
        free(data);
        return NULL;
    }
    ov_shape_free(&shape);
    void* tensor_data = NULL;
    ov_tensor_data(tensor, &tensor_data);
    memcpy(tensor_data, data, file_size);
    free(data);

    return tensor;
}
