// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openvino/genai/c/text2speech_pipeline.h"

static void write_u16(FILE* file, uint16_t value) {
    fputc(value & 0xff, file);
    fputc((value >> 8) & 0xff, file);
}

static void write_u32(FILE* file, uint32_t value) {
    write_u16(file, value & 0xffff);
    write_u16(file, value >> 16);
}

static int save_wav(const char* path, const float* samples, size_t count, uint32_t rate) {
    if (count > (UINT32_MAX - 36) / 2 || rate > UINT32_MAX / 2)
        return 0;
    FILE* file = fopen(path, "wb");
    if (!file)
        return 0;
    const uint32_t data_size = (uint32_t)count * 2;
    fwrite("RIFF", 1, 4, file);
    write_u32(file, 36 + data_size);
    fwrite("WAVEfmt ", 1, 8, file);
    write_u32(file, 16);
    write_u16(file, 1);  // PCM
    write_u16(file, 1);  // mono
    write_u32(file, rate);
    write_u32(file, rate * 2);
    write_u16(file, 2);
    write_u16(file, 16);
    fwrite("data", 1, 4, file);
    write_u32(file, data_size);
    for (size_t i = 0; i < count; ++i) {
        float sample = samples[i];
        if (sample > 1.0f)
            sample = 1.0f;
        if (sample < -1.0f)
            sample = -1.0f;
        write_u16(file, (uint16_t)(int16_t)(sample * 32767.0f));
    }
    int success = !ferror(file);
    if (fclose(file) != 0)
        success = 0;
    return success;
}

static ov_tensor_t* load_embedding(const char* path, ov_genai_text2speech_pipeline* pipeline) {
    ov_shape_t shape = {0};
    ov_tensor_t* embedding = NULL;
    FILE* file = NULL;
    if (ov_genai_text2speech_pipeline_get_speaker_embedding_shape(pipeline, &shape) != OK)
        return NULL;
    size_t count = 1;
    for (int64_t i = 0; i < shape.rank; ++i)
        count *= (size_t)shape.dims[i];
    if (ov_tensor_create(F32, shape, &embedding) != OK)
        goto done;
    file = fopen(path, "rb");
    if (!file)
        goto done;
    void* data = NULL;
    if (ov_tensor_data(embedding, &data) != OK || fread(data, sizeof(float), count, file) != count ||
        fgetc(file) != EOF) {
        goto done;
    }
    fclose(file);
    ov_shape_free(&shape);
    return embedding;
done:
    if (file)
        fclose(file);
    ov_tensor_free(embedding);
    ov_shape_free(&shape);
    return NULL;
}

int main(int argc, char** argv) {
    const int batch = argc > 1 && strcmp(argv[1], "--batch") == 0;
    if ((batch && argc != 8) || (!batch && (argc < 4 || argc > 6))) {
        fprintf(stderr, "Usage: %s MODEL_DIR TEXT SPEAKER_EMBEDDING.bin|- [OUTPUT.wav] [DEVICE]\n", argv[0]);
        fprintf(stderr, "       %s --batch MODEL_DIR TEXT1 TEXT2 SPEAKER_EMBEDDING.bin|- OUTPUT1.wav OUTPUT2.wav\n",
                argv[0]);
        return 1;
    }
    const char* model_path = argv[batch ? 2 : 1];
    const char* voice_path = argv[batch ? 5 : 3];
    const char* device = !batch && argc > 5 ? argv[5] : "CPU";
    ov_genai_text2speech_pipeline* pipeline = NULL;
    ov_genai_text2speech_decoded_results* results = NULL;
    ov_tensor_t* embedding = NULL;
    ov_tensor_t* speech = NULL;
    int success = 0;
    if (ov_genai_text2speech_pipeline_create(model_path, device, 0, &pipeline) != OK)
        goto done;
    if (!(voice_path[0] == '-' && voice_path[1] == '\0')) {
        embedding = load_embedding(voice_path, pipeline);
        if (!embedding)
            goto done;
    }
    if (batch) {
        const char* texts[] = {argv[3], argv[4]};
        if (ov_genai_text2speech_pipeline_generate_batch(pipeline, texts, 2, embedding, &results) != OK)
            goto done;
    } else if (ov_genai_text2speech_pipeline_generate(pipeline, argv[2], embedding, &results) != OK) {
        goto done;
    }
    size_t speech_count = 0;
    uint32_t rate = 0;
    const size_t expected_count = batch ? 2 : 1;
    if (ov_genai_text2speech_decoded_results_get_speeches_count(results, &speech_count) != OK ||
        speech_count != expected_count || ov_genai_text2speech_decoded_results_get_output_sample_rate(results, &rate) != OK)
        goto done;
    for (size_t i = 0; i < speech_count; ++i) {
        size_t sample_count = 0;
        void* data = NULL;
        const char* output_path = batch ? argv[6 + i] : (argc > 4 ? argv[4] : "output_audio.wav");
        if (ov_genai_text2speech_decoded_results_get_speech_at(results, i, &speech) != OK ||
            ov_tensor_get_size(speech, &sample_count) != OK || ov_tensor_data(speech, &data) != OK ||
            sample_count == 0 || !save_wav(output_path, (const float*)data, sample_count, rate))
            goto done;
        ov_tensor_free(speech);
        speech = NULL;
    }
    printf("Generated %zu speech waveform(s)\n", speech_count);
    success = 1;
done:
    ov_tensor_free(speech);
    ov_tensor_free(embedding);
    ov_genai_text2speech_decoded_results_free(results);
    ov_genai_text2speech_pipeline_free(pipeline);
    if (!success)
        fprintf(stderr, "Speech generation failed\n");
    return success ? 0 : 1;
}
