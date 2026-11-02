// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openvino/genai/c/text2speech_pipeline.h"

#define CHECK_STATUS(return_status)                                                              \
    if (return_status != OK) {                                                                   \
        fprintf(stderr, "[ERROR] return status %d, line %d\n", (int)(return_status), __LINE__);  \
        goto done;                                                                             \
    }

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

static void print_usage(const char* program) {
    fprintf(stderr,
            "Usage: %s MODEL_DIR TEXT [SPEAKER_EMBEDDING.bin|-] [OUTPUT.wav] "
            "[--speed <FLOAT>] [--language <CODE>] [--device <DEVICE>]\n",
            program);
    fprintf(stderr,
            "       %s MODEL_DIR TEXT1 TEXT2 SPEAKER_EMBEDDING.bin|- [OUTPUT1.wav [OUTPUT2.wav]] [flags]  (with "
            "--batch)\n",
            program);
}

int main(int argc, char** argv) {
    // Parse optional flags anywhere in the command line; the rest are positional arguments.
    // Positionals (single): MODEL_DIR TEXT [SPEAKER_EMBEDDING.bin|-] [OUTPUT.wav]
    // Positionals (batch): MODEL_DIR TEXT1 TEXT2 SPEAKER_EMBEDDING.bin|- [OUTPUT1.wav [OUTPUT2.wav]]
    int batch = 0;
    float speed = 0.0f;
    const char* language = NULL;
    const char* device = "CPU";
    const char* positional[6];
    size_t positional_count = 0;
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (strcmp(arg, "--batch") == 0) {
            batch = 1;
        } else if (strcmp(arg, "--speed") == 0) {
            if (++i >= argc) {
                print_usage(argv[0]);
                return 1;
            }
            char* end = NULL;
            speed = strtof(argv[i], &end);
            if (end == argv[i] || *end != '\0' || speed <= 0.0f) {
                fprintf(stderr, "Invalid --speed value: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(arg, "--language") == 0) {
            if (++i >= argc) {
                print_usage(argv[0]);
                return 1;
            }
            language = argv[i];
        } else if (strcmp(arg, "--device") == 0) {
            if (++i >= argc) {
                print_usage(argv[0]);
                return 1;
            }
            device = argv[i];
        } else if (arg[0] == '-' && arg[1] == '-') {
            print_usage(argv[0]);
            return 1;
        } else if (positional_count < sizeof(positional) / sizeof(positional[0])) {
            positional[positional_count++] = arg;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    const char* model_path = NULL;
    const char* text1 = NULL;
    const char* text2 = NULL;
    const char* voice_path = "-";
    const char* output_path1 = "output_audio.wav";
    const char* output_path2 = "output_audio2.wav";
    if (batch) {
        if (positional_count < 4 || positional_count > 6) {
            print_usage(argv[0]);
            return 1;
        }
        model_path = positional[0];
        text1 = positional[1];
        text2 = positional[2];
        voice_path = positional[3];
        if (positional_count >= 5)
            output_path1 = positional[4];
        if (positional_count >= 6)
            output_path2 = positional[5];
    } else {
        if (positional_count < 2 || positional_count > 4) {
            print_usage(argv[0]);
            return 1;
        }
        model_path = positional[0];
        text1 = positional[1];
        if (positional_count >= 3)
            voice_path = positional[2];
        if (positional_count >= 4)
            output_path1 = positional[3];
    }

    ov_genai_text2speech_pipeline* pipeline = NULL;
    ov_genai_text2speech_decoded_results* results = NULL;
    ov_genai_speech_generation_config* config = NULL;
    ov_genai_perf_metrics* metrics = NULL;
    ov_tensor_t* embedding = NULL;
    ov_tensor_t* speech = NULL;
    int success = 0;

    CHECK_STATUS(ov_genai_text2speech_pipeline_create(model_path, device, 0, &pipeline));

    // Apply optional speed and language settings to this generation call.
    if (speed != 0.0f || language != NULL) {
        CHECK_STATUS(ov_genai_speech_generation_config_create(&config));
        if (speed != 0.0f)
            CHECK_STATUS(ov_genai_speech_generation_config_set_speed(config, speed));
        if (language != NULL)
            CHECK_STATUS(ov_genai_speech_generation_config_set_language(config, language));
        CHECK_STATUS(ov_genai_speech_generation_config_validate(config));
        if (speed != 0.0f)
            printf("Applied speech speed: %.1f\n", speed);
        if (language != NULL)
            printf("Applied language: %s\n", language);
    }

    if (strcmp(voice_path, "-") != 0) {
        embedding = load_embedding(voice_path, pipeline);
        if (!embedding) {
            fprintf(stderr, "Failed to load speaker embedding: %s\n", voice_path);
            goto done;
        }
    }

    if (batch) {
        const char* texts[] = {text1, text2};
        CHECK_STATUS(ov_genai_text2speech_pipeline_generate_batch(pipeline, texts, 2, embedding, config, &results));
    } else {
        CHECK_STATUS(ov_genai_text2speech_pipeline_generate(pipeline, text1, embedding, config, &results));
    }

    size_t speech_count = 0;
    uint32_t rate = 0;
    const size_t expected_count = batch ? 2 : 1;
    size_t total_samples = 0;
    CHECK_STATUS(ov_genai_text2speech_decoded_results_get_speeches_count(results, &speech_count));
    if (speech_count != expected_count) {
        fprintf(stderr, "Expected %zu speech waveform(s), got %zu\n", expected_count, speech_count);
        goto done;
    }
    CHECK_STATUS(ov_genai_text2speech_decoded_results_get_output_sample_rate(results, &rate));
    for (size_t i = 0; i < speech_count; ++i) {
        size_t sample_count = 0;
        void* data = NULL;
        const char* output_path = batch ? (i == 0 ? output_path1 : output_path2) : output_path1;
        CHECK_STATUS(ov_genai_text2speech_decoded_results_get_speech_at(results, i, &speech));
        CHECK_STATUS(ov_tensor_get_size(speech, &sample_count));
        CHECK_STATUS(ov_tensor_data(speech, &data));
        if (sample_count == 0 || !save_wav(output_path, (const float*)data, sample_count, rate))
            goto done;
        total_samples += sample_count;
        ov_tensor_free(speech);
        speech = NULL;
    }

    size_t generated_samples = 0;
    CHECK_STATUS(ov_genai_text2speech_decoded_results_get_perf_metrics(results, &metrics));
    CHECK_STATUS(ov_genai_text2speech_decoded_results_get_num_generated_samples(results, &generated_samples));
    if (generated_samples != total_samples)
        goto done;
    printf("Metrics: %zu generated samples\n", generated_samples);
    printf("Generated %zu speech waveform(s)\n", speech_count);
    success = 1;

done:
    ov_tensor_free(speech);
    ov_tensor_free(embedding);
    ov_genai_text2speech_decoded_results_perf_metrics_free(metrics);
    ov_genai_speech_generation_config_free(config);
    ov_genai_text2speech_decoded_results_free(results);
    ov_genai_text2speech_pipeline_free(pipeline);
    if (!success)
        fprintf(stderr, "Speech generation failed\n");
    return success ? 0 : 1;
}
