// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openvino/genai/c/text2speech_pipeline.h"
#include "openvino/genai/c/generation_config.h"
#include "audio_utils.h"
#include "cli_utf8.h"

#define CHECK_STATUS(return_status)                                                                  \
    do {                                                                                             \
        const ov_status_e _status = (return_status);                                                 \
        if (_status != OK) {                                                                         \
            fprintf(stderr, "[ERROR] return status %d, line %d\n", (int)_status, __LINE__);       \
            goto done;                                                                               \
        }                                                                                            \
    } while (0)

static ov_tensor_t* load_speaker_embedding(const char* path,
                                           const ov_genai_text2speech_pipeline* pipeline) {
    ov_shape_t shape = {0};
    ov_tensor_t* embedding = NULL;
    FILE* file = NULL;
    size_t count = 1;
    void* data = NULL;

    if (ov_genai_text2speech_pipeline_get_speaker_embedding_shape(pipeline, &shape) != OK)
        return NULL;

    for (int64_t i = 0; i < shape.rank; ++i)
        count *= (size_t)shape.dims[i];

    if (ov_tensor_create(F32, shape, &embedding) != OK)
        goto fail;

    file = fopen(path, "rb");
    if (!file)
        goto fail;

    if (ov_tensor_data(embedding, &data) != OK || !data)
        goto fail;
    if (fread(data, sizeof(float), count, file) != count || fgetc(file) != EOF)
        goto fail;

    fclose(file);
    ov_shape_free(&shape);
    return embedding;

fail:
    if (file)
        fclose(file);
    ov_tensor_free(embedding);
    ov_shape_free(&shape);
    return NULL;
}

static int write_f32_bin(const char* path, const ov_tensor_t* tensor) {
    ov_element_type_e type;
    void* data = NULL;
    size_t count = 0;
    FILE* file = NULL;

    if (ov_tensor_get_element_type(tensor, &type) != OK || type != F32)
        return 0;
    if (ov_tensor_get_size(tensor, &count) != OK)
        return 0;
    if (ov_tensor_data((ov_tensor_t*)tensor, &data) != OK || !data)
        return 0;

    file = fopen(path, "wb");
    if (!file)
        return 0;
    if (fwrite(data, sizeof(float), count, file) != count) {
        fclose(file);
        return 0;
    }
    return fclose(file) == 0;
}

static int read_ref_codec_ids(const char* path, ov_tensor_t** tensor) {
    FILE* file = fopen(path, "rb");
    int64_t rank = 0;
    int64_t dims_local[8] = {0};
    ov_shape_t shape = {0};
    ov_tensor_t* out = NULL;
    void* data = NULL;
    size_t elem_count = 1;
    ov_status_e status = OK;

    if (!file)
        return 0;

    if (fread(&rank, sizeof(rank), 1, file) != 1)
        goto fail;
    if (rank <= 0 || rank > 8)
        goto fail;

    for (int64_t i = 0; i < rank; ++i) {
        int64_t dim = 0;
        if (fread(&dim, sizeof(dim), 1, file) != 1)
            goto fail;
        if (dim <= 0)
            goto fail;
        dims_local[i] = dim;
        elem_count *= (size_t)dim;
    }

    status = ov_shape_create((size_t)rank, dims_local, &shape);
    if (status != OK)
        goto fail;

    if (ov_tensor_create(I64, shape, &out) != OK)
        goto fail;
    if (ov_tensor_data(out, &data) != OK || !data)
        goto fail;

    if (fread(data, sizeof(int64_t), elem_count, file) != elem_count)
        goto fail;
    if (fgetc(file) != EOF)
        goto fail;

    fclose(file);
    ov_shape_free(&shape);
    *tensor = out;
    return 1;

fail:
    if (file)
        fclose(file);
    ov_tensor_free(out);
    ov_shape_free(&shape);
    return 0;
}

static int write_ref_codec_ids(const char* path, const ov_tensor_t* tensor) {
    FILE* file = NULL;
    ov_element_type_e type;
    ov_shape_t shape = {0};
    void* data = NULL;
    size_t elem_count = 0;
    int64_t rank = 0;

    if (ov_tensor_get_element_type(tensor, &type) != OK || type != I64)
        return 0;
    if (ov_tensor_get_shape((ov_tensor_t*)tensor, &shape) != OK)
        return 0;
    if (ov_tensor_get_size(tensor, &elem_count) != OK)
        goto fail;
    if (ov_tensor_data((ov_tensor_t*)tensor, &data) != OK || !data)
        goto fail;

    file = fopen(path, "wb");
    if (!file)
        goto fail;

    rank = shape.rank;
    if (fwrite(&rank, sizeof(rank), 1, file) != 1)
        goto fail;
    for (int64_t i = 0; i < shape.rank; ++i) {
        if (fwrite(&shape.dims[i], sizeof(shape.dims[i]), 1, file) != 1)
            goto fail;
    }
    if (fwrite(data, sizeof(int64_t), elem_count, file) != elem_count)
        goto fail;

    if (fclose(file) != 0) {
        file = NULL;
        goto fail;
    }
    ov_shape_free(&shape);
    return 1;

fail:
    if (file)
        fclose(file);
    ov_shape_free(&shape);
    return 0;
}

static int parse_positive_int64(const char* text, int64_t* value) {
    char* end = NULL;
    long long parsed = strtoll(text, &end, 10);
    if (end == text || *end != '\0' || parsed <= 0)
        return 0;
    *value = (int64_t)parsed;
    return 1;
}

static int parse_uint64(const char* text, uint64_t* value) {
    char* end = NULL;
    unsigned long long parsed = 0;
    if (text[0] == '-')
        return 0;
    parsed = strtoull(text, &end, 10);
    if (end == text || *end != '\0')
        return 0;
    *value = (uint64_t)parsed;
    return 1;
}

static int str_eq(const char* a, const char* b) {
    return strcmp(a, b) == 0;
}

static void print_usage(const char* program) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr,
            "  %s base <MODEL_DIR> \"<PROMPT>\" (--ref_audio_wav_path <PATH.wav> | --speaker_embedding_file_path <PATH.bin>)\\n"
            "      [--ref_text \"<TRANSCRIPT>\"] [--ref_codec_ids_file_path <PATH.bin>]\\n"
            "      [--save_speaker_embedding_file_path <PATH.bin>] [--save_ref_codec_ids_file_path <PATH.bin>]\\n"
                "      [--language <LANG>] [--device <DEVICE>] [--max_new_tokens <N>] [--seed <N>] [--output_wav_path <PATH.wav>]\n",
            program);
    fprintf(stderr,
            "  %s customvoice <MODEL_DIR> \"<PROMPT>\" --speaker <NAME>\\n"
            "      [--language <LANG>] [--instruct \"<STYLE>\"] [--device <DEVICE>]\\n"
                "      [--max_new_tokens <N>] [--seed <N>] [--output_wav_path <PATH.wav>]\n",
            program);
    fprintf(stderr,
            "  %s voice-design <MODEL_DIR> \"<PROMPT>\" --instruct \"<VOICE_DESCRIPTION>\"\\n"
                "      [--language <LANG>] [--device <DEVICE>] [--max_new_tokens <N>] [--seed <N>] [--output_wav_path <PATH.wav>]\n",
            program);
}

static int run_base(int argc, char** argv) {
    const char* model_dir = NULL;
    const char* prompt = NULL;
    const char* device = "CPU";
    const char* output_wav_path = "output_audio.wav";
    const char* language = NULL;
    const char* ref_text = NULL;
    const char* ref_audio_wav_path = NULL;
    const char* speaker_embedding_path = NULL;
    const char* ref_codec_ids_path = NULL;
    const char* save_speaker_embedding_path = NULL;
    const char* save_ref_codec_ids_path = NULL;
    int64_t max_new_tokens = 0;
    uint64_t seed = 0;
    int has_seed = 0;

    ov_status_e status = OK;
    ov_genai_text2speech_pipeline* pipeline = NULL;
    ov_genai_speech_generation_config* config = NULL;
    ov_genai_generation_config* generation_config = NULL;
    ov_genai_text2speech_decoded_results* results = NULL;
    ov_tensor_t* speaker_embedding = NULL;
    ov_tensor_t* ref_audio = NULL;
    ov_tensor_t* ref_codec_ids = NULL;
    ov_tensor_t* speech = NULL;
    ov_tensor_t* saved_speaker_embedding = NULL;
    ov_tensor_t* saved_ref_codec_ids = NULL;
    uint32_t sample_rate = 0;
    size_t speech_count = 0;
    size_t sample_count = 0;
    void* speech_data = NULL;
    int success = 0;

    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    model_dir = argv[2];
    prompt = argv[3];

    for (int i = 4; i < argc; ) {
        const char* option = argv[i++];
        if (i >= argc) {
            print_usage(argv[0]);
            return 1;
        }
        const char* value = argv[i++];

        if (str_eq(option, "--ref_audio_wav_path")) {
            ref_audio_wav_path = value;
        } else if (str_eq(option, "--speaker_embedding_file_path")) {
            speaker_embedding_path = value;
        } else if (str_eq(option, "--ref_text")) {
            ref_text = value;
        } else if (str_eq(option, "--ref_codec_ids_file_path")) {
            ref_codec_ids_path = value;
        } else if (str_eq(option, "--save_speaker_embedding_file_path")) {
            save_speaker_embedding_path = value;
        } else if (str_eq(option, "--save_ref_codec_ids_file_path")) {
            save_ref_codec_ids_path = value;
        } else if (str_eq(option, "--language")) {
            language = value;
        } else if (str_eq(option, "--device")) {
            device = value;
        } else if (str_eq(option, "--max_new_tokens")) {
            if (!parse_positive_int64(value, &max_new_tokens)) {
                print_usage(argv[0]);
                return 1;
            }
        } else if (str_eq(option, "--seed")) {
            if (!parse_uint64(value, &seed)) {
                print_usage(argv[0]);
                return 1;
            }
            has_seed = 1;
        } else if (str_eq(option, "--output_wav_path")) {
            output_wav_path = value;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!ref_audio_wav_path && !speaker_embedding_path) {
        fprintf(stderr,
                "Qwen3-TTS Base requires --ref_audio_wav_path <PATH.wav> or --speaker_embedding_file_path <PATH.bin>.\n");
        return 1;
    }

    if (ref_codec_ids_path && (!ref_text || ref_text[0] == '\0')) {
        fprintf(stderr,
                "--ref_text is required when --ref_codec_ids_file_path is provided (ICL mode).\n");
        return 1;
    }

    CHECK_STATUS(ov_genai_text2speech_pipeline_create(model_dir, device, 0, &pipeline));
    CHECK_STATUS(ov_genai_text2speech_pipeline_get_generation_config(pipeline, &config));

    if (language && language[0] != '\0')
        CHECK_STATUS(ov_genai_speech_generation_config_set_language(config, language));
    if (ref_text && ref_text[0] != '\0')
        CHECK_STATUS(ov_genai_speech_generation_config_set_ref_text(config, ref_text));
    if (max_new_tokens > 0) {
        CHECK_STATUS(ov_genai_speech_generation_config_get_generation_config(config, &generation_config));
        CHECK_STATUS(ov_genai_generation_config_set_max_new_tokens(generation_config, (size_t)max_new_tokens));
        ov_genai_generation_config_free(generation_config);
        generation_config = NULL;
    }
    if (has_seed) {
        CHECK_STATUS(ov_genai_speech_generation_config_get_generation_config(config, &generation_config));
        CHECK_STATUS(ov_genai_generation_config_set_rng_seed(generation_config, (size_t)seed));
        ov_genai_generation_config_free(generation_config);
        generation_config = NULL;
    }

    if (ref_audio_wav_path) {
        if (!load_wav_as_tensor(ref_audio_wav_path, 24000.0f, &ref_audio)) {
            fprintf(stderr, "Failed to load reference WAV '%s'. Expected mono PCM WAV at 24000 Hz.\n", ref_audio_wav_path);
            goto done;
        }
        CHECK_STATUS(ov_genai_speech_generation_config_set_ref_audio(config, ref_audio));
    }

    if (ref_codec_ids_path) {
        if (!read_ref_codec_ids(ref_codec_ids_path, &ref_codec_ids)) {
            fprintf(stderr, "Failed to load reference codec IDs '%s'.\n", ref_codec_ids_path);
            goto done;
        }
        CHECK_STATUS(ov_genai_speech_generation_config_set_ref_codec_ids(config, ref_codec_ids));
    }

    CHECK_STATUS(ov_genai_speech_generation_config_validate(config));

    if (speaker_embedding_path) {
        speaker_embedding = load_speaker_embedding(speaker_embedding_path, pipeline);
        if (!speaker_embedding) {
            fprintf(stderr, "Failed to load speaker embedding '%s'.\n", speaker_embedding_path);
            goto done;
        }
    }

    CHECK_STATUS(ov_genai_text2speech_pipeline_generate(pipeline, prompt, speaker_embedding, config, &results));
    CHECK_STATUS(ov_genai_text2speech_decoded_results_get_speeches_count(results, &speech_count));
    if (speech_count != 1) {
        fprintf(stderr, "Expected 1 waveform, got %zu.\n", speech_count);
        goto done;
    }

    CHECK_STATUS(ov_genai_text2speech_decoded_results_get_output_sample_rate(results, &sample_rate));
    CHECK_STATUS(ov_genai_text2speech_decoded_results_get_speech_at(results, 0, &speech));
    CHECK_STATUS(ov_tensor_get_size(speech, &sample_count));
    CHECK_STATUS(ov_tensor_data(speech, &speech_data));

    if (!save_wav_file(output_wav_path, (const float*)speech_data, sample_count, sample_rate)) {
        fprintf(stderr, "Failed to write output WAV '%s'.\n", output_wav_path);
        goto done;
    }

    printf("[Info] Text successfully converted to audio file \"%s\".\n", output_wav_path);

    if (save_speaker_embedding_path) {
        status = ov_genai_text2speech_decoded_results_get_speaker_embedding(results, &saved_speaker_embedding);
        if (status == NOT_FOUND) {
            fprintf(stderr, "No speaker embedding was produced to save. Provide --ref_audio_wav_path.\n");
            goto done;
        }
        CHECK_STATUS(status);
        if (!write_f32_bin(save_speaker_embedding_path, saved_speaker_embedding)) {
            fprintf(stderr, "Failed to write speaker embedding '%s'.\n", save_speaker_embedding_path);
            goto done;
        }
        printf("[Info] Saved speaker embedding to \"%s\".\n", save_speaker_embedding_path);
    }

    if (save_ref_codec_ids_path) {
        status = ov_genai_text2speech_decoded_results_get_ref_codec_ids(results, &saved_ref_codec_ids);
        if (status == NOT_FOUND) {
            fprintf(stderr,
                    "No reference codes were produced to save. ICL mode (--ref_text) with --ref_audio_wav_path is required.\n");
            goto done;
        }
        CHECK_STATUS(status);
        if (!write_ref_codec_ids(save_ref_codec_ids_path, saved_ref_codec_ids)) {
            fprintf(stderr, "Failed to write reference codec IDs '%s'.\n", save_ref_codec_ids_path);
            goto done;
        }
        printf("[Info] Saved reference codes to \"%s\".\n", save_ref_codec_ids_path);
    }

    success = 1;

done:
    ov_tensor_free(saved_ref_codec_ids);
    ov_tensor_free(saved_speaker_embedding);
    ov_tensor_free(speech);
    ov_tensor_free(ref_codec_ids);
    ov_tensor_free(ref_audio);
    ov_tensor_free(speaker_embedding);
    ov_genai_text2speech_decoded_results_free(results);
    ov_genai_generation_config_free(generation_config);
    ov_genai_speech_generation_config_free(config);
    ov_genai_text2speech_pipeline_free(pipeline);
    if (!success)
        fprintf(stderr, "Qwen3-TTS Base generation failed.\n");
    return success ? 0 : 1;
}

static int run_customvoice(int argc, char** argv) {
    const char* model_dir = NULL;
    const char* prompt = NULL;
    const char* device = "CPU";
    const char* output_wav_path = "output_audio.wav";
    const char* speaker = NULL;
    const char* language = NULL;
    const char* instruct = NULL;
    int64_t max_new_tokens = 0;
    uint64_t seed = 0;
    int has_seed = 0;

    ov_genai_text2speech_pipeline* pipeline = NULL;
    ov_genai_speech_generation_config* config = NULL;
    ov_genai_generation_config* generation_config = NULL;
    ov_genai_text2speech_decoded_results* results = NULL;
    ov_tensor_t* speech = NULL;
    uint32_t sample_rate = 0;
    size_t sample_count = 0;
    size_t speech_count = 0;
    void* speech_data = NULL;
    int success = 0;

    if (argc < 6) {
        print_usage(argv[0]);
        return 1;
    }

    model_dir = argv[2];
    prompt = argv[3];

    for (int i = 4; i < argc; ) {
        const char* option = argv[i++];
        if (i >= argc) {
            print_usage(argv[0]);
            return 1;
        }
        const char* value = argv[i++];

        if (str_eq(option, "--speaker")) {
            speaker = value;
        } else if (str_eq(option, "--language")) {
            language = value;
        } else if (str_eq(option, "--instruct")) {
            instruct = value;
        } else if (str_eq(option, "--device")) {
            device = value;
        } else if (str_eq(option, "--max_new_tokens")) {
            if (!parse_positive_int64(value, &max_new_tokens)) {
                print_usage(argv[0]);
                return 1;
            }
        } else if (str_eq(option, "--seed")) {
            if (!parse_uint64(value, &seed)) {
                print_usage(argv[0]);
                return 1;
            }
            has_seed = 1;
        } else if (str_eq(option, "--output_wav_path")) {
            output_wav_path = value;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!speaker || speaker[0] == '\0') {
        fprintf(stderr, "Qwen3-TTS CustomVoice requires --speaker <NAME>.\n");
        return 1;
    }

    CHECK_STATUS(ov_genai_text2speech_pipeline_create(model_dir, device, 0, &pipeline));
    CHECK_STATUS(ov_genai_text2speech_pipeline_get_generation_config(pipeline, &config));

    CHECK_STATUS(ov_genai_speech_generation_config_set_speaker(config, speaker));
    if (language && language[0] != '\0')
        CHECK_STATUS(ov_genai_speech_generation_config_set_language(config, language));
    if (instruct && instruct[0] != '\0')
        CHECK_STATUS(ov_genai_speech_generation_config_set_instruct(config, instruct));
    if (max_new_tokens > 0) {
        CHECK_STATUS(ov_genai_speech_generation_config_get_generation_config(config, &generation_config));
        CHECK_STATUS(ov_genai_generation_config_set_max_new_tokens(generation_config, (size_t)max_new_tokens));
        ov_genai_generation_config_free(generation_config);
        generation_config = NULL;
    }

    if (has_seed) {
        CHECK_STATUS(ov_genai_speech_generation_config_get_generation_config(config, &generation_config));
        CHECK_STATUS(ov_genai_generation_config_set_rng_seed(generation_config, (size_t)seed));
        ov_genai_generation_config_free(generation_config);
        generation_config = NULL;
    }

    CHECK_STATUS(ov_genai_speech_generation_config_validate(config));
    CHECK_STATUS(ov_genai_text2speech_pipeline_generate(pipeline, prompt, NULL, config, &results));

    CHECK_STATUS(ov_genai_text2speech_decoded_results_get_speeches_count(results, &speech_count));
    if (speech_count != 1)
        goto done;
    CHECK_STATUS(ov_genai_text2speech_decoded_results_get_output_sample_rate(results, &sample_rate));
    CHECK_STATUS(ov_genai_text2speech_decoded_results_get_speech_at(results, 0, &speech));
    CHECK_STATUS(ov_tensor_get_size(speech, &sample_count));
    CHECK_STATUS(ov_tensor_data(speech, &speech_data));

    if (!save_wav_file(output_wav_path, (const float*)speech_data, sample_count, sample_rate)) {
        fprintf(stderr, "Failed to write output WAV '%s'.\n", output_wav_path);
        goto done;
    }

    printf("[Info] Text successfully converted to audio file \"%s\".\n", output_wav_path);
    success = 1;

done:
    ov_tensor_free(speech);
    ov_genai_text2speech_decoded_results_free(results);
    ov_genai_generation_config_free(generation_config);
    ov_genai_speech_generation_config_free(config);
    ov_genai_text2speech_pipeline_free(pipeline);
    if (!success)
        fprintf(stderr, "Qwen3-TTS CustomVoice generation failed.\n");
    return success ? 0 : 1;
}

static int run_voice_design(int argc, char** argv) {
    const char* model_dir = NULL;
    const char* prompt = NULL;
    const char* device = "CPU";
    const char* output_wav_path = "output_audio.wav";
    const char* instruct = NULL;
    const char* language = NULL;
    int64_t max_new_tokens = 0;
    uint64_t seed = 0;
    int has_seed = 0;

    ov_genai_text2speech_pipeline* pipeline = NULL;
    ov_genai_speech_generation_config* config = NULL;
    ov_genai_generation_config* generation_config = NULL;
    ov_genai_text2speech_decoded_results* results = NULL;
    ov_tensor_t* speech = NULL;
    uint32_t sample_rate = 0;
    size_t sample_count = 0;
    size_t speech_count = 0;
    void* speech_data = NULL;
    int success = 0;

    if (argc < 6) {
        print_usage(argv[0]);
        return 1;
    }

    model_dir = argv[2];
    prompt = argv[3];

    for (int i = 4; i < argc; ) {
        const char* option = argv[i++];
        if (i >= argc) {
            print_usage(argv[0]);
            return 1;
        }
        const char* value = argv[i++];

        if (str_eq(option, "--instruct")) {
            instruct = value;
        } else if (str_eq(option, "--language")) {
            language = value;
        } else if (str_eq(option, "--device")) {
            device = value;
        } else if (str_eq(option, "--max_new_tokens")) {
            if (!parse_positive_int64(value, &max_new_tokens)) {
                print_usage(argv[0]);
                return 1;
            }
        } else if (str_eq(option, "--seed")) {
            if (!parse_uint64(value, &seed)) {
                print_usage(argv[0]);
                return 1;
            }
            has_seed = 1;
        } else if (str_eq(option, "--output_wav_path")) {
            output_wav_path = value;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!instruct || instruct[0] == '\0') {
        fprintf(stderr, "Qwen3-TTS VoiceDesign requires --instruct \"<VOICE_DESCRIPTION>\".\n");
        return 1;
    }

    CHECK_STATUS(ov_genai_text2speech_pipeline_create(model_dir, device, 0, &pipeline));
    CHECK_STATUS(ov_genai_text2speech_pipeline_get_generation_config(pipeline, &config));

    CHECK_STATUS(ov_genai_speech_generation_config_set_instruct(config, instruct));
    if (language && language[0] != '\0')
        CHECK_STATUS(ov_genai_speech_generation_config_set_language(config, language));
    if (max_new_tokens > 0) {
        CHECK_STATUS(ov_genai_speech_generation_config_get_generation_config(config, &generation_config));
        CHECK_STATUS(ov_genai_generation_config_set_max_new_tokens(generation_config, (size_t)max_new_tokens));
        ov_genai_generation_config_free(generation_config);
        generation_config = NULL;
    }
    if (has_seed) {
        CHECK_STATUS(ov_genai_speech_generation_config_get_generation_config(config, &generation_config));
        CHECK_STATUS(ov_genai_generation_config_set_rng_seed(generation_config, (size_t)seed));
        ov_genai_generation_config_free(generation_config);
        generation_config = NULL;
    }

    CHECK_STATUS(ov_genai_speech_generation_config_validate(config));
    CHECK_STATUS(ov_genai_text2speech_pipeline_generate(pipeline, prompt, NULL, config, &results));

    CHECK_STATUS(ov_genai_text2speech_decoded_results_get_speeches_count(results, &speech_count));
    if (speech_count != 1)
        goto done;
    CHECK_STATUS(ov_genai_text2speech_decoded_results_get_output_sample_rate(results, &sample_rate));
    CHECK_STATUS(ov_genai_text2speech_decoded_results_get_speech_at(results, 0, &speech));
    CHECK_STATUS(ov_tensor_get_size(speech, &sample_count));
    CHECK_STATUS(ov_tensor_data(speech, &speech_data));

    if (!save_wav_file(output_wav_path, (const float*)speech_data, sample_count, sample_rate)) {
        fprintf(stderr, "Failed to write output WAV '%s'.\n", output_wav_path);
        goto done;
    }

    printf("[Info] Text successfully converted to audio file \"%s\".\n", output_wav_path);
    success = 1;

done:
    ov_tensor_free(speech);
    ov_genai_text2speech_decoded_results_free(results);
    ov_genai_generation_config_free(generation_config);
    ov_genai_speech_generation_config_free(config);
    ov_genai_text2speech_pipeline_free(pipeline);
    if (!success)
        fprintf(stderr, "Qwen3-TTS VoiceDesign generation failed.\n");
    return success ? 0 : 1;
}

int main(int argc, char** argv) {
    int normalized_argc = 0;
    char** normalized_argv = NULL;
    int result = 1;

    if (!normalize_cli_argv(argc, argv, &normalized_argc, &normalized_argv)) {
        fprintf(stderr, "Failed to normalize command-line arguments to UTF-8.\n");
        return 1;
    }

    if (normalized_argc < 2) {
        print_usage(normalized_argv[0]);
        goto done;
    }

    if (str_eq(normalized_argv[1], "base")) {
        result = run_base(normalized_argc, normalized_argv);
        goto done;
    }
    if (str_eq(normalized_argv[1], "customvoice")) {
        result = run_customvoice(normalized_argc, normalized_argv);
        goto done;
    }
    if (str_eq(normalized_argv[1], "voice-design")) {
        result = run_voice_design(normalized_argc, normalized_argv);
        goto done;
    }

    fprintf(stderr,
            "Unknown Qwen3-TTS variant '%s'. Expected: base, customvoice, voice-design.\n",
            normalized_argv[1]);
    print_usage(normalized_argv[0]);

done:
    free_normalized_cli_argv(normalized_argc, normalized_argv);
    return result;
}
