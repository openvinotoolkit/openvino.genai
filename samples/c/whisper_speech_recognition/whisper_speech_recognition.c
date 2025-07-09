// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <errno.h>

#include "openvino/genai/c/whisper_pipeline.h"

#define MAX_PATH_LENGTH 1024
#define CHECK_STATUS(return_status)                                                      \
    if (return_status != OK) {                                                           \
        const char* error_msg = "Unknown error";                                         \
        switch(return_status) {                                                          \
            case INVALID_C_PARAM: error_msg = "Invalid parameter"; break;                \
            case NOT_FOUND: error_msg = "Not found"; break;                             \
            case OUT_OF_BOUNDS: error_msg = "Out of bounds"; break;                     \
            case UNEXPECTED: error_msg = "Unexpected error"; break;                     \
            case NOT_IMPLEMENTED: error_msg = "Not implemented"; break;                  \
            case UNKNOWN_EXCEPTION: error_msg = "Unknown exception"; break;              \
        }                                                                                \
        fprintf(stderr, "[ERROR] %s (status code: %d) at line %d\n",                    \
                error_msg, return_status, __LINE__);                                     \
        goto err;                                                                        \
    }

// Default values
#define DEFAULT_DEVICE         "CPU"
#define DEFAULT_LANGUAGE       ""
#define DEFAULT_TASK           "transcribe"
#define DEFAULT_NUM_WARMUP     1
#define DEFAULT_NUM_ITER       1
#define DEFAULT_SAMPLE_RATE    16000.0f
#define DEFAULT_DURATION       2.0f

typedef struct {
    const char* model_path;
    const char* audio_path;
    const char* device;
    const char* language;
    const char* task;
    const char* initial_prompt;
    bool return_timestamps;
    size_t num_warmup;
    size_t num_iter;
    bool use_synthetic_audio;
    float sample_rate;
    float duration;
} Options;

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
    printf("  --nw, --num_warmup     Number of warmup iterations (default: %d)\n", DEFAULT_NUM_WARMUP);
    printf("  -n, --num_iter         Number of benchmark iterations (default: %d)\n", DEFAULT_NUM_ITER);
    printf("  -h, --help             Print this help message\n");
    printf("\nSynthetic audio options (when no input file specified):\n");
    printf("  --duration             Duration of synthetic audio in seconds (default: %.1f)\n", DEFAULT_DURATION);
    printf("\nExamples:\n");
    printf("  # Transcribe an audio file\n");
    printf("  %s -m /path/to/whisper/model -i audio.wav\n", program_name);
    printf("\n  # Translate French audio to English\n");
    printf("  %s -m /path/to/whisper/model -i french_audio.wav -l fr -t translate\n", program_name);
    printf("\n  # Benchmark with synthetic audio\n");
    printf("  %s -m /path/to/whisper/model --num_warmup 3 --num_iter 10\n", program_name);
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
    options->num_warmup = DEFAULT_NUM_WARMUP;
    options->num_iter = DEFAULT_NUM_ITER;
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
        } else if (strcmp(argv[i], "--nw") == 0 || strcmp(argv[i], "--num_warmup") == 0) {
            if (i + 1 < argc) {
                options->num_warmup = (size_t)atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: --num_warmup requires an argument\n");
                return -1;
            }
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--num_iter") == 0) {
            if (i + 1 < argc) {
                options->num_iter = (size_t)atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: --num_iter requires an argument\n");
                return -1;
            }
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

// Simple WAV file header structure
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
} WAVHeader;

// Load audio from WAV file
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

// Generate synthetic audio (sine wave)
void generate_synthetic_audio(float* audio, size_t length, float frequency, float sample_rate) {
    for (size_t i = 0; i < length; i++) {
        audio[i] = 0.5f * sinf(2.0f * M_PI * frequency * (float)i / sample_rate);
    }
}

// Resample audio to 16kHz if needed (simple linear interpolation)
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

void print_configuration(const Options* options) {
    printf("\nConfiguration:\n");
    printf("  Model path: %s\n", options->model_path);
    if (options->audio_path) {
        printf("  Audio input: %s\n", options->audio_path);
    } else {
        printf("  Audio input: Synthetic (%.1fs @ %.0fHz)\n", options->duration, options->sample_rate);
    }
    printf("  Device: %s\n", options->device);
    printf("  Language: %s\n", strlen(options->language) > 0 ? options->language : "auto-detect");
    printf("  Task: %s\n", options->task);
    if (options->initial_prompt) {
        printf("  Initial prompt: %s\n", options->initial_prompt);
    }
    printf("  Return timestamps: %s\n", options->return_timestamps ? "yes" : "no");
    if (options->num_iter > 1) {
        printf("  Warmup iterations: %zu\n", options->num_warmup);
        printf("  Benchmark iterations: %zu\n", options->num_iter);
    }
    printf("\n");
}

int main(int argc, char* argv[]) {
    Options options;
    int result = parse_arguments(argc, argv, &options);
    if (result == 0) {
        return EXIT_SUCCESS;  // Help was printed
    } else if (result == -1) {
        return EXIT_FAILURE;  // Error in arguments
    }

    // Print configuration
    print_configuration(&options);

    // Initialize variables
    ov_genai_whisper_pipeline* pipeline = NULL;
    ov_genai_whisper_generation_config* config = NULL;
    ov_genai_whisper_decoded_results* results = NULL;
    ov_genai_perf_metrics* metrics = NULL;
    ov_genai_perf_metrics* cumulative_metrics = NULL;
    float* audio_data = NULL;
    float* resampled_audio = NULL;
    size_t audio_length = 0;
    char* output = NULL;
    size_t output_size = 0;
    
    // Load or generate audio
    if (options.audio_path) {
        float file_sample_rate;
        printf("Loading audio from %s...\n", options.audio_path);
        if (load_wav_file(options.audio_path, &audio_data, &audio_length, &file_sample_rate) != 0) {
            goto err;
        }
        
        // Resample to 16kHz if needed
        if (file_sample_rate != 16000.0f) {
            printf("Resampling from %.0fHz to 16000Hz...\n", file_sample_rate);
            size_t resampled_length;
            resampled_audio = resample_audio(audio_data, audio_length, file_sample_rate, 16000.0f, &resampled_length);
            if (!resampled_audio) {
                fprintf(stderr, "Error: Failed to resample audio\n");
                goto err;
            }
            free(audio_data);
            audio_data = resampled_audio;
            audio_length = resampled_length;
            resampled_audio = NULL;
        }
        
        printf("Loaded %.2f seconds of audio\n", audio_length / 16000.0f);
    } else {
        // Generate synthetic audio
        audio_length = (size_t)(options.sample_rate * options.duration);
        audio_data = (float*)malloc(audio_length * sizeof(float));
        if (!audio_data) {
            fprintf(stderr, "Error: Failed to allocate memory for audio data\n");
            goto err;
        }
        printf("Generating %.1f seconds of synthetic audio...\n", options.duration);
        generate_synthetic_audio(audio_data, audio_length, 440.0f, options.sample_rate);
    }
    
    // Validate model directory exists
    FILE* test_file = fopen(options.model_path, "r");
    if (test_file) {
        fclose(test_file);
        fprintf(stderr, "Error: Model path appears to be a file, not a directory: %s\n", options.model_path);
        goto err;
    }
    
    // Create pipeline
    printf("Creating Whisper pipeline...\n");
    ov_status_e status = ov_genai_whisper_pipeline_create(options.model_path, options.device, 0, &pipeline);
    if (status != OK) {
        if (status == UNKNOWN_EXCEPTION) {
            fprintf(stderr, "Error: Failed to create Whisper pipeline. Please check:\n");
            fprintf(stderr, "  - Model path exists and contains valid Whisper model files\n");
            fprintf(stderr, "  - Device '%s' is available and supported\n", options.device);
            fprintf(stderr, "  - Model is compatible with OpenVINO GenAI\n");
        }
        CHECK_STATUS(status);
    }
    
    // Create and configure generation config
    CHECK_STATUS(ov_genai_whisper_generation_config_create(&config));
    
    if (strlen(options.language) > 0) {
        CHECK_STATUS(ov_genai_whisper_generation_config_set_language(config, options.language));
    }
    
    CHECK_STATUS(ov_genai_whisper_generation_config_set_task(config, options.task));
    CHECK_STATUS(ov_genai_whisper_generation_config_set_return_timestamps(config, options.return_timestamps));
    
    if (options.initial_prompt) {
        CHECK_STATUS(ov_genai_whisper_generation_config_set_initial_prompt(config, options.initial_prompt));
    }
    
    // Warmup runs
    if (options.num_warmup > 0) {
        printf("Running %zu warmup iteration(s)...\n", options.num_warmup);
        for (size_t i = 0; i < options.num_warmup; i++) {
            if (results) {
                ov_genai_whisper_decoded_results_free(results);
                results = NULL;
            }
            CHECK_STATUS(ov_genai_whisper_pipeline_generate(pipeline, audio_data, audio_length, config, &results));
        }
    }
    
    // Benchmark runs
    printf("Running %zu iteration(s)...\n", options.num_iter);
    clock_t start_time = clock();
    
    for (size_t iter = 0; iter < options.num_iter; iter++) {
        if (results) {
            ov_genai_whisper_decoded_results_free(results);
            results = NULL;
        }
        
        CHECK_STATUS(ov_genai_whisper_pipeline_generate(pipeline, audio_data, audio_length, config, &results));
        
        // Get performance metrics
        if (iter == 0) {
            CHECK_STATUS(ov_genai_whisper_decoded_results_get_perf_metrics(results, &cumulative_metrics));
        } else {
            CHECK_STATUS(ov_genai_whisper_decoded_results_get_perf_metrics(results, &metrics));
            CHECK_STATUS(ov_genai_perf_metrics_add_in_place(cumulative_metrics, metrics));
            ov_genai_decoded_results_perf_metrics_free(metrics);
            metrics = NULL;
        }
    }
    
    clock_t end_time = clock();
    double total_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC * 1000.0;
    
    // Get and print results from the last iteration
    printf("\n=== Transcription Results ===\n");
    
    // Get the transcription text
    CHECK_STATUS(ov_genai_whisper_decoded_results_get_string(results, NULL, &output_size));
    output = (char*)malloc(output_size);
    if (!output) {
        fprintf(stderr, "Error: Failed to allocate memory for output\n");
        goto err;
    }
    
    CHECK_STATUS(ov_genai_whisper_decoded_results_get_string(results, output, &output_size));
    printf("\nFull transcription:\n%s\n", output);
    
    // Display individual text results with scores
    size_t texts_count = 0;
    CHECK_STATUS(ov_genai_whisper_decoded_results_get_texts_count(results, &texts_count));
    
    if (texts_count > 1) {
        printf("\nDetailed results (%zu texts):\n", texts_count);
        for (size_t i = 0; i < texts_count; i++) {
            size_t text_size = 0;
            CHECK_STATUS(ov_genai_whisper_decoded_results_get_text_at(results, i, NULL, &text_size));
            
            char* text = (char*)malloc(text_size);
            if (!text) {
                fprintf(stderr, "Warning: Failed to allocate memory for text %zu\n", i);
                continue;
            }
            
            CHECK_STATUS(ov_genai_whisper_decoded_results_get_text_at(results, i, text, &text_size));
            
            float score = 0.0f;
            CHECK_STATUS(ov_genai_whisper_decoded_results_get_score_at(results, i, &score));
            
            printf("  [%zu] Score: %.4f, Text: %s\n", i, score, text);
            free(text);
        }
    }
    
    // Display timestamps if available
    bool has_chunks = false;
    CHECK_STATUS(ov_genai_whisper_decoded_results_has_chunks(results, &has_chunks));
    
    if (has_chunks) {
        size_t chunks_count = 0;
        CHECK_STATUS(ov_genai_whisper_decoded_results_get_chunks_count(results, &chunks_count));
        
        printf("\nTimestamp information (%zu chunks):\n", chunks_count);
        for (size_t i = 0; i < chunks_count; i++) {
            ov_genai_whisper_decoded_result_chunk* chunk = NULL;
            CHECK_STATUS(ov_genai_whisper_decoded_results_get_chunk_at(results, i, &chunk));
            
            float start_ts = 0.0f, end_ts = 0.0f;
            CHECK_STATUS(ov_genai_whisper_decoded_result_chunk_get_start_ts(chunk, &start_ts));
            CHECK_STATUS(ov_genai_whisper_decoded_result_chunk_get_end_ts(chunk, &end_ts));
            
            size_t chunk_text_size = 0;
            CHECK_STATUS(ov_genai_whisper_decoded_result_chunk_get_text(chunk, NULL, &chunk_text_size));
            
            char* chunk_text = (char*)malloc(chunk_text_size);
            if (!chunk_text) {
                fprintf(stderr, "Warning: Failed to allocate memory for chunk text %zu\n", i);
                ov_genai_whisper_decoded_result_chunk_free(chunk);
                continue;
            }
            
            CHECK_STATUS(ov_genai_whisper_decoded_result_chunk_get_text(chunk, chunk_text, &chunk_text_size));
            
            printf("  [%zu] %.2fs - %.2fs: %s\n", i, start_ts, end_ts, chunk_text);
            
            free(chunk_text);
            ov_genai_whisper_decoded_result_chunk_free(chunk);
        }
    }
    
    // Print performance metrics if multiple iterations
    if (options.num_iter > 1 && cumulative_metrics) {
        printf("\n=== Performance Metrics ===\n");
        printf("Average inference time per iteration: %.2f ms\n", total_time / options.num_iter);
        
        float mean = 0.0f, std = 0.0f;
        
        CHECK_STATUS(ov_genai_perf_metrics_get_load_time(cumulative_metrics, &mean));
        printf("Model load time: %.2f ms\n", mean);
        
        CHECK_STATUS(ov_genai_perf_metrics_get_generate_duration(cumulative_metrics, &mean, &std));
        printf("Generate time: %.2f ± %.2f ms\n", mean, std);
        
        CHECK_STATUS(ov_genai_perf_metrics_get_tokenization_duration(cumulative_metrics, &mean, &std));
        printf("Tokenization time: %.2f ± %.2f ms\n", mean, std);
        
        CHECK_STATUS(ov_genai_perf_metrics_get_detokenization_duration(cumulative_metrics, &mean, &std));
        printf("Detokenization time: %.2f ± %.2f ms\n", mean, std);
        
        CHECK_STATUS(ov_genai_perf_metrics_get_ttft(cumulative_metrics, &mean, &std));
        printf("Time to first token (TTFT): %.2f ± %.2f ms\n", mean, std);
        
        CHECK_STATUS(ov_genai_perf_metrics_get_tpot(cumulative_metrics, &mean, &std));
        printf("Time per output token (TPOT): %.2f ± %.2f ms/token\n", mean, std);
        
        CHECK_STATUS(ov_genai_perf_metrics_get_throughput(cumulative_metrics, &mean, &std));
        printf("Throughput: %.2f ± %.2f tokens/s\n", mean, std);
        
        // Audio processing speed
        float audio_duration = audio_length / 16000.0f;  // Whisper uses 16kHz
        float rtf = (total_time / 1000.0f) / (audio_duration * options.num_iter);
        printf("\nAudio duration: %.2f seconds\n", audio_duration);
        printf("Real-time factor (RTF): %.3fx (lower is better)\n", rtf);
    }
    
    printf("\nSpeech recognition completed successfully!\n");

err:
    // Cleanup
    if (pipeline)
        ov_genai_whisper_pipeline_free(pipeline);
    if (config)
        ov_genai_whisper_generation_config_free(config);
    if (results)
        ov_genai_whisper_decoded_results_free(results);
    if (metrics)
        ov_genai_decoded_results_perf_metrics_free(metrics);
    if (cumulative_metrics)
        ov_genai_decoded_results_perf_metrics_free(cumulative_metrics);
    if (output)
        free(output);
    if (audio_data)
        free(audio_data);
    if (resampled_audio)
        free(resampled_audio);
    
    return EXIT_SUCCESS;
}