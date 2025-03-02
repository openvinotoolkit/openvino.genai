// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openvino/genai/c/llm_pipeline_c.h"

#define MAX_PROMPT_LENGTH 256
#define MAX_OUTPUT_LENGTH 1024

#define DEFAULT_PROMPT         "The Sky is blue because"
#define DEFAULT_NUM_WARMUP     1
#define DEFAULT_NUM_ITER       3
#define DEFAULT_MAX_NEW_TOKENS 20
#define DEFAULT_DEVICE         "CPU"

typedef struct {
    char* model;
    char* prompt;
    size_t num_warmup;
    size_t num_iter;
    size_t max_new_tokens;
    char* device;
} Options;

void print_usage() {
    printf("Usage: benchmark_vanilla_genai [OPTIONS]\n");
    printf("Options:\n");
    printf("  -m, --model            Path to model and tokenizers base directory\n");
    printf("  -p, --prompt           Prompt (default: \"%s\")\n", DEFAULT_PROMPT);
    printf("  -nw, --num_warmup      Number of warmup iterations (default: %d)\n", DEFAULT_NUM_WARMUP);
    printf("  -n, --num_iter         Number of iterations (default: %d)\n", DEFAULT_NUM_ITER);
    printf("  -mt, --max_new_tokens  Maximal number of new tokens (default: %d)\n", DEFAULT_MAX_NEW_TOKENS);
    printf("  -d, --device           Device (default: %s)\n", DEFAULT_DEVICE);
    printf("  -h, --help             Print usage\n");
}
int parse_arguments(int argc, char* argv[], Options* options) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (i + 1 < argc) {
                options->model = argv[++i];
            } else {
                printf("Error: --model requires an argument\n");
                return -1;
            }
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            if (i + 1 < argc) {
                options->prompt = argv[++i];
            } else {
                printf("Error: --prompt requires an argument\n");
                return -1;
            }
        } else if (strcmp(argv[i], "-nw") == 0 || strcmp(argv[i], "--num_warmup") == 0) {
            if (i + 1 < argc) {
                options->num_warmup = atoi(argv[++i]);
            } else {
                printf("Error: --num_warmup requires an argument\n");
                return -1;
            }
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--num_iter") == 0) {
            if (i + 1 < argc) {
                options->num_iter = atoi(argv[++i]);
            } else {
                printf("Error: --num_iter requires an argument\n");
                return -1;
            }
        } else if (strcmp(argv[i], "-mt") == 0 || strcmp(argv[i], "--max_new_tokens") == 0) {
            if (i + 1 < argc) {
                options->max_new_tokens = atoi(argv[++i]);
            } else {
                printf("Error: --max_new_tokens requires an argument\n");
                return -1;
            }
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--device") == 0) {
            if (i + 1 < argc) {
                options->device = argv[++i];
            } else {
                printf("Error: --device requires an argument\n");
                return -1;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 0;
        } else {
            printf("Error: Unknown option %s\n", argv[i]);
            return -1;
        }
    }
    return 1;
}

int main(int argc, char* argv[]) {
    Options options = {.model = NULL,
                       .prompt = DEFAULT_PROMPT,
                       .num_warmup = DEFAULT_NUM_WARMUP,
                       .num_iter = DEFAULT_NUM_ITER,
                       .max_new_tokens = DEFAULT_MAX_NEW_TOKENS,
                       .device = DEFAULT_DEVICE};

    int result = parse_arguments(argc, argv, &options);
    if (result == 0) {
        return EXIT_SUCCESS;
    } else if (result == -1) {
        return EXIT_FAILURE;
    }

    printf("Model: %s\n", options.model ? options.model : "Not specified");
    printf("Prompt: %s\n", options.prompt);
    printf("Num Warmup: %zu\n", options.num_warmup);
    printf("Num Iter: %zu\n", options.num_iter);
    printf("Max New Tokens: %zu\n", options.max_new_tokens);
    printf("Device: %s\n", options.device);

    char output[MAX_OUTPUT_LENGTH];

    ov_genai_llm_pipeline* pipe = ov_genai_llm_pipeline_create(options.model, options.device);

    ov_genai_generation_config* config = ov_genai_generation_config_create();
    ov_genai_generation_config_set_max_new_tokens(config, options.max_new_tokens);

    for (size_t i = 0; i < options.num_warmup; i++)
        ov_genai_llm_pipeline_generate(pipe, options.prompt, output, MAX_OUTPUT_LENGTH, config);

    ov_genai_llm_pipeline_generate(pipe, options.prompt, output, MAX_OUTPUT_LENGTH, config);

    ov_genai_decoded_results* results = ov_genai_llm_pipeline_generate_decode_results(pipe, options.prompt, config);

    ov_genai_decoded_results_get_string(results, output, MAX_OUTPUT_LENGTH);
    printf("%s\n", output);

    ov_genai_perf_metrics* metrics = NULL;
    ov_genai_decoded_results_get_perf_metrics(results, &metrics);

    for (size_t i = 0; i < options.num_iter - 1; i++) {
        results = ov_genai_llm_pipeline_generate_decode_results(pipe, options.prompt, config);
        ov_genai_perf_metrics* _metrics = NULL;
        ov_genai_decoded_results_get_perf_metrics(results, &_metrics);
        ov_genai_perf_metrics_add_in_place(metrics, _metrics);
        ov_genai_perf_metrics_free(_metrics);
        ov_genai_decoded_results_free(results);
    }

    printf("%.2f ms\n", ov_genai_perf_metrics_get_load_time(metrics));
    printf("Generate time: %.2f ± %.2f ms\n",
           ov_genai_perf_metrics_get_generate_duration(metrics).mean,
           ov_genai_perf_metrics_get_generate_duration(metrics).std);
    printf("Tokenization time: %.2f ± %.2f ms\n",
           ov_genai_perf_metrics_get_tokenization_duration(metrics).mean,
           ov_genai_perf_metrics_get_tokenization_duration(metrics).std);
    printf("Detokenization time: %.2f ± %.2f ms\n",
           ov_genai_perf_metrics_get_detokenization_duration(metrics).mean,
           ov_genai_perf_metrics_get_detokenization_duration(metrics).std);
    printf("TTFT: %.2f ± %.2f ms\n", ov_genai_perf_metrics_get_ttft(metrics).mean, ov_genai_perf_metrics_get_ttft(metrics).std);
    printf("TPOT: %.2f ± %.2f ms/token\n", ov_genai_perf_metrics_get_tpot(metrics).mean, ov_genai_perf_metrics_get_tpot(metrics).std);
    printf("Throughput: %.2f ± %.2f tokens/s\n",
           ov_genai_perf_metrics_get_throughput(metrics).mean,
           ov_genai_perf_metrics_get_throughput(metrics).std);

    // Release Resources
    ov_genai_llm_pipeline_free(pipe);
    ov_genai_generation_config_free(config);
    ov_genai_perf_metrics_free(metrics);
    return EXIT_SUCCESS;
}
