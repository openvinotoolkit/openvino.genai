// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openvino/genai/c_wrapper/llm_pipeline_c.h"

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

    LLMPipelineHandle* pipe = CreateLLMPipeline(options.model, options.device);

    GenerationConfigHandle* config = CreateGenerationConfig();
    GenerationConfigSetMaxNewTokens(config, options.max_new_tokens);

    for (size_t i = 0; i < options.num_warmup; i++)
        LLMPipelineGenerate(pipe, options.prompt, output, MAX_OUTPUT_LENGTH, config);

    LLMPipelineGenerate(pipe, options.prompt, output, MAX_OUTPUT_LENGTH, config);

    DecodedResultsHandle* results = LLMPipelineGenerateDecodeResults(pipe, options.prompt, config);

    DecodeResultsGetString(results, output, MAX_OUTPUT_LENGTH);
    printf("%s\n", output);

    PerfMetricsHandle* metrics = NULL;
    DecodedeResultsGetPerfMetrics(results, &metrics);

    for (size_t i = 0; i < options.num_iter - 1; i++) {
        results = LLMPipelineGenerateDecodeResults(pipe, options.prompt, config);
        PerfMetricsHandle* _metrics = NULL;
        DecodedeResultsGetPerfMetrics(results, &_metrics);
        AddPerfMetricsInPlace(metrics, _metrics);
        DestroyPerfMetics(_metrics);
        DestroyDecodedResults(results);
    }

    printf("%.2f ms\n", PerfMetricsGetLoadTime(metrics));
    printf("Generate time: %.2f ± %.2f ms\n",
           PerfMetricsGetGenerateDuration(metrics).mean,
           PerfMetricsGetGenerateDuration(metrics).std);
    printf("Tokenization time: %.2f ± %.2f ms\n",
           PerfMetricsGetTokenizationDuration(metrics).mean,
           PerfMetricsGetTokenizationDuration(metrics).std);
    printf("Detokenization time: %.2f ± %.2f ms\n",
           PerfMetricsGetDetokenizationDuration(metrics).mean,
           PerfMetricsGetDetokenizationDuration(metrics).std);
    printf("TTFT: %.2f ± %.2f ms\n", PerfMetricsGetTtft(metrics).mean, PerfMetricsGetTtft(metrics).std);
    printf("TPOT: %.2f ± %.2f ms/token\n", PerfMetricsGetTpot(metrics).mean, PerfMetricsGetTpot(metrics).std);
    printf("Throughput: %.2f ± %.2f tokens/s\n",
           PerfMetricsGetThroughput(metrics).mean,
           PerfMetricsGetThroughput(metrics).std);

    // Release Resources
    DestroyLLMPipeline(pipe);
    DestroyGenerationConfig(config);
    DestroyPerfMetics(metrics);
    return EXIT_SUCCESS;
}
