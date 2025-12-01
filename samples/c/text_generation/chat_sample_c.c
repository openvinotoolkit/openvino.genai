// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openvino/genai/c/llm_pipeline.h"
#include "openvino/genai/c/chat_history.h"
#include "openvino/genai/c/json_container.h"

#define MAX_PROMPT_LENGTH 1024
#define MAX_JSON_LENGTH 4096

#define CHECK_STATUS(return_status)                                                      \
    if (return_status != OK) {                                                           \
        fprintf(stderr, "[ERROR] return status %d, line %d\n", return_status, __LINE__); \
        goto err;                                                                        \
    }

#define CHECK_CHAT_HISTORY_STATUS(return_status)                                                      \
    if (return_status != OV_GENAI_CHAT_HISTORY_OK) {                                                           \
        fprintf(stderr, "[ERROR] chat history status %d, line %d\n", return_status, __LINE__); \
        goto err;                                                                        \
    }

#define CHECK_JSON_CONTAINER_STATUS(return_status)                                                      \
    if (return_status != OV_GENAI_JSON_CONTAINER_OK) {                                                           \
        fprintf(stderr, "[ERROR] json container status %d, line %d\n", return_status, __LINE__); \
        goto err;                                                                        \
    }

static void json_escape_string(const char* input, char* output, size_t output_size) {
    size_t i = 0;
    size_t j = 0;
    while (input[i] != '\0' && j < output_size - 1) {
        switch (input[i]) {
            case '"':
                if (j < output_size - 2) {
                    output[j++] = '\\';
                    output[j++] = '"';
                }
                break;
            case '\\':
                if (j < output_size - 2) {
                    output[j++] = '\\';
                    output[j++] = '\\';
                }
                break;
            case '\n':
                if (j < output_size - 2) {
                    output[j++] = '\\';
                    output[j++] = 'n';
                }
                break;
            case '\r':
                if (j < output_size - 2) {
                    output[j++] = '\\';
                    output[j++] = 'r';
                }
                break;
            case '\t':
                if (j < output_size - 2) {
                    output[j++] = '\\';
                    output[j++] = 't';
                }
                break;
            default:
                output[j++] = input[i];
                break;
        }
        i++;
    }
    output[j] = '\0';
}

ov_genai_streaming_status_e print_callback(const char* str, void* args) {
    if (str) {
        // If args is not null, it needs to be cast to its actual type.
        fprintf(stdout, "%s", str);
        fflush(stdout);
        return OV_GENAI_STREAMING_STATUS_RUNNING;
    } else {
        printf("Callback executed with NULL message!\n");
        return OV_GENAI_STREAMING_STATUS_STOP;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <MODEL_DIR> [DEVICE]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* models_path = argv[1];
    const char* device = (argc == 3) ? argv[2] : "CPU";  // GPU, NPU can be used as well

    ov_genai_generation_config* config = NULL;
    ov_genai_llm_pipeline* pipeline = NULL;
    ov_genai_chat_history* chat_history = NULL;
    ov_genai_decoded_results* results = NULL;
    ov_genai_json_container* message_container = NULL;
    ov_genai_json_container* assistant_message_container = NULL;
    streamer_callback streamer;
    streamer.callback_func = print_callback;
    streamer.args = NULL;
    char prompt[MAX_PROMPT_LENGTH];
    char message_json[MAX_JSON_LENGTH];
    char output_buffer[MAX_JSON_LENGTH];
    size_t output_size = 0;
    char assistant_message_json[MAX_JSON_LENGTH];
    char escaped_prompt[MAX_PROMPT_LENGTH * 2];
    char escaped_output[MAX_JSON_LENGTH * 2];

    CHECK_STATUS(ov_genai_llm_pipeline_create(models_path, device, 0, &pipeline));
    CHECK_STATUS(ov_genai_generation_config_create(&config));
    CHECK_STATUS(ov_genai_generation_config_set_max_new_tokens(config, 100));

    CHECK_CHAT_HISTORY_STATUS(ov_genai_chat_history_create(&chat_history));

    printf("question:\n");
    while (fgets(prompt, MAX_PROMPT_LENGTH, stdin)) {
        // Remove newline character
        prompt[strcspn(prompt, "\n")] = 0;
        
        // Skip empty lines
        if (strlen(prompt) == 0) {
            continue;
        }

        json_escape_string(prompt, escaped_prompt, sizeof(escaped_prompt));
        
        snprintf(message_json, sizeof(message_json), 
                 "{\"role\": \"user\", \"content\": \"%s\"}", escaped_prompt);
        
        if (message_container) {
            ov_genai_json_container_free(message_container);
            message_container = NULL;
        }
        CHECK_JSON_CONTAINER_STATUS(ov_genai_json_container_create_from_json_string(
            message_json, &message_container));
        
        // Push message using JsonContainer
        CHECK_CHAT_HISTORY_STATUS(ov_genai_chat_history_push_back(chat_history, message_container));

        results = NULL;
        CHECK_STATUS(ov_genai_llm_pipeline_generate_with_history(pipeline,
                                                                  chat_history,
                                                                  config,
                                                                  &streamer,
                                                                  &results));

        if (results) {
            output_size = sizeof(output_buffer);
            CHECK_STATUS(ov_genai_decoded_results_get_string(results, output_buffer, &output_size));
            
            json_escape_string(output_buffer, escaped_output, sizeof(escaped_output));
            
            snprintf(assistant_message_json, sizeof(assistant_message_json),
                     "{\"role\": \"assistant\", \"content\": \"%s\"}", escaped_output);
            
            if (assistant_message_container) {
                ov_genai_json_container_free(assistant_message_container);
                assistant_message_container = NULL;
            }
            CHECK_JSON_CONTAINER_STATUS(ov_genai_json_container_create_from_json_string(
                assistant_message_json, &assistant_message_container));
            
            // Push message using JsonContainer
            CHECK_CHAT_HISTORY_STATUS(ov_genai_chat_history_push_back(chat_history, assistant_message_container));
            
            ov_genai_decoded_results_free(results);
            results = NULL;
        }

        printf("\n----------\nquestion:\n");
    }

err:
    if (results)
        ov_genai_decoded_results_free(results);
    if (message_container)
        ov_genai_json_container_free(message_container);
    if (assistant_message_container)
        ov_genai_json_container_free(assistant_message_container);
    if (chat_history)
        ov_genai_chat_history_free(chat_history);
    if (pipeline)
        ov_genai_llm_pipeline_free(pipeline);
    if (config)
        ov_genai_generation_config_free(config);

    return EXIT_SUCCESS;
}
