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

// Worst-case escape: each char → \uXXXX (6 bytes) + null terminator
#define MAX_ESCAPED_PROMPT_LENGTH (MAX_PROMPT_LENGTH * 6 + 1)

// JSON template: {"role": "user", "content": ""} ≈ 32 bytes
#define MAX_MESSAGE_JSON_LENGTH (MAX_ESCAPED_PROMPT_LENGTH + 32 + 1)

// Worst-case escaped output: MAX_JSON_LENGTH * 6 + null terminator
#define MAX_ESCAPED_OUTPUT_LENGTH ((MAX_JSON_LENGTH - 1) * 6 + 1)

// JSON template: {"role": "assistant", "content": ""} ≈ 35 bytes
#define MAX_ASSISTANT_MESSAGE_JSON_LENGTH (MAX_ESCAPED_OUTPUT_LENGTH + 35 + 1)

#define CHECK_STATUS(return_status)                                                      \
    if (return_status != OK) {                                                           \
        fprintf(stderr, "[ERROR] return status %d, line %d\n", return_status, __LINE__); \
        goto err;                                                                        \
    }

#define CHECK_CHAT_HISTORY_STATUS(return_status)                                                      \
    if (return_status != OV_GENAI_CHAT_HISTORY_OK) {                                                  \
        fprintf(stderr, "[ERROR] chat history status %d, line %d\n", return_status, __LINE__);         \
        goto err;                                                                                      \
    }

#define CHECK_JSON_CONTAINER_STATUS(return_status)                                                     \
    if (return_status != OV_GENAI_JSON_CONTAINER_OK) {                                                 \
        fprintf(stderr, "[ERROR] json container status %d, line %d\n", return_status, __LINE__);        \
        goto err;                                                                                      \
    }

// Returns 0 on success, -1 if buffer is too small
static int json_escape_string(const char* input, char* output, size_t output_size) {
    if (!input || !output || output_size == 0) {
        return -1;
    }
    size_t i = 0;
    size_t j = 0;
    while (input[i] != '\0' && j < output_size - 1) {
        unsigned char c = (unsigned char)input[i];
        switch (c) {
            case '"':
                if (j >= output_size - 3) {
                    return -1;  
                }
                output[j++] = '\\';
                output[j++] = '"';
                break;
            case '\\':
                if (j >= output_size - 3) {
                    return -1;  
                }
                output[j++] = '\\';
                output[j++] = '\\';
                break;
            case '\b':
                if (j >= output_size - 3) {
                    return -1; 
                }
                output[j++] = '\\';
                output[j++] = 'b';
                break;
            case '\f':
                if (j >= output_size - 3) {
                    return -1;  
                }
                output[j++] = '\\';
                output[j++] = 'f';
                break;
            case '\n':
                if (j >= output_size - 3) {
                    return -1; 
                }
                output[j++] = '\\';
                output[j++] = 'n';
                break;
            case '\r':
                if (j >= output_size - 3) {
                    return -1; 
                }
                output[j++] = '\\';
                output[j++] = 'r';
                break;
            case '\t':
                if (j >= output_size - 3) {
                    return -1; 
                }
                output[j++] = '\\';
                output[j++] = 't';
                break;
            default:
                // Escape control characters (0x00-0x1F) as \uXXXX
                if (c < 0x20) {
                    if (j >= output_size - 7) {
                        return -1; 
                    }
                    output[j++] = '\\';
                    output[j++] = 'u';
                    output[j++] = '0';
                    output[j++] = '0';
                    // Convert to hex (upper case)
                    char hex1 = (c >> 4) & 0x0F;
                    char hex2 = c & 0x0F;
                    output[j++] = (hex1 < 10) ? ('0' + hex1) : ('A' + hex1 - 10);
                    output[j++] = (hex2 < 10) ? ('0' + hex2) : ('A' + hex2 - 10);
                } else {
                    // Handle UTF-8 multi-byte characters
                    int utf8_len = 1;
                    if ((c & 0xE0) == 0xC0) {
                        utf8_len = 2;  // 2-byte UTF-8
                    } else if ((c & 0xF0) == 0xE0) {
                        utf8_len = 3;  // 3-byte UTF-8
                    } else if ((c & 0xF8) == 0xF0) {
                        utf8_len = 4;  // 4-byte UTF-8
                    }
                    
                    // Copy UTF-8 sequence if valid, otherwise copy single byte
                    if (utf8_len > 1) {
                        // Check if we have enough bytes and they are valid continuation bytes
                        int valid = 1;
                        for (int k = 1; k < utf8_len; k++) {
                            if (input[i + k] == '\0' || (input[i + k] & 0xC0) != 0x80) {
                                valid = 0;
                                break;
                            }
                        }
                        if (valid && j + utf8_len <= output_size - 1) {
                            // Copy entire UTF-8 sequence
                            for (int k = 0; k < utf8_len; k++) {
                                output[j++] = input[i + k];
                            }
                            i += utf8_len - 1;
                        } else {
                            // Invalid UTF-8 or buffer too small, copy single byte
                            if (j >= output_size - 2) {
                                return -1;
                            }
                            output[j++] = input[i];
                        }
                    } else {
                        // Single byte character (ASCII or invalid)
                        if (j >= output_size - 2) {
                            return -1;
                        }
                        output[j++] = input[i];
                    }
                }
                break;
        }
        i++;
    }
    output[j] = '\0';
    return 0;
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
    streamer_callback streamer = {
        .callback_func = print_callback,
        .args = NULL
    };
    char prompt[MAX_PROMPT_LENGTH];
    char message_json[MAX_MESSAGE_JSON_LENGTH];
    char output_buffer[MAX_JSON_LENGTH];
    size_t output_size = 0;
    char assistant_message_json[MAX_ASSISTANT_MESSAGE_JSON_LENGTH];
    char escaped_prompt[MAX_ESCAPED_PROMPT_LENGTH];
    char escaped_output[MAX_ESCAPED_OUTPUT_LENGTH];

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
            printf("question:\n");
            continue;
        }

        if (json_escape_string(prompt, escaped_prompt, sizeof(escaped_prompt)) != 0) {
            fprintf(stderr, "[ERROR] Failed to escape prompt: buffer too small\n");
            continue;
        }
        
        int message_json_len = snprintf(message_json, sizeof(message_json), 
                 "{\"role\": \"user\", \"content\": \"%s\"}", escaped_prompt);
        if (message_json_len < 0 || (size_t)message_json_len >= sizeof(message_json)) {
            fprintf(stderr, "[ERROR] Message JSON truncated: buffer too small (needed %d bytes)\n", message_json_len);
            continue;
        }
        
        if (message_container) {
            ov_genai_json_container_free(message_container);
            message_container = NULL;
        }
        CHECK_JSON_CONTAINER_STATUS(ov_genai_json_container_create_from_json_string(
            &message_container, message_json));
        
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
            
            if (json_escape_string(output_buffer, escaped_output, sizeof(escaped_output)) != 0) {
                fprintf(stderr, "[ERROR] Failed to escape output: buffer too small\n");
                ov_genai_decoded_results_free(results);
                results = NULL;
                continue;
            }
            
            int assistant_message_json_len = snprintf(assistant_message_json, sizeof(assistant_message_json),
                     "{\"role\": \"assistant\", \"content\": \"%s\"}", escaped_output);
            if (assistant_message_json_len < 0 || (size_t)assistant_message_json_len >= sizeof(assistant_message_json)) {
                fprintf(stderr, "[ERROR] Assistant message JSON truncated: buffer too small (needed %d bytes)\n", assistant_message_json_len);
                ov_genai_decoded_results_free(results);
                results = NULL;
                continue;
            }
            
            if (assistant_message_container) {
                ov_genai_json_container_free(assistant_message_container);
                assistant_message_container = NULL;
            }
            CHECK_JSON_CONTAINER_STATUS(ov_genai_json_container_create_from_json_string(
                &assistant_message_container, assistant_message_json));
            
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
