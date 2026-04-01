// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file vlm_pipeline.c
 * @brief Example demonstrating how to use OpenVINO GenAI VLM Pipeline C API
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openvino/genai/c/chat_history.h"
#include "openvino/genai/c/json_container.h"
#include "openvino/genai/c/vlm_pipeline.h"
#include "load_image.h"

#define MAX_PROMPT_LENGTH 1024
#define MAX_JSON_LENGTH 4096

// Worst-case escape: each char -> \uXXXX (6 bytes) + null terminator
#define MAX_ESCAPED_PROMPT_LENGTH (MAX_PROMPT_LENGTH * 6 + 1)

// JSON template: {"role": "user", "content": ""} ~= 32 bytes
#define MAX_MESSAGE_JSON_LENGTH (MAX_ESCAPED_PROMPT_LENGTH + 32 + 1)

// Worst-case escaped output: MAX_JSON_LENGTH * 6 + null terminator
#define MAX_ESCAPED_OUTPUT_LENGTH ((MAX_JSON_LENGTH - 1) * 6 + 1)

// JSON template: {"role": "assistant", "content": ""} ~= 35 bytes
#define MAX_ASSISTANT_MESSAGE_JSON_LENGTH (MAX_ESCAPED_OUTPUT_LENGTH + 35 + 1)

#define CHECK_STATUS(return_status)                                                          \
    if ((return_status) != OK) {                                                             \
        fprintf(stderr, "[ERROR] return status %d, line %d\n", (return_status), __LINE__);  \
        goto err;                                                                            \
    }

#define CHECK_CHAT_HISTORY_STATUS(return_status)                                                         \
    if ((return_status) != OV_GENAI_CHAT_HISTORY_OK) {                                                   \
        fprintf(stderr, "[ERROR] chat history status %d, line %d\n", (return_status), __LINE__);        \
        goto err;                                                                                         \
    }

#define CHECK_JSON_CONTAINER_STATUS(return_status)                                                       \
    if ((return_status) != OV_GENAI_JSON_CONTAINER_OK) {                                                 \
        fprintf(stderr, "[ERROR] json container status %d, line %d\n", (return_status), __LINE__);      \
        goto err;                                                                                         \
    }

// Returns 0 on success, -1 if buffer is too small.
static int json_escape_string(const char* input, char* output, size_t output_size) {
    size_t i = 0;
    size_t j = 0;

    if (!input || !output || output_size == 0) {
        return -1;
    }

    while (input[i] != '\0' && j < output_size - 1) {
        unsigned char c = (unsigned char)input[i];
        switch (c) {
            case '"':
            case '\\':
                if (j >= output_size - 3) {
                    return -1;
                }
                output[j++] = '\\';
                output[j++] = (char)c;
                break;
            case '\b':
            case '\f':
            case '\n':
            case '\r':
            case '\t':
                if (j >= output_size - 3) {
                    return -1;
                }
                output[j++] = '\\';
                output[j++] = (c == '\b') ? 'b' :
                              (c == '\f') ? 'f' :
                              (c == '\n') ? 'n' :
                              (c == '\r') ? 'r' : 't';
                break;
            default:
                if (c < 0x20) {
                    char hex1;
                    char hex2;
                    if (j >= output_size - 7) {
                        return -1;
                    }
                    output[j++] = '\\';
                    output[j++] = 'u';
                    output[j++] = '0';
                    output[j++] = '0';
                    hex1 = (char)((c >> 4) & 0x0F);
                    hex2 = (char)(c & 0x0F);
                    output[j++] = (hex1 < 10) ? ('0' + hex1) : ('A' + hex1 - 10);
                    output[j++] = (hex2 < 10) ? ('0' + hex2) : ('A' + hex2 - 10);
                } else {
                    int utf8_len = 1;
                    if ((c & 0xE0) == 0xC0) {
                        utf8_len = 2;
                    } else if ((c & 0xF0) == 0xE0) {
                        utf8_len = 3;
                    } else if ((c & 0xF8) == 0xF0) {
                        utf8_len = 4;
                    }

                    if (utf8_len > 1) {
                        int valid = 1;
                        int k = 0;
                        for (k = 1; k < utf8_len; ++k) {
                            if (input[i + k] == '\0' || (((unsigned char)input[i + k] & 0xC0) != 0x80)) {
                                valid = 0;
                                break;
                            }
                        }
                        if (valid && j + (size_t)utf8_len <= output_size - 1) {
                            for (k = 0; k < utf8_len; ++k) {
                                output[j++] = input[i + k];
                            }
                            i += (size_t)utf8_len - 1;
                        } else {
                            if (j >= output_size - 2) {
                                return -1;
                            }
                            output[j++] = input[i];
                        }
                    } else {
                        if (j >= output_size - 2) {
                            return -1;
                        }
                        output[j++] = input[i];
                    }
                }
                break;
        }
        ++i;
    }

    output[j] = '\0';
    return 0;
}

static ov_genai_streaming_status_e stream_callback(const char* str, void* args) {
    (void)args;
    if (str) {
        printf("%s", str);
        fflush(stdout);
        return OV_GENAI_STREAMING_STATUS_RUNNING;
    }
    return OV_GENAI_STREAMING_STATUS_STOP;
}

int main(int argc, char* argv[]) {
    const char* models_path = NULL;
    const char* device = NULL;
    const char* image_path = NULL;
    const ov_tensor_t** tensors = NULL;
    size_t tensor_count = 0;
    ov_genai_vlm_pipeline* pipeline = NULL;
    ov_genai_generation_config* config = NULL;
    ov_genai_chat_history* chat_history = NULL;
    ov_genai_vlm_decoded_results* results = NULL;
    ov_genai_json_container* message_container = NULL;
    ov_genai_json_container* assistant_message_container = NULL;
    streamer_callback callback = {
        .callback_func = stream_callback,
        .args = NULL
    };
    int is_first_turn = 1;
    char prompt[MAX_PROMPT_LENGTH];
    char message_json[MAX_MESSAGE_JSON_LENGTH];
    char assistant_message_json[MAX_ASSISTANT_MESSAGE_JSON_LENGTH];
    char escaped_prompt[MAX_ESCAPED_PROMPT_LENGTH];
    char escaped_output[MAX_ESCAPED_OUTPUT_LENGTH];
    char output_buffer[MAX_JSON_LENGTH];
    size_t output_size = 0;

    if (argc < 4) {
        printf("Usage: %s <models_path> <device> <image_path>\n", argv[0]);
        printf("Example: %s ./models CPU ./image.jpg\n", argv[0]);
        return EXIT_FAILURE;
    }

    models_path = argv[1];
    device = argv[2];
    image_path = argv[3];
    tensors = load_images(image_path, &tensor_count);

    CHECK_STATUS(ov_genai_vlm_pipeline_create(models_path, device, 0, &pipeline));
    CHECK_STATUS(ov_genai_generation_config_create(&config));
    CHECK_STATUS(ov_genai_generation_config_set_max_new_tokens(config, 100));
    CHECK_CHAT_HISTORY_STATUS(ov_genai_chat_history_create(&chat_history));

    printf("question:\n");
    while (fgets(prompt, MAX_PROMPT_LENGTH, stdin)) {
        const ov_tensor_t** turn_tensors = NULL;
        size_t turn_tensor_count = 0;
        int message_json_len = 0;
        int assistant_message_json_len = 0;

        prompt[strcspn(prompt, "\n")] = 0;
        if (strlen(prompt) == 0) {
            printf("question:\n");
            continue;
        }

        if (json_escape_string(prompt, escaped_prompt, sizeof(escaped_prompt)) != 0) {
            fprintf(stderr, "[ERROR] Failed to escape prompt: buffer too small\n");
            continue;
        }

        message_json_len = snprintf(
            message_json,
            sizeof(message_json),
            "{\"role\": \"user\", \"content\": \"%s\"}",
            escaped_prompt
        );
        if (message_json_len < 0 || (size_t)message_json_len >= sizeof(message_json)) {
            fprintf(stderr, "[ERROR] Message JSON truncated: buffer too small (needed %d bytes)\n", message_json_len);
            continue;
        }

        if (message_container) {
            ov_genai_json_container_free(message_container);
            message_container = NULL;
        }
        CHECK_JSON_CONTAINER_STATUS(
            ov_genai_json_container_create_from_json_string(&message_container, message_json)
        );
        CHECK_CHAT_HISTORY_STATUS(ov_genai_chat_history_push_back(chat_history, message_container));

        if (is_first_turn) {
            turn_tensors = tensors;
            turn_tensor_count = tensor_count;
        }

        CHECK_STATUS(
            ov_genai_vlm_pipeline_generate_with_history(
                pipeline,
                chat_history,
                turn_tensors,
                turn_tensor_count,
                config,
                &callback,
                &results
            )
        );

        output_size = sizeof(output_buffer);
        CHECK_STATUS(ov_genai_vlm_decoded_results_get_string(results, output_buffer, &output_size));

        if (json_escape_string(output_buffer, escaped_output, sizeof(escaped_output)) != 0) {
            fprintf(stderr, "[ERROR] Failed to escape output: buffer too small\n");
            ov_genai_vlm_decoded_results_free(results);
            results = NULL;
            continue;
        }

        assistant_message_json_len = snprintf(
            assistant_message_json,
            sizeof(assistant_message_json),
            "{\"role\": \"assistant\", \"content\": \"%s\"}",
            escaped_output
        );
        if (assistant_message_json_len < 0 || (size_t)assistant_message_json_len >= sizeof(assistant_message_json)) {
            fprintf(stderr, "[ERROR] Assistant message JSON truncated: buffer too small (needed %d bytes)\n", assistant_message_json_len);
            ov_genai_vlm_decoded_results_free(results);
            results = NULL;
            continue;
        }

        if (assistant_message_container) {
            ov_genai_json_container_free(assistant_message_container);
            assistant_message_container = NULL;
        }
        CHECK_JSON_CONTAINER_STATUS(
            ov_genai_json_container_create_from_json_string(&assistant_message_container, assistant_message_json)
        );
        CHECK_CHAT_HISTORY_STATUS(ov_genai_chat_history_push_back(chat_history, assistant_message_container));

        ov_genai_vlm_decoded_results_free(results);
        results = NULL;
        is_first_turn = 0;

        printf("\n----------\nquestion:\n");
    }

err:
    if (results) {
        ov_genai_vlm_decoded_results_free(results);
    }
    if (message_container) {
        ov_genai_json_container_free(message_container);
    }
    if (assistant_message_container) {
        ov_genai_json_container_free(assistant_message_container);
    }
    if (chat_history) {
        ov_genai_chat_history_free(chat_history);
    }
    if (config) {
        ov_genai_generation_config_free(config);
    }
    if (pipeline) {
        ov_genai_vlm_pipeline_free(pipeline);
    }

    return EXIT_SUCCESS;
}
