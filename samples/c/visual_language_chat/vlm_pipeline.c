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
#include "json_utils.h"
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
    int return_code = EXIT_FAILURE;
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
    char* output_dynamic = NULL;
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
    if (!tensors || tensor_count == 0) {
        fprintf(stderr, "[ERROR] Failed to load image: %s\n", image_path);
        goto err;
    }

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

        output_size = 0;
        CHECK_STATUS(ov_genai_vlm_decoded_results_get_string(results, NULL, &output_size));
        output_dynamic = (char*)malloc(output_size);
        if (!output_dynamic) {
            fprintf(stderr, "[ERROR] Failed to allocate memory for output buffer (%zu bytes)\n", output_size);
            ov_genai_vlm_decoded_results_free(results);
            results = NULL;
            continue;
        }
        CHECK_STATUS(ov_genai_vlm_decoded_results_get_string(results, output_dynamic, &output_size));

        if (json_escape_string(output_dynamic, escaped_output, sizeof(escaped_output)) != 0) {
            fprintf(stderr, "[ERROR] Failed to escape output: buffer too small\n");
            free(output_dynamic);
            output_dynamic = NULL;
            ov_genai_vlm_decoded_results_free(results);
            results = NULL;
            continue;
        }
        free(output_dynamic);
        output_dynamic = NULL;

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
    return_code = EXIT_SUCCESS;

err:
    if (tensors) {
        free_tensor_array((ov_tensor_t**)tensors, tensor_count);
    }
    if (results) {
        ov_genai_vlm_decoded_results_free(results);
    }
    if (output_dynamic) {
        free(output_dynamic);
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

    return return_code;
}
