// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// This is a C wrapper for ov::genai::ChatHistory class.

#pragma once

#include "visibility.h"
#include <stddef.h>

// Forward declaration for JsonContainer
typedef struct ov_genai_json_container_opaque ov_genai_json_container;

/**
 * @struct ov_genai_chat_history
 * @brief Opaque type for ChatHistory
 */
typedef struct ov_genai_chat_history_opaque ov_genai_chat_history;

/**
 * @brief Status codes for chat history operations
 */
typedef enum {
    OV_GENAI_CHAT_HISTORY_OK = 0,
    OV_GENAI_CHAT_HISTORY_INVALID_PARAM = -1,
    OV_GENAI_CHAT_HISTORY_OUT_OF_BOUNDS = -2,
    OV_GENAI_CHAT_HISTORY_EMPTY = -3,
    OV_GENAI_CHAT_HISTORY_INVALID_JSON = -4,
    OV_GENAI_CHAT_HISTORY_ERROR = -5
} ov_genai_chat_history_status_e;

/**
 * @brief Create a new empty ChatHistory instance.
 * @param history A pointer to the newly created ov_genai_chat_history.
 * @return ov_genai_chat_history_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_chat_history_status_e ov_genai_chat_history_create(ov_genai_chat_history** history);

/**
 * @brief Create a ChatHistory instance from a JsonContainer (array).
 * @param history A pointer to the newly created ov_genai_chat_history.
 * @param messages A JsonContainer containing an array of message objects.
 * @return ov_genai_chat_history_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_chat_history_status_e ov_genai_chat_history_create_from_json_container(
    ov_genai_chat_history** history,
    const ov_genai_json_container* messages
   );

/**
 * @brief Release the memory allocated by ov_genai_chat_history.
 * @param history A pointer to the ov_genai_chat_history to free memory.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_chat_history_free(ov_genai_chat_history* history);

/**
 * @brief Add a message to the chat history from a JsonContainer.
 * @param history A pointer to the ov_genai_chat_history instance.
 * @param message A JsonContainer containing a message object (e.g., {"role": "user", "content": "Hello"}).
 * @return ov_genai_chat_history_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_chat_history_status_e ov_genai_chat_history_push_back(
    ov_genai_chat_history* history,
    const ov_genai_json_container* message);

/**
 * @brief Remove the last message from the chat history.
 * @param history A pointer to the ov_genai_chat_history instance.
 * @return ov_genai_chat_history_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_chat_history_status_e ov_genai_chat_history_pop_back(ov_genai_chat_history* history);

/**
 * @brief Get all messages as a JsonContainer (array).
 * @param history A pointer to the ov_genai_chat_history instance.
 * @param messages A pointer to store the returned JsonContainer containing all messages.
 * @return ov_genai_chat_history_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_chat_history_status_e ov_genai_chat_history_get_messages(
    const ov_genai_chat_history* history,
    ov_genai_json_container** messages);

/**
 * @brief Get a message at a specific index as a JsonContainer.
 * @param history A pointer to the ov_genai_chat_history instance.
 * @param index The index of the message to retrieve.
 * @param message A pointer to store the returned JsonContainer containing the message.
 * @return ov_genai_chat_history_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_chat_history_status_e ov_genai_chat_history_get_message(
    const ov_genai_chat_history* history,
    size_t index,
    ov_genai_json_container** message);

/**
 * @brief Get the first message as a JsonContainer.
 * @param history A pointer to the ov_genai_chat_history instance.
 * @param message A pointer to store the returned JsonContainer containing the first message.
 * @return ov_genai_chat_history_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_chat_history_status_e ov_genai_chat_history_get_first(
    const ov_genai_chat_history* history,
    ov_genai_json_container** message);

/**
 * @brief Get the last message as a JsonContainer.
 * @param history A pointer to the ov_genai_chat_history instance.
 * @param message A pointer to store the returned JsonContainer containing the last message.
 * @return ov_genai_chat_history_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_chat_history_status_e ov_genai_chat_history_get_last(
    const ov_genai_chat_history* history,
    ov_genai_json_container** message);

/**
 * @brief Clear all messages from the chat history.
 * @param history A pointer to the ov_genai_chat_history instance.
 * @return ov_genai_chat_history_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_chat_history_status_e ov_genai_chat_history_clear(ov_genai_chat_history* history);

/**
 * @brief Get the number of messages in the chat history.
 * @param history A pointer to the ov_genai_chat_history instance.
 * @param size A pointer to store the size (number of messages).
 * @return ov_genai_chat_history_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_chat_history_status_e ov_genai_chat_history_size(
    const ov_genai_chat_history* history,
    size_t* size);

/**
 * @brief Check if the chat history is empty.
 * @param history A pointer to the ov_genai_chat_history instance.
 * @param empty A pointer to store the boolean result (1 for empty, 0 for not empty).
 * @return ov_genai_chat_history_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_chat_history_status_e ov_genai_chat_history_empty(
    const ov_genai_chat_history* history,
    int* empty);

/**
 * @brief Set tools definitions (for function calling) as a JsonContainer (array).
 * @param history A pointer to the ov_genai_chat_history instance.
 * @param tools A JsonContainer containing an array of tool definitions.
 * @return ov_genai_chat_history_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_chat_history_status_e ov_genai_chat_history_set_tools(
    ov_genai_chat_history* history,
    const ov_genai_json_container* tools);

/**
 * @brief Get tools definitions as a JsonContainer (array).
 * @param history A pointer to the ov_genai_chat_history instance.
 * @param tools A pointer to store the returned JsonContainer containing tools definitions.
 * @return ov_genai_chat_history_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_chat_history_status_e ov_genai_chat_history_get_tools(
    const ov_genai_chat_history* history,
    ov_genai_json_container** tools);

/**
 * @brief Set extra context (for custom template variables) as a JsonContainer (object).
 * @param history A pointer to the ov_genai_chat_history instance.
 * @param extra_context A JsonContainer containing an object with extra context.
 * @return ov_genai_chat_history_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_chat_history_status_e ov_genai_chat_history_set_extra_context(
    ov_genai_chat_history* history,
    const ov_genai_json_container* extra_context);

/**
 * @brief Get extra context as a JsonContainer (object).
 * @param history A pointer to the ov_genai_chat_history instance.
 * @param extra_context A pointer to store the returned JsonContainer containing extra context.
 * @return ov_genai_chat_history_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_genai_chat_history_status_e ov_genai_chat_history_get_extra_context(
    const ov_genai_chat_history* history,
    ov_genai_json_container** extra_context);

