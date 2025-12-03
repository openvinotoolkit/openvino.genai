// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/chat_history.h"
#include "openvino/genai/c/json_container.h"

#include "openvino/genai/chat_history.hpp"
#include "openvino/genai/json_container.hpp"
#include "openvino/core/except.hpp"
#include "types_c.h"
#include <cstring>
#include <string>
#include <memory>

ov_genai_chat_history_status_e ov_genai_chat_history_create(ov_genai_chat_history** history) {
    if (!history) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_chat_history> _history = std::make_unique<ov_genai_chat_history>();
        _history->object = std::make_shared<ov::genai::ChatHistory>();
        *history = _history.release();
    } catch (const std::exception& e) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    } catch (...) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    }
    return OV_GENAI_CHAT_HISTORY_OK;
}

ov_genai_chat_history_status_e ov_genai_chat_history_create_from_json_container(
    ov_genai_chat_history** history,
    const ov_genai_json_container* messages
) {
    if (!messages || !(messages->object) || !history) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        if (!messages->object->is_array()) {
            return OV_GENAI_CHAT_HISTORY_INVALID_JSON;
        }
        std::unique_ptr<ov_genai_chat_history> _history = std::make_unique<ov_genai_chat_history>();
        _history->object = std::make_shared<ov::genai::ChatHistory>(*(messages->object));
        *history = _history.release();
    } catch (...) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    }
    return OV_GENAI_CHAT_HISTORY_OK;
}

void ov_genai_chat_history_free(ov_genai_chat_history* history) {
    if (history) {
        delete history;
    }
}

ov_genai_chat_history_status_e ov_genai_chat_history_push_back(
    ov_genai_chat_history* history,
    const ov_genai_json_container* message) {
    if (!history || !(history->object) || !message || !(message->object)) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        history->object->push_back(*(message->object));
    } catch (const ov::Exception& e) {
        return OV_GENAI_CHAT_HISTORY_INVALID_JSON;
    } catch (const std::exception& e) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    } catch (...) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    }
    return OV_GENAI_CHAT_HISTORY_OK;
}

ov_genai_chat_history_status_e ov_genai_chat_history_pop_back(ov_genai_chat_history* history) {
    if (!history || !(history->object)) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        if (history->object->empty()) {
            return OV_GENAI_CHAT_HISTORY_EMPTY;
        }
        history->object->pop_back();
    } catch (const ov::Exception& e) {
        return OV_GENAI_CHAT_HISTORY_EMPTY;
    } catch (const std::exception& e) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    } catch (...) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    }
    return OV_GENAI_CHAT_HISTORY_OK;
}

ov_genai_chat_history_status_e ov_genai_chat_history_get_messages(
    const ov_genai_chat_history* history,
    ov_genai_json_container** messages) {
    if (!history || !(history->object) || !messages) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        const ov::genai::JsonContainer& messages_ref = history->object->get_messages();
        std::unique_ptr<ov_genai_json_container> _messages = std::make_unique<ov_genai_json_container>();
        _messages->object = std::make_shared<ov::genai::JsonContainer>(messages_ref.share());
        *messages = _messages.release();
    } catch (const std::exception& e) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    } catch (...) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    }
    return OV_GENAI_CHAT_HISTORY_OK;
}

ov_genai_chat_history_status_e ov_genai_chat_history_get_message(
    const ov_genai_chat_history* history,
    size_t index,
    ov_genai_json_container** message) {
    if (!history || !(history->object) || !message) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        if (index >= history->object->size()) {
            return OV_GENAI_CHAT_HISTORY_OUT_OF_BOUNDS;
        }
        ov::genai::JsonContainer message_ref = (*history->object)[index];
        std::unique_ptr<ov_genai_json_container> _message = std::make_unique<ov_genai_json_container>();
        _message->object = std::make_shared<ov::genai::JsonContainer>(std::move(message_ref));
        *message = _message.release();
    } catch (const ov::Exception& e) {
        return OV_GENAI_CHAT_HISTORY_OUT_OF_BOUNDS;
    } catch (const std::exception& e) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    } catch (...) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    }
    return OV_GENAI_CHAT_HISTORY_OK;
}

ov_genai_chat_history_status_e ov_genai_chat_history_get_first(
    const ov_genai_chat_history* history,
    ov_genai_json_container** message) {
    if (!history || !(history->object) || !message) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        if (history->object->empty()) {
            return OV_GENAI_CHAT_HISTORY_EMPTY;
        }
        ov::genai::JsonContainer message_ref = history->object->first();
        std::unique_ptr<ov_genai_json_container> _message = std::make_unique<ov_genai_json_container>();
        _message->object = std::make_shared<ov::genai::JsonContainer>(std::move(message_ref));
        *message = _message.release();
    } catch (const ov::Exception& e) {
        return OV_GENAI_CHAT_HISTORY_EMPTY;
    } catch (const std::exception& e) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    } catch (...) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    }
    return OV_GENAI_CHAT_HISTORY_OK;
}

ov_genai_chat_history_status_e ov_genai_chat_history_get_last(
    const ov_genai_chat_history* history,
    ov_genai_json_container** message) {
    if (!history || !(history->object) || !message) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        if (history->object->empty()) {
            return OV_GENAI_CHAT_HISTORY_EMPTY;
        }
        ov::genai::JsonContainer message_ref = history->object->last();
        std::unique_ptr<ov_genai_json_container> _message = std::make_unique<ov_genai_json_container>();
        _message->object = std::make_shared<ov::genai::JsonContainer>(std::move(message_ref));
        *message = _message.release();
    } catch (const ov::Exception& e) {
        return OV_GENAI_CHAT_HISTORY_EMPTY;
    } catch (const std::exception& e) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    } catch (...) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    }
    return OV_GENAI_CHAT_HISTORY_OK;
}

ov_genai_chat_history_status_e ov_genai_chat_history_clear(ov_genai_chat_history* history) {
    if (!history || !(history->object)) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        history->object->clear();
    } catch (const std::exception& e) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    } catch (...) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    }
    return OV_GENAI_CHAT_HISTORY_OK;
}

ov_genai_chat_history_status_e ov_genai_chat_history_size(
    const ov_genai_chat_history* history,
    size_t* size) {
    if (!history || !(history->object) || !size) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        *size = history->object->size();
    } catch (const std::exception& e) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    } catch (...) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    }
    return OV_GENAI_CHAT_HISTORY_OK;
}

ov_genai_chat_history_status_e ov_genai_chat_history_empty(
    const ov_genai_chat_history* history,
    int* empty) {
    if (!history || !(history->object) || !empty) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        *empty = history->object->empty() ? 1 : 0;
    } catch (const std::exception& e) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    } catch (...) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    }
    return OV_GENAI_CHAT_HISTORY_OK;
}

ov_genai_chat_history_status_e ov_genai_chat_history_set_tools(
    ov_genai_chat_history* history,
    const ov_genai_json_container* tools) {
    if (!history || !(history->object) || !tools || !(tools->object)) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        if (!tools->object->is_array()) {
            return OV_GENAI_CHAT_HISTORY_INVALID_JSON;
        }
        history->object->set_tools(*(tools->object));
    } catch (const ov::Exception& e) {
        return OV_GENAI_CHAT_HISTORY_INVALID_JSON;
    } catch (const std::exception& e) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    } catch (...) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    }
    return OV_GENAI_CHAT_HISTORY_OK;
}

ov_genai_chat_history_status_e ov_genai_chat_history_get_tools(
    const ov_genai_chat_history* history,
    ov_genai_json_container** tools) {
    if (!history || !(history->object) || !tools) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        const ov::genai::JsonContainer& tools_ref = history->object->get_tools();
        std::unique_ptr<ov_genai_json_container> _tools = std::make_unique<ov_genai_json_container>();
        _tools->object = std::make_shared<ov::genai::JsonContainer>(tools_ref.share());
        *tools = _tools.release();
    } catch (const std::exception& e) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    } catch (...) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    }
    return OV_GENAI_CHAT_HISTORY_OK;
}

ov_genai_chat_history_status_e ov_genai_chat_history_set_extra_context(
    ov_genai_chat_history* history,
    const ov_genai_json_container* extra_context) {
    if (!history || !(history->object) || !extra_context || !(extra_context->object)) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        if (!extra_context->object->is_object()) {
            return OV_GENAI_CHAT_HISTORY_INVALID_JSON;
        }
        history->object->set_extra_context(*(extra_context->object));
    } catch (const ov::Exception& e) {
        return OV_GENAI_CHAT_HISTORY_INVALID_JSON;
    } catch (const std::exception& e) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    } catch (...) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    }
    return OV_GENAI_CHAT_HISTORY_OK;
}

ov_genai_chat_history_status_e ov_genai_chat_history_get_extra_context(
    const ov_genai_chat_history* history,
    ov_genai_json_container** extra_context) {
    if (!history || !(history->object) || !extra_context) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        const ov::genai::JsonContainer& extra_context_ref = history->object->get_extra_context();
        std::unique_ptr<ov_genai_json_container> _extra_context = std::make_unique<ov_genai_json_container>();
        _extra_context->object = std::make_shared<ov::genai::JsonContainer>(extra_context_ref.share());
        *extra_context = _extra_context.release();
    } catch (const std::exception& e) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    } catch (...) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    }
    return OV_GENAI_CHAT_HISTORY_OK;
}

