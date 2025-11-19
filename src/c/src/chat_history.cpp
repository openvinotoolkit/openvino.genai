// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/chat_history.h"

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

ov_genai_chat_history_status_e ov_genai_chat_history_create_from_json(
    const char* messages_json,
    ov_genai_chat_history** history) {
    if (!messages_json || !history) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        ov::genai::JsonContainer messages = ov::genai::JsonContainer::from_json_string(std::string(messages_json));
        if (!messages.is_array()) {
            return OV_GENAI_CHAT_HISTORY_INVALID_JSON;
        }
        std::unique_ptr<ov_genai_chat_history> _history = std::make_unique<ov_genai_chat_history>();
        _history->object = std::make_shared<ov::genai::ChatHistory>(messages);
        *history = _history.release();
    } catch (const ov::Exception& e) {
        return OV_GENAI_CHAT_HISTORY_INVALID_JSON;
    } catch (const std::exception& e) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
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
    const char* message_json) {
    if (!history || !(history->object) || !message_json) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        ov::genai::JsonContainer message = ov::genai::JsonContainer::from_json_string(std::string(message_json));
        history->object->push_back(message);
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
    char* output,
    size_t* output_size) {
    if (!history || !(history->object) || !output_size) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        const ov::genai::JsonContainer& messages = history->object->get_messages();
        std::string json_str = messages.to_json_string();
        
        if (!output) {
            *output_size = json_str.length() + 1;
        } else {
            if (*output_size < json_str.length() + 1) {
                return OV_GENAI_CHAT_HISTORY_OUT_OF_BOUNDS;
            }
            std::strncpy(output, json_str.c_str(), json_str.length() + 1);
            output[json_str.length()] = '\0';
            *output_size = json_str.length() + 1;
        }
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
    char* output,
    size_t* output_size) {
    if (!history || !(history->object) || !output_size) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        if (index >= history->object->size()) {
            return OV_GENAI_CHAT_HISTORY_OUT_OF_BOUNDS;
        }
        ov::genai::JsonContainer message = (*history->object)[index];
        std::string json_str = message.to_json_string();
        
        if (!output) {
            *output_size = json_str.length() + 1;
        } else {
            if (*output_size < json_str.length() + 1) {
                return OV_GENAI_CHAT_HISTORY_OUT_OF_BOUNDS;
            }
            std::strncpy(output, json_str.c_str(), json_str.length() + 1);
            output[json_str.length()] = '\0';
            *output_size = json_str.length() + 1;
        }
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
    char* output,
    size_t* output_size) {
    if (!history || !(history->object) || !output_size) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        if (history->object->empty()) {
            return OV_GENAI_CHAT_HISTORY_EMPTY;
        }
        ov::genai::JsonContainer message = history->object->first();
        std::string json_str = message.to_json_string();
        
        if (!output) {
            *output_size = json_str.length() + 1;
        } else {
            if (*output_size < json_str.length() + 1) {
                return OV_GENAI_CHAT_HISTORY_OUT_OF_BOUNDS;
            }
            std::strncpy(output, json_str.c_str(), json_str.length() + 1);
            output[json_str.length()] = '\0';
            *output_size = json_str.length() + 1;
        }
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
    char* output,
    size_t* output_size) {
    if (!history || !(history->object) || !output_size) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        if (history->object->empty()) {
            return OV_GENAI_CHAT_HISTORY_EMPTY;
        }
        ov::genai::JsonContainer message = history->object->last();
        std::string json_str = message.to_json_string();
        
        if (!output) {
            *output_size = json_str.length() + 1;
        } else {
            if (*output_size < json_str.length() + 1) {
                return OV_GENAI_CHAT_HISTORY_OUT_OF_BOUNDS;
            }
            std::strncpy(output, json_str.c_str(), json_str.length() + 1);
            output[json_str.length()] = '\0';
            *output_size = json_str.length() + 1;
        }
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
    const char* tools_json) {
    if (!history || !(history->object) || !tools_json) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        ov::genai::JsonContainer tools = ov::genai::JsonContainer::from_json_string(std::string(tools_json));
        if (!tools.is_array()) {
            return OV_GENAI_CHAT_HISTORY_INVALID_JSON;
        }
        history->object->set_tools(tools);
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
    char* output,
    size_t* output_size) {
    if (!history || !(history->object) || !output_size) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        const ov::genai::JsonContainer& tools = history->object->get_tools();
        std::string json_str = tools.to_json_string();
        
        if (!output) {
            *output_size = json_str.length() + 1;
        } else {
            if (*output_size < json_str.length() + 1) {
                return OV_GENAI_CHAT_HISTORY_OUT_OF_BOUNDS;
            }
            std::strncpy(output, json_str.c_str(), json_str.length() + 1);
            output[json_str.length()] = '\0';
            *output_size = json_str.length() + 1;
        }
    } catch (const std::exception& e) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    } catch (...) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    }
    return OV_GENAI_CHAT_HISTORY_OK;
}

ov_genai_chat_history_status_e ov_genai_chat_history_set_extra_context(
    ov_genai_chat_history* history,
    const char* extra_context_json) {
    if (!history || !(history->object) || !extra_context_json) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        ov::genai::JsonContainer extra_context = ov::genai::JsonContainer::from_json_string(std::string(extra_context_json));
        if (!extra_context.is_object()) {
            return OV_GENAI_CHAT_HISTORY_INVALID_JSON;
        }
        history->object->set_extra_context(extra_context);
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
    char* output,
    size_t* output_size) {
    if (!history || !(history->object) || !output_size) {
        return OV_GENAI_CHAT_HISTORY_INVALID_PARAM;
    }
    try {
        const ov::genai::JsonContainer& extra_context = history->object->get_extra_context();
        std::string json_str = extra_context.to_json_string();
        
        if (!output) {
            *output_size = json_str.length() + 1;
        } else {
            if (*output_size < json_str.length() + 1) {
                return OV_GENAI_CHAT_HISTORY_OUT_OF_BOUNDS;
            }
            std::strncpy(output, json_str.c_str(), json_str.length() + 1);
            output[json_str.length()] = '\0';
            *output_size = json_str.length() + 1;
        }
    } catch (const std::exception& e) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    } catch (...) {
        return OV_GENAI_CHAT_HISTORY_ERROR;
    }
    return OV_GENAI_CHAT_HISTORY_OK;
}

