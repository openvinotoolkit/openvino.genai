// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/c/json_container.h"

#include "openvino/genai/json_container.hpp"
#include "openvino/core/except.hpp"
#include "types_c.h"
#include <cstring>
#include <string>
#include <memory>

ov_genai_json_container_status_e ov_genai_json_container_create(ov_genai_json_container** container) {
    if (!container) {
        return OV_GENAI_JSON_CONTAINER_INVALID_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_json_container> _container = std::make_unique<ov_genai_json_container>();
        _container->object = std::make_shared<ov::genai::JsonContainer>();
        *container = _container.release();
    } catch (...) {
        return OV_GENAI_JSON_CONTAINER_ERROR;
    }
    return OV_GENAI_JSON_CONTAINER_OK;
}

ov_genai_json_container_status_e ov_genai_json_container_create_from_json_string(
    const char* json_str,
    ov_genai_json_container** container) {
    if (!json_str || !container) {
        return OV_GENAI_JSON_CONTAINER_INVALID_PARAM;
    }
    try {
        ov::genai::JsonContainer json_obj = ov::genai::JsonContainer::from_json_string(std::string(json_str));
        std::unique_ptr<ov_genai_json_container> _container = std::make_unique<ov_genai_json_container>();
        _container->object = std::make_shared<ov::genai::JsonContainer>(std::move(json_obj));
        *container = _container.release();
    } catch (...) {
        return OV_GENAI_JSON_CONTAINER_INVALID_JSON;
    }
    return OV_GENAI_JSON_CONTAINER_OK;
}

ov_genai_json_container_status_e ov_genai_json_container_create_object(ov_genai_json_container** container) {
    if (!container) {
        return OV_GENAI_JSON_CONTAINER_INVALID_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_json_container> _container = std::make_unique<ov_genai_json_container>();
        _container->object = std::make_shared<ov::genai::JsonContainer>(ov::genai::JsonContainer::object());
        *container = _container.release();
    } catch (...) {
        return OV_GENAI_JSON_CONTAINER_ERROR;
    }
    return OV_GENAI_JSON_CONTAINER_OK;
}

ov_genai_json_container_status_e ov_genai_json_container_create_array(ov_genai_json_container** container) {
    if (!container) {
        return OV_GENAI_JSON_CONTAINER_INVALID_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_json_container> _container = std::make_unique<ov_genai_json_container>();
        _container->object = std::make_shared<ov::genai::JsonContainer>(ov::genai::JsonContainer::array());
        *container = _container.release();
    } catch (...) {
        return OV_GENAI_JSON_CONTAINER_ERROR;
    }
    return OV_GENAI_JSON_CONTAINER_OK;
}

void ov_genai_json_container_free(ov_genai_json_container* container) {
    if (container) {
        delete container;
    }
}

ov_genai_json_container_status_e ov_genai_json_container_to_json_string(
    const ov_genai_json_container* container,
    char* output,
    size_t* output_size) {
    if (!container || !(container->object) || !output_size) {
        return OV_GENAI_JSON_CONTAINER_INVALID_PARAM;
    }
    try {
        std::string json_str = container->object->to_json_string();
        
        if (!output) {
            *output_size = json_str.length() + 1;
        } else {
            if (*output_size < json_str.length() + 1) {
                return OV_GENAI_JSON_CONTAINER_OUT_OF_BOUNDS;
            }
            std::memcpy(output, json_str.c_str(), json_str.length() + 1);
            *output_size = json_str.length() + 1;
        }
    } catch (...) {
        return OV_GENAI_JSON_CONTAINER_ERROR;
    }
    return OV_GENAI_JSON_CONTAINER_OK;
}

ov_genai_json_container_status_e ov_genai_json_container_copy(
    const ov_genai_json_container* container,
    ov_genai_json_container** copy_container) {
    if (!container || !(container->object) || !copy_container) {
        return OV_GENAI_JSON_CONTAINER_INVALID_PARAM;
    }
    try {
        std::unique_ptr<ov_genai_json_container> _copy = std::make_unique<ov_genai_json_container>();
        _copy->object = std::make_shared<ov::genai::JsonContainer>(container->object->copy());
        *copy_container = _copy.release();
    } catch (...) {
        return OV_GENAI_JSON_CONTAINER_ERROR;
    }
    return OV_GENAI_JSON_CONTAINER_OK;
}

