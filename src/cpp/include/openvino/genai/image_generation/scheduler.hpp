// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "openvino/genai/visibility.hpp"
#include "openvino/core/deprecated.hpp"

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS Scheduler {
public:
    enum Type {
        AUTO,
        LCM,
        DDIM,
        LMS_DISCRETE OPENVINO_ENUM_DEPRECATED("LMS_DISCRETE is deprecated. Please, select different scheduler type") = DDIM,
        EULER_DISCRETE,
        FLOW_MATCH_EULER_DISCRETE,
        PNDM,
        EULER_ANCESTRAL_DISCRETE
    };

    static std::shared_ptr<Scheduler> from_config(const std::filesystem::path& scheduler_config_path,
                                                  Type scheduler_type = AUTO);

    virtual ~Scheduler();
};

} // namespace genai
} // namespace ov
