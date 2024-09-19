// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ostream>

#include "openvino/genai/text2image/pipeline.hpp"

#include "utils.hpp"

namespace ov {
namespace genai {

enum class BetaSchedule {
    LINEAR,
    SCALED_LINEAR,
    SQUAREDCOS_CAP_V2
};

enum class PredictionType {
    EPSILON,
    SAMPLE,
    V_PREDICTION
};

enum class TimestepSpacing {
    LINSPACE,
    TRAILING,
    LEADING
};

namespace utils {

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, BetaSchedule& param);

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, PredictionType& param);

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, Text2ImagePipeline::Scheduler::Type& param);

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, TimestepSpacing& param);

}  // namespace utils
}  // namespace genai
}  // namespace ov

std::ostream& operator<<(std::ostream& os, const ov::genai::Text2ImagePipeline::Scheduler::Type& scheduler_type);
