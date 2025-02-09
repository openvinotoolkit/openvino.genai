// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ostream>

#include "openvino/genai/image_generation/scheduler.hpp"

#include "json_utils.hpp"

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

enum class InterpolationType {
    LINEAR,
    LOG_LINEAR
};

enum class FinalSigmaType {
    ZERO,
    SIGMA_MIN
};

enum class TimestepType {
    DISCRETE,
    CONTINUOUS
};

namespace utils {

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, BetaSchedule& param);

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, PredictionType& param);

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, Scheduler::Type& param);

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, TimestepSpacing& param);

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, InterpolationType& param);

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, FinalSigmaType& param);

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, TimestepType& param);

}  // namespace utils
}  // namespace genai
}  // namespace ov

std::ostream& operator<<(std::ostream& os, const ov::genai::Scheduler::Type& scheduler_type);
