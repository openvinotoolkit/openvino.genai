// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image_generation/schedulers/types.hpp"

#include "openvino/core/except.hpp"

namespace ov {
namespace genai {
namespace utils {

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, BetaSchedule& param) {
    if (data.contains(name) && data[name].is_string()) {
        std::string beta_schedule_str = data[name].get<std::string>();
        if (beta_schedule_str == "linear")
            param = BetaSchedule::LINEAR;
        else if (beta_schedule_str == "scaled_linear")
            param = BetaSchedule::SCALED_LINEAR;
        else if (beta_schedule_str == "squaredcos_cap_v2")
            param = BetaSchedule::SQUAREDCOS_CAP_V2;
        else if (!beta_schedule_str.empty()) {
            OPENVINO_THROW("Unsupported value for 'beta_schedule' ", beta_schedule_str);
        }
    }
}

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, PredictionType& param) {
    if (data.contains(name) && data[name].is_string()) {
        std::string prediction_type_str = data[name].get<std::string>();
        if (prediction_type_str == "epsilon")
            param = PredictionType::EPSILON;
        else if (prediction_type_str == "sample")
            param = PredictionType::SAMPLE;
        else if (prediction_type_str == "v_prediction")
            param = PredictionType::V_PREDICTION;
        else if (!prediction_type_str.empty()) {
            OPENVINO_THROW("Unsupported value for 'prediction_type' ", prediction_type_str);
        }
    }
}

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, Scheduler::Type& param) {
    if (data.contains(name) && data[name].is_string()) {
        std::string scheduler_type_str = data[name].get<std::string>();
        if (scheduler_type_str == "LCMScheduler")
            param = Scheduler::LCM;
        else if (scheduler_type_str == "DDIMScheduler")
            param = Scheduler::DDIM;
        else if (scheduler_type_str == "LMSDiscreteScheduler") {
            OPENVINO_SUPPRESS_DEPRECATED_START
            param = Scheduler::LMS_DISCRETE;
            OPENVINO_SUPPRESS_DEPRECATED_END
        } else if (scheduler_type_str == "EulerDiscreteScheduler")
            param = Scheduler::EULER_DISCRETE;
        else if (scheduler_type_str == "FlowMatchEulerDiscreteScheduler")
            param = Scheduler::FLOW_MATCH_EULER_DISCRETE;
        else if (scheduler_type_str == "PNDMScheduler")
            param = Scheduler::PNDM;
        else if (scheduler_type_str == "EulerAncestralDiscreteScheduler")
            param = Scheduler::EULER_ANCESTRAL_DISCRETE;
        else if (!scheduler_type_str.empty()) {
            OPENVINO_THROW("Unsupported value for 'scheduler' ", scheduler_type_str);
        }
    }
}

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, TimestepSpacing& param) {
    if (data.contains(name) && data[name].is_string()) {
        std::string timestep_spacing_str = data[name].get<std::string>();
        if (timestep_spacing_str == "linspace")
            param = TimestepSpacing::LINSPACE;
        else if (timestep_spacing_str == "trailing")
            param = TimestepSpacing::TRAILING;
        else if (timestep_spacing_str == "leading")
            param = TimestepSpacing::LEADING;
        else if (!timestep_spacing_str.empty()) {
            OPENVINO_THROW("Unsupported value for 'timestep_spacing' ", timestep_spacing_str);
        }
    }
}

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, InterpolationType& param) {
    if (data.contains(name) && data[name].is_string()) {
        std::string interpolation_type = data[name].get<std::string>();
        if (interpolation_type == "linear")
            param = InterpolationType::LINEAR;
        else if (interpolation_type == "log_linear")
            param = InterpolationType::LOG_LINEAR;
        else if (!interpolation_type.empty()) {
            OPENVINO_THROW("Unsupported value for 'interpolation_type' ", interpolation_type);
        }
    }
}

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, FinalSigmaType& param) {
    if (data.contains(name) && data[name].is_string()) {
        std::string final_sigma_type = data[name].get<std::string>();
        if (final_sigma_type == "zero")
            param = FinalSigmaType::ZERO;
        else if (final_sigma_type == "sigma_min")
            param = FinalSigmaType::SIGMA_MIN;
        else if (!final_sigma_type.empty()) {
            OPENVINO_THROW("Unsupported value for 'final_sigma_type' ", final_sigma_type);
        }
    }
}

template <>
void read_json_param(const nlohmann::json& data, const std::string& name, TimestepType& param) {
    if (data.contains(name) && data[name].is_string()) {
        std::string timestep_type = data[name].get<std::string>();
        if (timestep_type == "discrete")
            param = TimestepType::DISCRETE;
        else if (timestep_type == "continuous")
            param = TimestepType::CONTINUOUS;
        else if (!timestep_type.empty()) {
            OPENVINO_THROW("Unsupported value for 'timestep_type' ", timestep_type);
        }
    }
}

}  // namespace utils
}  // namespace genai
}  // namespace ov

std::ostream& operator<<(std::ostream& os, const ov::genai::Scheduler::Type& scheduler_type) {
    switch (scheduler_type) {
    case ov::genai::Scheduler::Type::LCM:
        return os << "LCMScheduler";
    case ov::genai::Scheduler::Type::DDIM:
        return os << "DDIMScheduler";
    case ov::genai::Scheduler::Type::EULER_DISCRETE:
        return os << "EulerDiscreteScheduler";
    case ov::genai::Scheduler::Type::EULER_ANCESTRAL_DISCRETE:
        return os << "EulerAncestralScheduler";
    case ov::genai::Scheduler::Type::FLOW_MATCH_EULER_DISCRETE:
        return os << "FlowMatchEulerDiscreteScheduler";
    case ov::genai::Scheduler::Type::AUTO:
        return os << "AutoScheduler";
    default:
        OPENVINO_THROW("Unsupported scheduler type value");
    }
}
