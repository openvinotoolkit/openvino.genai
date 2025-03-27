// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

#include "json_utils.hpp"

#include "image_generation/schedulers/lcm.hpp"
#include "image_generation/schedulers/ddim.hpp"
#include "image_generation/schedulers/euler_discrete.hpp"
#include "image_generation/schedulers/flow_match_euler_discrete.hpp"
#include "image_generation/schedulers/pndm.hpp"
#include "image_generation/schedulers/euler_ancestral_discrete.hpp"

namespace ov {
namespace genai {

std::shared_ptr<Scheduler> Scheduler::from_config(const std::filesystem::path& scheduler_config_path, Type scheduler_type) {
    std::ifstream file(scheduler_config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", scheduler_config_path);

    if (scheduler_type == Scheduler::AUTO) {
        nlohmann::json data = nlohmann::json::parse(file);
        auto it = data.find("_class_name");
        OPENVINO_ASSERT(it != data.end(), "Failed to find '_class_name' field in ", scheduler_config_path);

        ov::genai::utils::read_json_param(data, "_class_name", scheduler_type);
        OPENVINO_ASSERT(scheduler_type != Scheduler::AUTO, "Failed to guess scheduler based on its config ", scheduler_config_path);
    }

    std::shared_ptr<Scheduler> scheduler = nullptr;
    if (scheduler_type == Scheduler::Type::LCM) {
        scheduler = std::make_shared<LCMScheduler>(scheduler_config_path);
    } else if (scheduler_type == Scheduler::Type::DDIM) {
        scheduler = std::make_shared<DDIMScheduler>(scheduler_config_path);
    } else if (scheduler_type == Scheduler::Type::EULER_DISCRETE) {
        scheduler = std::make_shared<EulerDiscreteScheduler>(scheduler_config_path);
    } else if (scheduler_type == Scheduler::Type::FLOW_MATCH_EULER_DISCRETE) {
        scheduler = std::make_shared<FlowMatchEulerDiscreteScheduler>(scheduler_config_path);
    } else if (scheduler_type == Scheduler::Type::PNDM) {
        scheduler = std::make_shared<PNDMScheduler>(scheduler_config_path);
    } else if (scheduler_type == Scheduler::Type::EULER_ANCESTRAL_DISCRETE) {
        scheduler = std::make_shared<EulerAncestralDiscreteScheduler>(scheduler_config_path);
    } else {
        OPENVINO_THROW("Unsupported scheduler type '", scheduler_type, ". Please, manually create scheduler via supported one");
    }

    return scheduler;
}

Scheduler::~Scheduler() = default;

} // namespace genai
} // namespace ov
