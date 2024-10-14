// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/text2image/pipeline.hpp"

#include <fstream>

#include "utils.hpp"

#include "text2image/schedulers/lcm.hpp"
#include "text2image/schedulers/lms_discrete.hpp"
#include "text2image/schedulers/ddim.hpp"
#include "text2image/schedulers/euler_discrete.hpp"

namespace ov {
namespace genai {

std::shared_ptr<Text2ImagePipeline::Scheduler> Text2ImagePipeline::Scheduler::from_config(const std::string& scheduler_config_path, Type scheduler_type) {
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
        // TODO: do we need to pass RNG generator somehow to LCM?
        scheduler = std::make_shared<LCMScheduler>(scheduler_config_path);
    } else if (scheduler_type == Scheduler::Type::LMS_DISCRETE) {
        scheduler = std::make_shared<LMSDiscreteScheduler>(scheduler_config_path);
    } else if (scheduler_type == Scheduler::Type::DDIM) {
        scheduler = std::make_shared<DDIMScheduler>(scheduler_config_path);
    } else if (scheduler_type == Scheduler::Type::EULER_DISCRETE) {
        scheduler = std::make_shared<EulerDiscreteScheduler>(scheduler_config_path);
    } else {
        OPENVINO_THROW("Unsupported scheduler type '", scheduler_type, ". Please, manually create scheduler via supported one");
    }

    return scheduler;
}

Text2ImagePipeline::Scheduler::~Scheduler() = default;

} // namespace genai
} // namespace ov
