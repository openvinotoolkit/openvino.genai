// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ov::genai {

struct ModelDesc {
    std::string model_path;
    std::string device;
    ov::genai::SchedulerConfig scheduler_config;
    ov::AnyMap plugin_config;

    ModelDesc(const std::string& model_path,
              const std::string& device = "",
              const ov::AnyMap& plugin_config = {},
              const ov::genai::SchedulerConfig& scheduler_config = {}) :
        model_path(model_path),
        device(device),
        plugin_config(plugin_config),
        scheduler_config(scheduler_config) {}
};

} 