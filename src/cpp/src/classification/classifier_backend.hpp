// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "json_utils.hpp"
#include "nlohmann/json.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

struct ClassifierConfig {
    std::vector<std::string> architectures;
    std::optional<std::string> problem_type;

    ClassifierConfig(const std::filesystem::path& models_path) {
        const std::filesystem::path config_path = models_path / "config.json";
        std::ifstream config_file(config_path);
        OPENVINO_ASSERT(config_file.is_open(), "Failed to open ", config_path);
        const nlohmann::json config = nlohmann::json::parse(config_file);

        utils::read_json_param(config, "architectures", architectures);
        utils::read_json_param(config, "problem_type", problem_type);
    }
};

/// Inputs of one classified item. Each backend rejects the modalities its model does not accept.
struct ClassificationInputs {
    std::string text;
    std::vector<ov::Tensor> images;
};

/// Model-specific part of ClassificationPipeline: turns one item into the logits of the classification head.
class ClassifierBackend {
public:
    virtual ~ClassifierBackend() = default;

    virtual size_t num_labels() const = 0;

    /// Returns the logits [1, num_labels]; the tensor owns its data.
    virtual ov::Tensor infer(const ClassificationInputs& inputs) = 0;
};

}  // namespace ov::genai
