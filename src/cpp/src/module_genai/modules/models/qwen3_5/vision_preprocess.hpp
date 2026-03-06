// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "module_genai/modules/md_vision_preprocess/vision_preprocess.hpp"
#include "module_genai/utils/vision_preprocess.hpp"
#include "module_genai/modules/models/qwen3_5/qwen3_5preprocessor.hpp"

namespace ov::genai::module {

class Qwen3_5VisionPreprocess final : public VisionPreprocess {
public:
    Qwen3_5VisionPreprocess() = delete;
    Qwen3_5VisionPreprocess(const std::filesystem::path& model_path, VLMModelType model_type);

    PreprocessOutput preprocess(const std::vector<ov::Tensor>& images, const std::vector<ov::Tensor>& videos) override;

    // void result_to_output(std::map<std::string, OutputModule>& output) const override;

private:
    std::shared_ptr<Qwen3_5Preprocessor> m_preprocessor;
};

}  // namespace ov::genai::module
