// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/runtime/tensor.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/visual_language/video_metadata.hpp"
#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "openvino/genai/common_types.hpp"

namespace ov::genai {

class InputsEmbedder;

struct ProcessedInputs {
    ov::Tensor inputs_embeds;
    ov::Tensor attention_mask;
    ov::Tensor position_ids;
    std::unordered_map<std::string, ov::Tensor> lm_extra_inputs;
    std::optional<int64_t> rope_delta;
    VLMRawPerfMetrics raw_perf_metrics;
};

class OPENVINO_GENAI_EXPORTS VLMProcessor {
public:
    VLMProcessor(
        const std::filesystem::path& models_path,
        const Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap& properties = {}
    );

    VLMProcessor(
        const std::filesystem::path& models_path,
        const std::string& device,
        const ov::AnyMap& properties = {}
    );

    VLMProcessor(
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap& properties = {}
    );

    ~VLMProcessor();

    ProcessedInputs process(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images = {},
        const std::vector<ov::Tensor>& videos = {},
        const std::vector<VideoMetadata>& videos_metadata = {}
    );

    ProcessedInputs process(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images = {},
        const std::vector<ov::Tensor>& videos = {},
        const std::vector<VideoMetadata>& videos_metadata = {}
    );

    Tokenizer get_tokenizer() const;

private:
    class VLMProcessorImpl;
    std::unique_ptr<VLMProcessorImpl> m_pimpl;

    friend std::shared_ptr<InputsEmbedder> get_shared_inputs_embedder(const VLMProcessor& processor);
};

} // namespace ov::genai
