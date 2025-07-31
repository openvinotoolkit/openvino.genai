// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <limits>

#include "llm/pipeline_base.hpp"

namespace ov::genai {

class StatefulLLMPipelineNPU final : public LLMPipelineImplBase {
public:
    StatefulLLMPipelineNPU(
        const std::filesystem::path& models_path,
        const ov::genai::Tokenizer& tokenizer,
        const ov::AnyMap& plugin_config
    );

    StatefulLLMPipelineNPU(
        const std::filesystem::path& models_path,
        const ov::AnyMap& plugin_config
    );

    StatefulLLMPipelineNPU(
        const std::shared_ptr<ov::Model>& model,
        const ov::genai::Tokenizer& tokenizer,
        const ov::AnyMap& config,
        const ov::genai::GenerationConfig& generation_config
    );

    DecodedResults generate(
        StringInputs inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) override;

    EncodedResults generate(
        const EncodedInputs& inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) override;

    void start_chat(const std::string& system_message) override;

    void finish_chat() override;

    ~StatefulLLMPipelineNPU() = default;

private:
    std::unique_ptr<LLMPipelineImplBase> m_pimpl;
};

} // namespace ov::genai
