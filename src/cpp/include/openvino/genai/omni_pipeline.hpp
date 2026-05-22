// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/core/any.hpp"
#include "openvino/genai/chat_history.hpp"
#include "openvino/genai/omni_types.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/runtime/infer_request.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS OmniPipeline {
public:
    OmniPipeline(const std::filesystem::path& models_path,
                 const std::string& device,
                 const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    OmniPipeline(const std::filesystem::path& models_path, const std::string& device, Properties&&... properties)
        : OmniPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    ~OmniPipeline();

    OmniDecodedResults generate(const OmniInput& input,
                                const OmniGenerationConfig& generation_config,
                                const StreamerVariant& streamer = std::monostate(),
                                const AudioStreamerVariant& audio_streamer = std::monostate());

    OmniDecodedResults generate(const OmniInput& input, const ov::AnyMap& config_map);

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<OmniDecodedResults, Properties...> generate(const OmniInput& input,
                                                                               Properties&&... properties) {
        return generate(input, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    OmniDecodedResults generate(const ov::genai::ChatHistory& history,
                                const OmniInput& input,
                                const OmniGenerationConfig& generation_config,
                                const StreamerVariant& streamer = std::monostate(),
                                const AudioStreamerVariant& audio_streamer = std::monostate());

    OmniDecodedResults generate(const ov::genai::ChatHistory& history,
                                const OmniInput& input,
                                const ov::AnyMap& config_map);

    void start_chat(const std::string& system_message = {});
    void finish_chat();

    OmniGenerationConfig get_generation_config() const;
    void set_generation_config(const OmniGenerationConfig& config);

private:
    class OmniPipelineImpl;
    std::unique_ptr<OmniPipelineImpl> m_impl;
    ov::AnyMap build_pipeline_inputs(const OmniInput& input) const;
    OmniDecodedResults run_omni(ov::AnyMap run_inputs,
                                const StreamerVariant& streamer,
                                const AudioStreamerVariant& audio_streamer,
                                OmniOutputModality output_modality);
};

}  // namespace genai
}  // namespace ov
