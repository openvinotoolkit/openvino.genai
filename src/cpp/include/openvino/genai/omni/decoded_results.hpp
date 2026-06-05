// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/visual_language/pipeline.hpp"

namespace ov::genai {

/**
 * @brief Omni-specific decoded results including speech outputs.
 *
 * Extends VLMDecodedResults with speech output waveforms for Qwen3-Omni models.
 */
class OPENVINO_GENAI_EXPORTS OmniDecodedResults : public VLMDecodedResults {
public:
    /// Speech output waveforms (one per generated result).
    /// Empty if speech generation was not requested (return_audio=false).
    std::vector<ov::Tensor> speech_outputs;
};

}  // namespace ov::genai
