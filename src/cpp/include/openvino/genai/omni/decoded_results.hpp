// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/omni/talker.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"

namespace ov::genai {

/**
 * @brief Omni-specific decoded results including speech outputs.
 *
 * Extends VLMDecodedResults with a TalkerResults that holds speech waveforms and perf metrics.
 *
 * @note This is a preview API and is subject to change.
 */
class OPENVINO_GENAI_EXPORTS OmniDecodedResults : public VLMDecodedResults {
public:
    /// Talker-side result: speech waveforms + perf metrics.
    /// `speech_result.waveforms` is empty when speech generation was not requested.
    TalkerResults speech_result;
};

}  // namespace ov::genai
