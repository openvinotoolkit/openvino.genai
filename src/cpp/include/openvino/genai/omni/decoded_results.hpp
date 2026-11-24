// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/omni/talker.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"

namespace ov::genai {

/**
 * @brief Result of OmniPipeline::generate. The speech output is the inherited
 * VLMDecodedResults::speech_result; `speech_result.waveforms` is empty when speech was not requested.
 *
 * @note This is a preview API and is subject to change.
 */
class OPENVINO_GENAI_EXPORTS OmniDecodedResults : public VLMDecodedResults {};

}  // namespace ov::genai
