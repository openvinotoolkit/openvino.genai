// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/deepseek_ocr2/classes.hpp"
#include "visual_language/gemma3/classes.hpp"
#include "visual_language/gemma3n/classes.hpp"
#include "visual_language/gemma4/classes.hpp"
#include "visual_language/internvl_chat/classes.hpp"
#include "visual_language/lfm2/classes.hpp"
#include "visual_language/llava/classes.hpp"
#include "visual_language/llava_next/classes.hpp"
#include "visual_language/llava_next_video/classes.hpp"
#include "visual_language/minicpm/classes.hpp"
#include "visual_language/minicpmv4_7/classes.hpp"
#include "visual_language/muse_glimmer/classes.hpp"
#include "visual_language/nanollava/classes.hpp"
#include "visual_language/phi3_vision/classes.hpp"
#include "visual_language/phi4mm/classes.hpp"
#include "visual_language/qwen2_5_vl/classes.hpp"
#include "visual_language/qwen2vl/classes.hpp"
#include "visual_language/qwen3_5/classes.hpp"
#include "visual_language/qwen3_omni/classes.hpp"
#include "visual_language/qwen3_vl/classes.hpp"
#include "visual_language/videochat_flash/classes.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai {

/// @brief A VLM family's classes, selected by VLMModelType.
template <typename Encoder, typename Embedder>
struct VLMFamily {
    using VisionEncoderType = Encoder;
    using InputsEmbedderType = Embedder;
};

/// @brief Placeholder for families without a vision encoder.
struct NoVisionEncoder {};

/// @brief Calls `visit(VLMFamily<...>{})` for the family of `type`. Add new VLM families here only.
template <typename Visitor>
decltype(auto) visit_vlm_family(VLMModelType type, Visitor&& visit) {
    switch (type) {
    case VLMModelType::LFM2_AUDIO:
        return visit(VLMFamily<NoVisionEncoder, InputsEmbedderLFM2Audio>{});
    case VLMModelType::MINICPM:
        return visit(VLMFamily<VisionEncoderMiniCPM, InputsEmbedderMiniCPM>{});
    case VLMModelType::MINICPMV4_7:
        return visit(VLMFamily<VisionEncoderMiniCPMv4_7, InputsEmbedderMiniCPMv4_7>{});
    case VLMModelType::LLAVA:
        return visit(VLMFamily<VisionEncoderLLaVA, InputsEmbedderLLaVA>{});
    case VLMModelType::NANOLLAVA:
        return visit(VLMFamily<VisionEncoderNanoLLaVA, InputsEmbedderNanoLLaVA>{});
    case VLMModelType::LLAVA_NEXT:
        return visit(VLMFamily<VisionEncoderLLaVANext, InputsEmbedderLLaVANext>{});
    case VLMModelType::LLAVA_NEXT_VIDEO:
        return visit(VLMFamily<VisionEncoderLLaVANextVideo, InputsEmbedderLLaVANextVideo>{});
    case VLMModelType::INTERNVL_CHAT:
        return visit(VLMFamily<VisionEncoderInternVLChat, InputsEmbedderInternVLChat>{});
    case VLMModelType::PHI3_V:
        return visit(VLMFamily<VisionEncoderPhi3V, InputsEmbedderPhi3V>{});
    case VLMModelType::PHI4MM:
        return visit(VLMFamily<VisionEncoderPhi4MM, InputsEmbedderPhi4MM>{});
    case VLMModelType::QWEN2_VL:
        return visit(VLMFamily<VisionEncoderQwen2VL, InputsEmbedderQwen2VL>{});
    case VLMModelType::QWEN2_5_VL:
        return visit(VLMFamily<VisionEncoderQwen2_5_VL, InputsEmbedderQwen2_5_VL>{});
    case VLMModelType::QWEN3_VL:
        return visit(VLMFamily<VisionEncoderQwen3VL, InputsEmbedderQwen3VL>{});
    case VLMModelType::QWEN3_5:
    case VLMModelType::QWEN3_5_MOE:
        return visit(VLMFamily<VisionEncoderQwen3_5, InputsEmbedderQwen3_5>{});
    case VLMModelType::QWEN3_OMNI:
        return visit(VLMFamily<VisionEncoderQwen3Omni, InputsEmbedderQwen3Omni>{});
    case VLMModelType::GEMMA3:
        return visit(VLMFamily<VisionEncoderGemma3, InputsEmbedderGemma3>{});
    case VLMModelType::GEMMA3N:
        return visit(VLMFamily<VisionEncoderGemma3n, InputsEmbedderGemma3n>{});
    case VLMModelType::LFM2_VL:
        return visit(VLMFamily<VisionEncoderLFM2, InputsEmbedderLFM2Vision>{});
    case VLMModelType::GEMMA4:
    case VLMModelType::GEMMA4_UNIFIED:
        return visit(VLMFamily<VisionEncoderGemma4, InputsEmbedderGemma4>{});
    case VLMModelType::VIDEOCHAT_FLASH_QWEN:
        return visit(VLMFamily<VisionEncoderVideoChatFlashQwen, InputsEmbedderVideoChatFlashQwen>{});
    case VLMModelType::DEEPSEEK_OCR2:
        return visit(VLMFamily<VisionEncoderDeepseekOCR2, InputsEmbedderDeepseekOCR2>{});
    case VLMModelType::MUSE_GLIMMER:
        return visit(VLMFamily<VisionEncoderMuseGlimmer, InputsEmbedderMuseGlimmer>{});
    default:
        OPENVINO_THROW("Unsupported model type in VLM pipeline. Please, create feature request on new model support");
    }
}

}  // namespace ov::genai
