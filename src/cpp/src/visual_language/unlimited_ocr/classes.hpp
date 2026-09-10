// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vision_encoder.hpp"

namespace ov::genai {

// baidu/Unlimited-OCR (model_type == "unlimited-ocr"): a DeepEncoder vision branch
// (SAM ViT-B fused into CLIP-L followed by a linear projector) coupled with a
// DeepSeek-V2 MoE language model.
//
// The vision branch is exported as two static-shape submodels that already include
// the multimodal projector, so their output lives in the text-embedding space:
//   * openvino_vision_embeddings_model        -> global view  (1024x1024 -> 256 tokens)
//   * openvino_vision_embeddings_tiles_model  -> crop tiles   (640x640   -> 100 tokens)
//
// The per-image visual feature reproduces UnlimitedOCRModel.forward: every feature
// grid row is terminated by a learnable ``image_newline`` vector and each image is
// closed by a learnable ``view_separator`` vector. In crop mode the layout is
// ``cat([local_tiles, global, view_separator])`` and in the non-crop mode it is
// ``cat([global, view_separator])``. Every placeholder position uses the
// ``<image>`` token id (128815), so the text side reuses InputsEmbedderDeepseekOCR2.
class VisionEncoderUnlimitedOCR : public VisionEncoder {
public:
    VisionEncoderUnlimitedOCR(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap properties);

    VisionEncoderUnlimitedOCR(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap properties);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;

private:
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_vision_encoder_tiles;
    VLMConfig m_vlm_config;
};

}  // namespace ov::genai
