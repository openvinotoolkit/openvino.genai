// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/vision_encoder.hpp"
#include "openvino/genai/vlm_config.hpp"

namespace ov::genai {
struct PromptImage {
    std::string prompt;
    ov::Tensor image;
};

class OPENVINO_GENAI_EXPORTS VLMPipeline {
public:
    VLMConfig vlm_config;
    Tokenizer tokenizer;
    VisionEncoder vision_encoder;
    ov::InferRequest resampler, ireq_embed, ireq;
    std::vector<float> language_embeddings_history;
    size_t history_length = 0;
    ov::Tensor pos_embed_cache;

    VLMPipeline(
        const ov::genai::Tokenizer& tokenizer,
        const VisionEncoder& vision_encoder,
        const ov::InferRequest& resampler,
        const ov::InferRequest& embedding,
        const ov::InferRequest& language_model
    );

    explicit VLMPipeline(const std::filesystem::path& model_dir, const std::string& device="CPU", const ov::AnyMap device_config={}, ov::Core core=ov::Core{}) :
        VLMPipeline{
            ov::genai::Tokenizer(model_dir.string(), device_config),
            VisionEncoder(model_dir, device, device_config, core),
            core.compile_model(
                model_dir / "openvino_resampler.xml", device, device_config
            ).create_infer_request(),
            core.compile_model(
                model_dir / "openvino_embedding.xml", device, device_config
            ).create_infer_request(),
            core.compile_model(
                model_dir / "openvino_model.xml", device, device_config
            ).create_infer_request()
        } {}
    std::string generate(const PromptImage& pi, const std::function<bool(std::string&&)>& callback);
    std::string generate(const PromptImage& pi, const std::shared_ptr<ov::genai::StreamerBase>& streamer=nullptr);
    void start_chat() {}
    void finish_chat() {}
    void set_2d_pos_cache(const HeightWidth& max_size);
    void adjust_pos_cache(const std::vector<HeightWidth>& target_sizes);
};
}
