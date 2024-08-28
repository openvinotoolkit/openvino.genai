// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/vision_encoder.hpp"
#include <openvino/openvino.hpp>

namespace ov::genai {
struct PromptImage {
    std::string prompt;
    ov::Tensor image;
};

class OPENVINO_GENAI_EXPORTS VLMPipeline {
public:
    size_t query_num = 64;  // query_num throughput this impl - the number of <unk> to insert into the prompt per image slice.
    float scale_emb = 12.0f;  // multiply embeddings by it. Hardcoded throughout this impl
    bool slice_mode = true;  // Don't resize and slice an input image.
    ov::genai::Tokenizer tokenizer;
    VisionEncoder vision_encoder;
    ov::InferRequest resampler, ireq_embed, ireq;
    ov::Tensor imgEmbedTensor;
    ov::Shape img_embed_shape;
    size_t encoder_embed_dim;  // check that it's the same as embed_dim
    std::vector<float> llm_inputs_embeds;
    // input length, output length, first time, other time
    std::vector<std::tuple<size_t, size_t, double, double>> perf_records;
    size_t max_lenth = 2048;
    size_t embed_lenth = 0;
    int count = 0;
    double total_time = 0;
    const size_t BATCH_SIZE = 1;
    HeightWidth max_size{70, 70};
    size_t embed_dim = 2304;
    ov::Tensor _pos_embeds;

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
