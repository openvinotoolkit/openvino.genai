// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "text_embedding_pipeline.hpp"

#include "openvino/core/except.hpp"
#include "rag/text_embedding_utils.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

InferRequest create_text_embedding_npu_request(std::shared_ptr<ov::Model>& model,
                                               const TextEmbeddingPipeline::Config& config,
                                               const ov::AnyMap& properties,
                                               std::optional<size_t> max_position_embeddings,
                                               const bool is_seq_len_fixed) {
    if (config.batch_size.has_value() && is_seq_len_fixed) {
        utils::reshape_model(model, config, max_position_embeddings);
    }

    ov::CompiledModel compiled_model;
    if (model->is_dynamic()) {
        bool is_padding_on_left = config.padding_side.has_value() && config.padding_side.value() == "left";
        if (is_padding_on_left && is_seq_len_fixed && config.pooling_type != TextEmbeddingPipeline::PoolingType::MEAN) {
            OPENVINO_THROW("Padding on left is only supported for the MEAN pooling type for dynamic inputs models."
                           " In order to fix model shape, set batch_size, max_length and pad_to_max_length in the "
                           "configuration.");
        }

        auto kv_pos = utils::get_kv_axes_pos(model);
        utils::KVDesc kv_desc;
        std::tie(compiled_model, kv_desc) =
            utils::compile_decoder_for_npu_text_embedding(model, properties, kv_pos, config);
    } else {
        ov::Core core = utils::singleton_core();
        model = utils::apply_postprocessing(model, config);
        compiled_model = core.compile_model(model, "NPU", properties);
    }
    utils::print_compiled_model_properties(compiled_model, "npu text embedding model");
    return compiled_model.create_infer_request();
}

InferRequest create_text_embedding_npu_post_request(std::shared_ptr<ov::Model>& model,
                                                    const TextEmbeddingPipeline::Config& config) {
    if (model->is_dynamic()) {
        ov::Core core = utils::singleton_core();
        auto post_model = utils::create_post_model(model, config);
        auto post_compiled_model = core.compile_model(post_model, "CPU");
        return post_compiled_model.create_infer_request();
    } else {
        return InferRequest{};
    }
}

}  // namespace genai
}  // namespace ov
