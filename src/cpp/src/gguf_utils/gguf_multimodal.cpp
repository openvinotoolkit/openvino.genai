// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_multimodal.hpp"

#include <cmath>
#include <sstream>

#include "gguf_modeling.hpp"
#include "gguf_tokenizer.hpp"
#include "openvino/frontend/gguf/adapt_mmproj_to_genai.hpp"
#include "openvino/frontend/gguf/adapt_to_genai.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"

namespace ov::genai {
GGUFMultimodalModels read_gguf_multimodal(const std::filesystem::path& language,
                                          const std::filesystem::path& mmproj,
                                          const ov::AnyMap& properties) {
    namespace gguf = ov::frontend::gguf;
    GGUFMultimodalModels result;
    result.language = convert_gguf_with_frontend(language.string());
    result.tokenizer = Tokenizer(GGUFTokenizerParameters(take_gguf_tokenizer_metadata(result.language)), properties);
    auto combined = convert_gguf_with_frontend(mmproj.string());
    OPENVINO_ASSERT(result.language->get_rt_info<std::string>("gguf_architecture") == "gemma3" &&
                        combined->get_rt_info<std::string>({"gguf_mmproj", "vision.projector"}) == "gemma3",
                    "GGUF VLMPipeline currently requires a Gemma3 language model and matching Gemma3 mmproj");

    // Read the processor metadata off the freshly converted mmproj before the adapter pass
    // rewrites the graph, so the model can then be adapted in place instead of cloned.
    const auto integer = [&](const std::string& key) {
        return std::stoull(combined->get_rt_info<std::string>({"gguf_mmproj", key}));
    };
    result.processor.size_height = result.processor.size_width = integer("clip.vision.image_size");
    result.processor.patch_size = integer("clip.vision.patch_size");
    result.config.vision_config_patch_size = result.processor.patch_size;
    const auto array = [&](const std::string& key, std::array<float, 3>& values, bool positive) {
        std::istringstream stream(combined->get_rt_info<std::string>({"gguf_mmproj", key}));
        for (size_t i = 0; i < values.size(); ++i) {
            std::string field;
            OPENVINO_ASSERT(std::getline(stream, field, ','), "Missing GGUF processor value ", key);
            values[i] = std::stof(field);
            OPENVINO_ASSERT(std::isfinite(values[i]) && (!positive || values[i] > 0),
                            "Invalid GGUF processor value ",
                            key);
        }
    };
    array("clip.vision.image_mean", result.processor.image_mean, /*positive=*/false);
    array("clip.vision.image_std", result.processor.image_std, /*positive=*/true);

    gguf::pass::AdaptToGenAI adapt(gguf::pass::AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS);
    adapt.run_on_model(result.language);
    result.text_embeddings = adapt.get_embedding_model();
    result.vision = std::move(combined);
    gguf::pass::AdaptMmprojToGenAI(gguf::pass::AdaptMmprojToGenAI::Modality::Vision).run_on_model(result.vision);
    const auto width = result.language->input("inputs_embeds").get_partial_shape()[2];
    OPENVINO_ASSERT(width == result.vision->output().get_partial_shape()[2],
                    "GGUF language and mmproj embedding widths do not match");
    // Gemma3 scales token lookups, but llama.cpp's embedding-input route leaves image
    // features unchanged. AdaptToGenAI retains the decoder's token scaling, so compensate
    // projected image features before combining them with raw token embeddings.
    auto output = result.vision->get_results().front();
    auto features = output->input_value(0);
    auto unscaled = std::make_shared<ov::op::v1::Multiply>(
        features,
        ov::op::v0::Constant::create(ov::element::f32, {}, {1.f / std::sqrt(float(width.get_length()))}));
    unscaled->output(0).set_names(features.get_names());
    output->input(0).replace_source_output(unscaled);
    result.vision->validate_nodes_and_infer_types();
    result.config.model_type = VLMModelType::GEMMA3;
    result.config.hidden_size = width.get_length();
    result.config.scale_emb = 1.f;
    // GGUF Gemma3 vocabularies do not contain HF's added <image_soft_token>.
    // Use the existing padding token only as an assembly placeholder; every occurrence
    // is replaced with a projected image vector before language-model inference.
    result.config.image_soft_token = "<pad>";
    const auto placeholder = result.tokenizer.encode(result.config.image_soft_token, add_special_tokens(false));
    OPENVINO_ASSERT(placeholder.input_ids.get_size() == 1,
                    "GGUF Gemma3 requires a single-token padding placeholder for image assembly");
    return result;
}
}  // namespace ov::genai
