// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_multimodal.hpp"

#include <charconv>
#include <nlohmann/json.hpp>
#include <sstream>

#include "gguf_modeling.hpp"
#include "gguf_tokenizer.hpp"
#include "gguf_utils.hpp"
#include "openvino/frontend/gguf/adapt_to_genai.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/genai/gguf.hpp"
#include "utils.hpp"

namespace ov::genai {
namespace {
using nlohmann::json;

class MmprojMetadata {
public:
    explicit MmprojMetadata(const std::shared_ptr<ov::Model>& mmproj) {
        OPENVINO_ASSERT(mmproj->has_rt_info("gguf_mmproj"), "mmproj_path must point to a GGUF mmproj file");
        m_metadata = mmproj->get_rt_info<ov::AnyMap>("gguf_mmproj");
    }
    std::string text(const std::string& key) const {
        const auto it = m_metadata.find(key);
        OPENVINO_ASSERT(it != m_metadata.end(), "GGUF mmproj metadata has no ", key);
        return it->second.as<std::string>();
    }
    size_t integer(const std::string& key) const {
        const auto value_text = text(key);
        size_t value = 0;
        const auto [end, error] = std::from_chars(value_text.data(), value_text.data() + value_text.size(), value);
        OPENVINO_ASSERT(!value_text.empty() && error == std::errc{} && end == value_text.data() + value_text.size(),
                        "GGUF mmproj metadata ",
                        key,
                        " must be a non-negative integer: ",
                        value_text);
        return value;
    }
    std::vector<float> floats(const std::string& key) const {
        std::vector<float> values;
        std::istringstream stream(text(key));
        for (std::string field; std::getline(stream, field, ',');)
            values.push_back(std::stof(field));
        return values;
    }

private:
    ov::AnyMap m_metadata;
};

struct Configs {
    json config, image, video;
};

Configs gguf_configs(const std::string& architecture, const MmprojMetadata& meta, const GGUFMultimodalModels& models) {
    Configs c;
    c.config["model_type"] = architecture == "qwen35moe" ? "qwen3_5_moe" : "qwen3_5";
    c.config["hidden_size"] = models.language->input("inputs_embeds").get_partial_shape()[2].get_length();
    const auto patch = meta.integer("clip.vision.patch_size");
    const auto image_size = meta.integer("clip.vision.image_size");
    OPENVINO_ASSERT(patch > 0 && image_size >= patch, "GGUF mmproj image size must cover at least one patch");
    c.config["vision_config"]["patch_size"] = patch;
    const auto side = image_size / patch;
    c.config["vision_config"]["num_position_embeddings"] = side * side;
    const auto merge = meta.integer("vision.merge");
    c.image["patch_size"] = patch;
    c.image["image_mean"] = meta.floats("clip.vision.image_mean");
    c.image["image_std"] = meta.floats("clip.vision.image_std");
    c.image["merge_size"] = merge;
    c.image["temporal_patch_size"] = 2;
    const auto merged_patch = patch * merge;
    c.image["min_pixels"] = 8 * merged_patch * merged_patch;
    c.image["max_pixels"] = 4096 * merged_patch * merged_patch;
    // Videos use the image settings, with the frames passed through unsampled.
    c.video = c.image;
    c.video["do_sample_frames"] = false;
    return c;
}
}  // namespace

void validate_gguf_multimodal_dimensions(const std::shared_ptr<ov::Model>& language, const VLMModels& models) {
    const auto& inputs_shape = language->input("inputs_embeds").get_partial_shape();
    OPENVINO_ASSERT(inputs_shape.rank().is_static() && inputs_shape.rank().get_length() == 3 &&
                        inputs_shape[2].is_static() && inputs_shape[2].get_length() > 0,
                    "GGUF language inputs_embeds requires a static positive hidden size");
    const auto hidden_size = inputs_shape[2].get_length();
    const auto check_width = [&](const ov::Output<const ov::Node>& output, const std::string& role) {
        const auto& shape = output.get_partial_shape();
        OPENVINO_ASSERT(shape.rank().is_static() && shape.rank().get_length() >= 2,
                        "GGUF ",
                        role,
                        " requires an embedding output with static rank");
        const auto& width = shape[shape.rank().get_length() - 1];
        OPENVINO_ASSERT(width.is_static() && width.get_length() == hidden_size,
                        "GGUF ",
                        role,
                        " embedding width ",
                        width,
                        " does not match language inputs_embeds hidden size ",
                        hidden_size);
    };
    check_width(models.at("text_embeddings")->output(0), "text_embeddings");
    const auto merger = models.find("vision_embeddings_merger");
    const auto vision = merger ? merger : models.at("vision_embeddings");
    for (const auto& output : vision->outputs())
        check_width(output, merger ? "vision_embeddings_merger" : "vision_embeddings");
}

GGUFMultimodalModels read_gguf_multimodal(const std::filesystem::path& language,
                                          const std::string& device,
                                          ov::AnyMap& properties) {
    OPENVINO_ASSERT(properties.count(utils::DRAFT_MODEL_ARG_NAME) == 0,
                    "draft_model is not supported for GGUF VLMPipeline");
    const auto it = properties.find(mmproj_path.name());
    OPENVINO_ASSERT(it != properties.end(), "A language GGUF requires mmproj_path for VLMPipeline");
    const auto mmproj = it->second.as<std::string>();
    properties.erase(it);
    auto gguf_properties = gguf_utils::extract_gguf_properties(properties);
    OPENVINO_ASSERT(!gguf_properties.legacy_reader.value_or(false),
                    "GGUF multimodal generation requires GGUF_READER=FRONTEND");
    const auto enable_save_ov_model = gguf_properties.enable_save_ov_model;
    properties = std::move(gguf_properties.rest);
    utils::clear_false_prompt_lookup_from_config(properties);
    utils::validate_vlm_model_properties(properties);

    namespace gguf = ov::frontend::gguf;
    GGUFMultimodalModels result;
    for (const auto& extension : utils::extract_extensions(properties)) {
        std::visit(
            [](const auto& value) {
                utils::singleton_core().add_extension(value);
            },
            extension);
    }
    gguf::FrontEnd language_frontend;
    const auto language_path = language.string();
    auto language_models = convert_gguf_with_frontend(language_frontend,
                                                      language_path,
                                                      {gguf::pass::AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS});
    result.language = get_gguf_language_model(language_models, language_path);
    result.language->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    auto tokenizer_metadata = take_gguf_tokenizer_metadata(result.language);
    result.stop_token_ids = gguf_stop_token_ids(tokenizer_metadata);
    result.tokenizer = GGUFTokenizer(GGUFTokenizerParameters(std::move(tokenizer_metadata)), properties);

    gguf::FrontEnd projector_frontend;
    auto combined = projector_frontend.convert(projector_frontend.load(mmproj));
    const MmprojMetadata meta(combined);
    auto encoders = gguf::adapt_to_genai(combined);
    combined.reset();
    const auto architecture = result.language->get_rt_info<std::string>("gguf_architecture");
    const auto projector = meta.text("vision.projector");
    OPENVINO_ASSERT((architecture == "qwen35" || architecture == "qwen35moe") && projector == "qwen3vl_merger",
                    "Unsupported GGUF language/projector pair: ",
                    architecture,
                    " / ",
                    projector);

    auto& vlm = result.vlm;
    for (auto& [name, model] : encoders) {
        if (name.rfind("vision_embeddings", 0) == 0)
            vlm.models[name] = std::move(model);
    }
    vlm.models["text_embeddings"] = language_models.at("text_embeddings");

    validate_gguf_multimodal_dimensions(result.language, vlm);
    const auto configs = gguf_configs(architecture, meta, result);
    vlm.config = VLMConfig(configs.config);
    vlm.processor_config = ProcessorConfig(configs.image);
    vlm.video_processor_config = VideoProcessorConfig(configs.video);
    if (enable_save_ov_model) {
        utils::save_openvino_model(result.language, language.string() + ".language.xml", false);
        for (const auto& [name, model] : vlm.models)
            utils::save_openvino_model(model, mmproj + "." + name + ".xml", false);
    }
    return result;
}
}  // namespace ov::genai
