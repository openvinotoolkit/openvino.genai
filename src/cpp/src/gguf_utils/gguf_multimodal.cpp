// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_multimodal.hpp"

#include <algorithm>
#include <nlohmann/json.hpp>
#include <sstream>

#include "gguf_clip_adapter.hpp"
#include "gguf_frontend_extensions.hpp"
#include "gguf_modeling.hpp"
#include "gguf_tokenizer.hpp"
#include "gguf_utils.hpp"
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
    bool has(const std::string& key) const {
        return m_metadata.count(key) != 0;
    }
    std::string text(const std::string& key) const {
        const auto it = m_metadata.find(key);
        OPENVINO_ASSERT(it != m_metadata.end(), "GGUF mmproj metadata has no ", key);
        return it->second.as<std::string>();
    }
    size_t integer(const std::string& key) const {
        return std::stoull(text(key));
    }
    size_t integer(const std::string& key, size_t fallback) const {
        return has(key) ? integer(key) : fallback;
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

    // Videos use the image settings, with the frames passed through unsampled.
    void unsampled_video_from_image() {
        video = image;
        video["do_sample_frames"] = false;
    }
};

Configs gguf_configs(const std::string& architecture,
                     const std::string& projector,
                     const MmprojMetadata& meta,
                     const GGUFMultimodalModels& models) {
    Configs c;
    c.config["hidden_size"] = models.language->input("inputs_embeds").get_partial_shape()[2].get_length();
    const auto patch = meta.integer("clip.vision.patch_size");
    c.config["vision_config"]["patch_size"] = patch;
    c.image["patch_size"] = patch;
    c.image["image_mean"] = meta.floats("clip.vision.image_mean");
    c.image["image_std"] = meta.floats("clip.vision.image_std");
    if (projector == "mlp" || projector == "internvl") {
        const auto size = meta.integer("clip.vision.image_size");
        OPENVINO_ASSERT(size > 0 && patch > 0 && size % patch == 0,
                        "GGUF LLaVA and InternVL image size must be a positive patch multiple");
        c.config["model_type"] = projector == "mlp" ? "llava" : "internvl_chat";
        c.image["crop_size"] = {{"height", size}, {"width", size}};
        c.image["size"]["shortest_edge"] = size;
        c.video = c.image;
        return c;
    }
    if (architecture == "gemma3") {
        const auto size = meta.integer("clip.vision.image_size");
        c.config["model_type"] = "gemma3";
        c.config["position_ids_offset"] = 0;
        c.config["image_separator"] = "";
        c.image["pad_to_target"] = true;
        c.image["size"] = {{"height", size}, {"width", size}};
        c.video = c.image;
        return c;
    }
    if (architecture == "gemma4") {
        const bool unified = projector == "gemma4uv";
        c.config["model_type"] = unified ? "gemma4_unified" : "gemma4";
        if (const auto per_layer = models.vlm.find("text_embeddings_per_layer"))
            c.config["text_config"]["hidden_size_per_layer_input"] =
                per_layer->output().get_partial_shape()[3].get_length();
        const auto& inputs = models.language->inputs();
        if (std::any_of(inputs.begin(), inputs.end(), [](const ov::Output<ov::Node>& input) {
                return input.get_names().count("token_type_ids") > 0;
            }))
            c.config["text_config"]["use_bidirectional_attention"] = "vision";
        c.image["pooling_kernel_size"] = meta.integer("clip.vision.projector.scale_factor", 3);
        const auto merged_patch = patch * c.image["pooling_kernel_size"].get<size_t>();
        c.image["preserve_native_resolution"] = true;
        c.image["pad_to_target"] = true;
        c.image["min_pixels"] = 70 * merged_patch * merged_patch;
        c.image["max_pixels"] = 1120 * merged_patch * merged_patch;
        c.image["max_soft_tokens"] = 1120;
        c.unsampled_video_from_image();
        return c;
    }
    const auto merge = meta.integer("vision.merge");
    c.image["merge_size"] = merge;
    if (architecture == "lfm3") {
        c.config["model_type"] = "lfm3_vl";
        c.config["position_ids_offset"] = 0;
        c.image["temporal_patch_size"] = meta.integer("vision.temporal_patch_size");
        const auto min_pixels = meta.integer("clip.vision.image_min_pixels", 65536);
        const auto max_pixels = meta.integer("clip.vision.image_max_pixels", 16777216);
        OPENVINO_ASSERT(min_pixels > 0 && max_pixels >= min_pixels,
                        "GGUF LFM3 image pixel limits must be positive and ordered");
        c.image["min_pixels"] = min_pixels;
        c.image["max_pixels"] = max_pixels;
        c.unsampled_video_from_image();
        c.video["min_pixels"] = 4096;
        c.video["max_pixels"] = 25165824;
        return c;
    }
    if (architecture == "muse-glimmer") {
        c.config["model_type"] = "muse_glimmer";
        c.image["preserve_native_resolution"] = true;
        // GGUF collapses HF's two-frame patch kernel.
        c.image["temporal_patch_size"] = 1;
        c.image["max_image_tokens"] = 4096;
        c.unsampled_video_from_image();
        c.video["max_video_frame_tokens"] = 4096;
        return c;
    }
    c.config["model_type"] = architecture == "qwen35moe" ? "qwen3_5_moe" : "qwen3_5";
    const auto side = meta.integer("clip.vision.image_size") / patch;
    c.config["vision_config"]["num_position_embeddings"] = side * side;
    c.image["temporal_patch_size"] = 2;
    const auto merged_patch = patch * merge;
    c.image["preserve_native_resolution"] = true;
    c.image["pad_to_target"] = true;
    c.image["min_pixels"] = 8 * merged_patch * merged_patch;
    c.image["max_pixels"] = 4096 * merged_patch * merged_patch;
    c.unsampled_video_from_image();
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
    if (const auto audio = models.find("audio_embeddings"))
        check_width(audio->output(0), "audio_embeddings");

    const auto& inputs = language->inputs();
    const auto per_layer_input = std::find_if(inputs.begin(), inputs.end(), [](const ov::Output<ov::Node>& input) {
        return input.get_names().count("per_layer_inputs") > 0;
    });
    const auto per_layer = models.find("text_embeddings_per_layer");
    OPENVINO_ASSERT(bool(per_layer) == (per_layer_input != inputs.end()),
                    "GGUF per-layer embeddings must match language per_layer_inputs presence");
    if (per_layer) {
        const auto& expected = per_layer_input->get_partial_shape();
        const auto& actual = per_layer->output(0).get_partial_shape();
        OPENVINO_ASSERT(expected.rank().is_static() && expected.rank().get_length() == 4 && actual.rank().is_static() &&
                            actual.rank().get_length() == 4,
                        "GGUF per-layer embeddings require rank 4 [batch, tokens, layers, width]");
        for (size_t axis : {2, 3})
            OPENVINO_ASSERT(expected[axis].is_static() && expected[axis].get_length() > 0 && actual[axis].is_static() &&
                                actual[axis] == expected[axis],
                            "GGUF text_embeddings_per_layer shape ",
                            actual,
                            " does not match language per_layer_inputs shape ",
                            expected);
    }
}

GGUFMultimodalModels read_gguf_multimodal(const std::filesystem::path& language,
                                          const std::string& device,
                                          ov::AnyMap& properties) {
    OPENVINO_ASSERT(device == "CPU", "GGUF multimodal generation is currently qualified on CPU only");
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
    const auto extensions = utils::extract_extensions(properties);
    for (const auto& extension : extensions) {
        std::visit(
            [](const auto& value) {
                utils::singleton_core().add_extension(value);
            },
            extension);
    }
    auto language_models = convert_gguf_with_frontend(language.string(),
                                                      extensions,
                                                      {gguf::pass::AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS});
    result.language = language_models.at("language");
    result.language->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    auto tokenizer_metadata = take_gguf_tokenizer_metadata(result.language);
    result.stop_token_ids = gguf_stop_token_ids(tokenizer_metadata);
    gguf::FrontEnd projector_frontend;
    register_gguf_extensions(projector_frontend, extensions);
    auto combined = projector_frontend.convert(projector_frontend.load(mmproj));
    const MmprojMetadata meta(combined);
    auto encoders = gguf::adapt_to_genai(combined);
    combined.reset();
    const auto architecture = result.language->get_rt_info<std::string>("gguf_architecture");
    const auto projector = meta.text("vision.projector");
    int64_t image_padding_id = -1;
    bool add_eos_token = false;
    if (projector == "mlp") {
        if (const auto it = tokenizer_metadata.find("add_eos_token"); it != tokenizer_metadata.end()) {
            const auto flag = it->second.as<ov::Tensor>();
            OPENVINO_ASSERT(flag.get_element_type() == ov::element::boolean && flag.get_size() == 1,
                            "GGUF LLaVA add_eos_token must be a single boolean value");
            add_eos_token = flag.data<const bool>()[0];
        }
        const auto aspect_ratio =
            meta.has("clip.vision.image_aspect_ratio") ? meta.text("clip.vision.image_aspect_ratio") : "square";
        OPENVINO_ASSERT(!meta.has("clip.vision.image_grid_pinpoints") &&
                            !meta.has("clip.vision.image_crop_resolution") &&
                            !meta.has("clip.vision.image_split_resolution") &&
                            (!meta.has("clip.vision.mm_patch_merge_type") ||
                             meta.text("clip.vision.mm_patch_merge_type") == "flat") &&
                            aspect_ratio == "square",
                        "Unsupported GGUF LLaVA variant: only fixed-resolution LLaVA 1.5 is supported; "
                        "LLaVA Next, any-resolution image inputs and image padding are not supported");
        if (const auto it = tokenizer_metadata.find("unknown_token_id"); it != tokenizer_metadata.end()) {
            const auto unknown = it->second.as<ov::Tensor>();
            OPENVINO_ASSERT(unknown.get_element_type() == ov::element::u32 && unknown.get_size() == 1,
                            "GGUF LLaVA unknown_token_id must be a single uint32 value");
            image_padding_id = unknown.data<const uint32_t>()[0];
            OPENVINO_ASSERT(image_padding_id < tokenizer_metadata.at("tokens").as<std::vector<std::string>>().size(),
                            "GGUF LLaVA unknown_token_id is outside the vocabulary");
        } else {
            const auto token_types = tokenizer_metadata.at("token_type").as<ov::Tensor>();
            const auto end = token_types.data<const int32_t>() + token_types.get_size();
            const auto unknown = std::find(token_types.data<const int32_t>(), end, 2);
            OPENVINO_ASSERT(unknown != end, "GGUF LLaVA requires an unknown token for image padding");
            image_padding_id = unknown - token_types.data<const int32_t>();
        }
    }
    if (projector == "internvl" && !tokenizer_metadata.count("eos_token_id")) {
        OPENVINO_ASSERT(!result.stop_token_ids.empty(), "GGUF InternVL requires an EOS or end-of-generation token");
        ov::Tensor eos(ov::element::u32, {});
        eos.data<uint32_t>()[0] = static_cast<uint32_t>(*result.stop_token_ids.begin());
        tokenizer_metadata["eos_token_id"] = eos;
    }
    result.tokenizer = GGUFTokenizer(GGUFTokenizerParameters(std::move(tokenizer_metadata)), properties);
    const bool clip = projector == "mlp" || projector == "internvl";
    if (clip)
        validate_gguf_clip_language(result.language);
    const bool gemma4 = architecture == "gemma4" && (projector == "gemma4v" || projector == "gemma4uv");
    OPENVINO_ASSERT((architecture == "gemma3" && projector == "gemma3") ||
                        ((architecture == "qwen35" || architecture == "qwen35moe") && projector == "qwen3vl_merger") ||
                        (architecture == "lfm3" && projector == "lfm3vl" && !meta.has("audio.projector")) ||
                        (architecture == "muse-glimmer" && projector == "muse-glimmer") || gemma4 || clip,
                    "Unsupported GGUF language/projector pair: ",
                    architecture,
                    " / ",
                    projector);

    auto& vlm = result.vlm;
    for (auto& [name, model] : encoders) {
        if (name.rfind("vision_embeddings", 0) == 0)
            vlm.models[name] = std::move(model);
    }
    if (gemma4 && meta.has("audio.projector")) {
        OPENVINO_ASSERT(meta.text("audio.projector") == "gemma4ua" || (meta.text("audio.projector") == "gemma4a" &&
                                                                       meta.integer("clip.audio.num_mel_bins") == 128),
                        "Unsupported GGUF Gemma4 audio configuration");
        vlm.models["audio_embeddings"] = encoders.at("audio_embeddings");
    }
    vlm.models["text_embeddings"] = language_models.at("text_embeddings");
    if (const auto it = language_models.find("text_embeddings_per_layer"); it != language_models.end())
        vlm.models["text_embeddings_per_layer"] = it->second;

    validate_gguf_multimodal_dimensions(result.language, vlm);
    const auto configs = gguf_configs(architecture, projector, meta, result);
    vlm.config = VLMConfig(configs.config);
    if (architecture == "lfm3") {
        const auto image = result.tokenizer.encode(vlm.config.image_token, add_special_tokens(false)).input_ids;
        const auto video = result.tokenizer.encode(vlm.config.video_token, add_special_tokens(false)).input_ids;
        OPENVINO_ASSERT(image.get_size() == 1 && video.get_size() == 1,
                        "GGUF LFM3 requires single-token image and video placeholders");
        vlm.config.image_token_id = image.data<const int64_t>()[0];
        vlm.config.video_token_id = video.data<const int64_t>()[0];
    }
    if (projector == "mlp") {
        OPENVINO_ASSERT(image_padding_id >= 0, "GGUF LLaVA requires a vocabulary padding token");
        vlm.config.image_token_id = image_padding_id;
        vlm.config.add_eos_token = add_eos_token;
    }
    if (vlm.find("audio_embeddings")) {
        const auto token = result.tokenizer.encode(vlm.config.audio_token, add_special_tokens(false)).input_ids;
        OPENVINO_ASSERT(token.get_size() == 1, "GGUF Gemma4 requires a single audio placeholder token");
        vlm.config.audio_token_id = token.data<const int64_t>()[0];
    }
    vlm.processor_config = ProcessorConfig(configs.image);
    vlm.video_processor_config = VideoProcessorConfig(configs.video);
    if (architecture == "gemma3") {
        // GGUF Gemma3 vocabularies lack <image_soft_token>; <pad> only marks image positions.
        vlm.config.image_soft_token = "<pad>";
        const auto placeholder = result.tokenizer.encode(vlm.config.image_soft_token, add_special_tokens(false));
        OPENVINO_ASSERT(placeholder.input_ids.get_size() == 1,
                        "GGUF Gemma3 requires a single-token padding placeholder for image assembly");
    }
    if (enable_save_ov_model) {
        utils::save_openvino_model(result.language, language.string() + ".language.xml", false);
        for (const auto& [name, model] : vlm.models)
            utils::save_openvino_model(model, mmproj + "." + name + ".xml", false);
    }
    return result;
}
}  // namespace ov::genai
