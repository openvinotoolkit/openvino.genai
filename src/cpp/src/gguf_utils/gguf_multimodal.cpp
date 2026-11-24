// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_multimodal.hpp"

#include <algorithm>
#include <charconv>
#include <nlohmann/json.hpp>
#include <sstream>

#include "gguf_modeling.hpp"
#include "gguf_tokenizer.hpp"
#include "gguf_utils.hpp"
#include "openvino/frontend/gguf/adapt_to_genai.hpp"
#include "openvino/frontend/gguf/extension/genai.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/genai/extensions.hpp"
#include "openvino/genai/gguf.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/select.hpp"
#include "utils.hpp"

namespace ov::genai {
size_t add_gguf_image_token(ov::AnyMap& metadata, const std::string& token) {
    auto& tokens = metadata.at("tokens").as<std::vector<std::string>>();
    auto types = metadata.at("token_type").as<ov::Tensor>();
    OPENVINO_ASSERT(types.get_element_type() == ov::element::i32 && types.get_size() == tokens.size(),
                    "GGUF tokenizer token_type must contain one int32 per token");
    const auto existing = std::find(tokens.begin(), tokens.end(), token);
    if (existing != tokens.end()) {
        const size_t id = existing - tokens.begin();
        const int32_t type = types.data<const int32_t>()[id];
        OPENVINO_ASSERT(type == 3 || type == 4, "GGUF image token must be a control or user-defined token: ", token);
        return id;
    }

    const size_t id = tokens.size();
    ov::Tensor new_types(ov::element::i32, {id + 1});
    std::copy_n(types.data<const int32_t>(), id, new_types.data<int32_t>());
    new_types.data<int32_t>()[id] = 3;
    metadata["token_type"] = new_types;
    if (const auto it = metadata.find("scores"); it != metadata.end()) {
        const auto scores = it->second.as<ov::Tensor>();
        OPENVINO_ASSERT(scores.get_element_type() == ov::element::f32 && scores.get_size() == id,
                        "GGUF tokenizer scores must contain one float per token");
        ov::Tensor new_scores(ov::element::f32, {id + 1});
        std::copy_n(scores.data<const float>(), id, new_scores.data<float>());
        new_scores.data<float>()[id] = 0.f;
        metadata["scores"] = new_scores;
    }
    tokens.push_back(token);
    return id;
}

void map_gguf_image_token_for_embeddings(const std::shared_ptr<ov::Model>& model, size_t token_id) {
    const auto input = model->input(0);
    const auto type = input.get_element_type();
    OPENVINO_ASSERT(type == ov::element::i64 || type == ov::element::i32,
                    "GGUF text embeddings require int32 or int64 token IDs");
    const auto consumers = input.get_target_inputs();
    const auto marker = ov::op::v0::Constant::create(type, {}, {token_id});
    const auto replacement = ov::op::v0::Constant::create(type, {}, {0});
    const auto selected = std::make_shared<ov::op::v1::Select>(
        std::make_shared<ov::op::v1::Equal>(input, marker), replacement, input);
    for (auto consumer : consumers)
        consumer.replace_source_output(selected);
    model->validate_nodes_and_infer_types();
}

namespace {
using nlohmann::json;

void register_vlm_gguf_extensions(ov::frontend::gguf::FrontEnd& frontend, const ExtensionList& extensions) {
    const auto add_extension = [&](const std::shared_ptr<ov::Extension>& object) {
        // Adaptation is applied after conversion, not through GenAIExtension.
        if (!std::dynamic_pointer_cast<ov::frontend::gguf::GenAIExtension>(object))
            frontend.add_extension(object);
    };

    for (const auto& extension : extensions) {
        if (const auto* path = std::get_if<std::filesystem::path>(&extension)) {
            auto library = load_shared_object(*path);
            using CreateExtensions = void (*)(std::vector<std::shared_ptr<ov::Extension>>&);
            std::vector<std::shared_ptr<ov::Extension>> objects;
            reinterpret_cast<CreateExtensions>(get_symbol(library, "create_extensions"))(objects);
            for (const auto& object : objects) {
                // Keep the library loaded for the extension's lifetime.
                auto owner =
                    std::make_shared<std::pair<std::shared_ptr<void>, std::shared_ptr<ov::Extension>>>(library, object);
                add_extension(std::shared_ptr<ov::Extension>(owner, object.get()));
            }
        } else {
            add_extension(std::get<std::shared_ptr<ov::Extension>>(extension));
        }
    }
}

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
        c.image["max_soft_tokens"] = 1120;
        c.unsampled_video_from_image();
        return c;
    }
    const auto merge = meta.integer("vision.merge");
    c.image["merge_size"] = merge;
    if (architecture == "lfm3") {
        c.config["model_type"] = "lfm3_vl";
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
        // GGUF collapses HF's two-frame patch kernel.
        c.image["temporal_patch_size"] = 1;
        c.image["max_image_tokens"] = 4096;
        c.unsampled_video_from_image();
        c.video["max_video_frame_tokens"] = 4096;
        return c;
    }
    c.config["model_type"] = architecture == "qwen3vl" || architecture == "qwen3vlmoe" ? "qwen3_vl"
                            : architecture == "qwen35moe"                              ? "qwen3_5_moe"
                                                                                       : "qwen3_5";
    if (architecture == "qwen3vl" || architecture == "qwen3vlmoe") {
        const auto shape = models.vlm.at("vision_embeddings_merger")
                               ->output("deepstack_feature_lists")
                               .get_partial_shape();
        OPENVINO_ASSERT(shape.rank().is_static() && shape.rank().get_length() == 3 && shape[0].is_static() &&
                            shape[0].get_length() > 0,
                        "Qwen3-VL requires a static positive DeepStack layer count");
        c.config["vision_config"]["deepstack_visual_indexes"] =
            std::vector<size_t>(static_cast<size_t>(shape[0].get_length()), 0);
    }
    const auto side = meta.integer("clip.vision.image_size") / patch;
    c.config["vision_config"]["num_position_embeddings"] = side * side;
    c.image["temporal_patch_size"] = 2;
    const auto merged_patch = patch * merge;
    c.image["min_pixels"] = 8 * merged_patch * merged_patch;
    c.image["max_pixels"] = 4096 * merged_patch * merged_patch;
    c.unsampled_video_from_image();
    return c;
}
}  // namespace

void validate_gguf_clip_language(const std::shared_ptr<ov::Model>& model) {
    const auto positions = model->input("position_ids");
    OPENVINO_ASSERT(positions.get_element_type() == ov::element::i64 && positions.get_partial_shape().rank() == 2,
                    "GGUF LLaVA and InternVL require rank-2 int64 language position_ids [batch, tokens]");
    for (const auto& input : model->inputs()) {
        const auto& names = input.get_names();
        OPENVINO_ASSERT(names.count("inputs_embeds") || names.count("attention_mask") || names.count("position_ids") ||
                            names.count("beam_idx"),
                        "GGUF LLaVA and InternVL do not support additional language input ",
                        input.get_any_name());
    }
}

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
    gguf::FrontEnd language_frontend;
    register_vlm_gguf_extensions(language_frontend, extensions);
    const auto language_path = language.string();
    auto language_models = convert_gguf_with_frontend(language_frontend,
                                                      language_path,
                                                      {gguf::pass::AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS});
    result.language = get_gguf_language_model(language_models, language_path);
    result.language->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    auto tokenizer_metadata = take_gguf_tokenizer_metadata(result.language);
    result.stop_token_ids = gguf_stop_token_ids(tokenizer_metadata);
    gguf::FrontEnd projector_frontend;
    register_vlm_gguf_extensions(projector_frontend, extensions);
    auto combined = projector_frontend.convert(projector_frontend.load(mmproj));
    const MmprojMetadata meta(combined);
    auto encoders = gguf::adapt_to_genai(combined);
    combined.reset();
    const auto architecture = result.language->get_rt_info<std::string>("gguf_architecture");
    const auto projector = meta.text("vision.projector");
    std::optional<size_t> image_token_id;
    if (projector == "mlp") {
        if (const auto it = tokenizer_metadata.find("add_eos_token"); it != tokenizer_metadata.end()) {
            const auto flag = it->second.as<ov::Tensor>();
            OPENVINO_ASSERT(flag.get_element_type() == ov::element::boolean && flag.get_size() == 1,
                            "GGUF LLaVA add_eos_token must be a single boolean value");
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
            const auto image_padding_id = unknown.data<const uint32_t>()[0];
            OPENVINO_ASSERT(image_padding_id < tokenizer_metadata.at("tokens").as<std::vector<std::string>>().size(),
                            "GGUF LLaVA unknown_token_id is outside the vocabulary");
        } else {
            const auto token_types = tokenizer_metadata.at("token_type").as<ov::Tensor>();
            const auto end = token_types.data<const int32_t>() + token_types.get_size();
            const auto unknown = std::find(token_types.data<const int32_t>(), end, 2);
            OPENVINO_ASSERT(unknown != end, "GGUF LLaVA requires an unknown token for image padding");
        }
        image_token_id = add_gguf_image_token(tokenizer_metadata, "<image>");
    }
    if (architecture == "gemma3" && projector == "gemma3")
        image_token_id = add_gguf_image_token(tokenizer_metadata, "<image_soft_token>");
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
                        ((architecture == "qwen3vl" || architecture == "qwen3vlmoe") &&
                         projector == "qwen3vl_merger" && !meta.has("audio.projector")) ||
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
    if (image_token_id)
        map_gguf_image_token_for_embeddings(vlm.models["text_embeddings"], *image_token_id);
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
    if (vlm.find("audio_embeddings")) {
        const auto token = result.tokenizer.encode(vlm.config.audio_token, add_special_tokens(false)).input_ids;
        OPENVINO_ASSERT(token.get_size() == 1, "GGUF Gemma4 requires a single audio placeholder token");
        vlm.config.audio_token_id = token.data<const int64_t>()[0];
    }
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
