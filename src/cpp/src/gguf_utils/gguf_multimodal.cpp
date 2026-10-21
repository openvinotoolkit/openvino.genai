// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_multimodal.hpp"

#include <algorithm>
#include <nlohmann/json.hpp>
#include <sstream>

#include "gguf_frontend_extensions.hpp"
#include "gguf_tokenizer.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/frontend/gguf/adapt_mmproj_to_genai.hpp"
#include "openvino/frontend/gguf/extension/genai.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/genai/gguf.hpp"
#include "openvino/opsets/opset15.hpp"
#include "utils.hpp"

namespace ov::genai {
namespace {
using nlohmann::json;

class MmprojMetadata {
public:
    explicit MmprojMetadata(const std::shared_ptr<ov::Model>& mmproj) : m_mmproj(mmproj) {}
    bool has(const std::string& key) const {
        return m_mmproj->has_rt_info({"gguf_mmproj", key});
    }
    std::string text(const std::string& key) const {
        return m_mmproj->get_rt_info<std::string>({"gguf_mmproj", key});
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
    std::shared_ptr<ov::Model> m_mmproj;
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

void adapt_gguf_audio_encoder(const std::shared_ptr<ov::Model>& model, bool unified) {
    using namespace ov::opset15;
    const auto axes = [](std::initializer_list<int64_t> values) {
        return Constant::create(ov::element::i64, {values.size()}, values);
    };
    const auto original = model->get_parameters().at(0);
    const auto features =
        std::make_shared<Parameter>(ov::element::f32,
                                    unified ? ov::PartialShape{1, -1, 640} : ov::PartialShape{1, -1, 128});
    features->output(0).set_names({"input_features"});
    ov::Output<ov::Node> input = features;
    if (!unified) {
        const auto mask = std::make_shared<Parameter>(ov::element::boolean, ov::PartialShape{1, -1});
        mask->output(0).set_names({"input_features_mask"});
        const auto length =
            std::make_shared<ReduceSum>(std::make_shared<Convert>(mask, ov::element::i64), axes({1}), false);
        input = std::make_shared<Slice>(input, axes({0}), length, axes({1}), axes({1}));
        input = std::make_shared<Transpose>(input, axes({0, 2, 1}));
        model->add_parameters({mask});
    }
    ov::replace_node(original, std::make_shared<Unsqueeze>(input, axes({1})));
    model->remove_parameter(original);
    model->add_parameters({features});
    auto output = model->get_results().at(0)->input_value(0);
    output.set_names({"last_hidden_state"});
    model->output().set_names({"last_hidden_state"});
    if (!unified) {
        const auto shape = std::make_shared<ShapeOf>(output);
        const auto dimensions = std::make_shared<Gather>(shape, axes({0, 1}), axes({0}));
        const auto mask = std::make_shared<Broadcast>(Constant::create(ov::element::boolean, {}, {true}), dimensions);
        mask->output(0).set_names({"attention_mask"});
        model->add_results({std::make_shared<Result>(mask)});
        model->output(1).set_names({"attention_mask"});
    }
    model->validate_nodes_and_infer_types();
}

GGUFMultimodalModels read_gguf_multimodal(const std::filesystem::path& language,
                                         const std::string& device,
                                         ov::AnyMap& properties) {
    OPENVINO_ASSERT(device == "CPU", "GGUF multimodal generation is currently qualified on CPU only");
    const auto it = properties.find(mmproj_path.name());
    OPENVINO_ASSERT(it != properties.end(), "A language GGUF requires mmproj_path for VLMPipeline");
    const auto mmproj = it->second.as<std::string>();
    properties.erase(it);
    auto [filtered_properties, enable_save_ov_model] = utils::extract_gguf_properties(properties);
    properties = std::move(filtered_properties);
    utils::clear_false_prompt_lookup_from_config(properties);
    utils::validate_vlm_model_properties(properties);
    const auto extensions = utils::extract_extensions_to_core(properties);

    namespace gguf = ov::frontend::gguf;
    GGUFMultimodalModels result;
    gguf::FrontEnd frontend;
    auto genai = register_gguf_extensions(frontend, extensions, gguf::GenAIExtension::InputMode::EMBEDS_TO_LOGITS);
    result.language = frontend.convert(frontend.load(language.string()));
    result.language->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    auto tokenizer_metadata = take_gguf_tokenizer_metadata(result.language);
    result.stop_token_ids = gguf_stop_token_ids(tokenizer_metadata);
    result.tokenizer = GGUFTokenizer(GGUFTokenizerParameters(std::move(tokenizer_metadata)), properties);
    gguf::FrontEnd projector_frontend;
    register_gguf_extensions(projector_frontend, extensions);
    auto combined = projector_frontend.convert(projector_frontend.load(mmproj));
    const MmprojMetadata meta(combined);
    const auto architecture = result.language->get_rt_info<std::string>("gguf_architecture");
    const auto projector = meta.text("vision.projector");
    const bool gemma4 = architecture == "gemma4" && (projector == "gemma4v" || projector == "gemma4uv");
    OPENVINO_ASSERT((architecture == "gemma3" && projector == "gemma3") ||
                        ((architecture == "qwen35" || architecture == "qwen35moe") && projector == "qwen3vl_merger") ||
                        (architecture == "muse-glimmer" && projector == "muse-glimmer") || gemma4,
                    "Unsupported GGUF language/projector pair: ",
                    architecture,
                    " / ",
                    projector);

    auto& vlm = result.vlm;
    using Adapter = gguf::pass::AdaptMmprojToGenAI;
    Adapter vision_adapter(Adapter::Modality::VISION, Adapter::Layout::VISION_ENCODERS);
    {
        auto vision = combined->clone();
        vision_adapter.run_on_model(vision);
        vlm.models = vision_adapter.get_vision_models();
    }
    if (gemma4 && meta.has("audio.projector")) {
        auto audio = combined->clone();
        Adapter(Adapter::Modality::AUDIO).run_on_model(audio);
        const bool unified_audio = meta.text("audio.projector") == "gemma4ua";
        OPENVINO_ASSERT(unified_audio || (meta.text("audio.projector") == "gemma4a" &&
                                          meta.integer("clip.audio.num_mel_bins") == 128),
                        "Unsupported GGUF Gemma4 audio configuration");
        adapt_gguf_audio_encoder(audio, unified_audio);
        vlm.models["audio_embeddings"] = audio;
    }
    vlm.models["text_embeddings"] = genai->get_embedding_model();
    if (const auto& per_layer = genai->get_per_layer_embedding_model())
        vlm.models["text_embeddings_per_layer"] = per_layer;

    const auto configs = gguf_configs(architecture, projector, meta, result);
    vlm.config = VLMConfig(configs.config);
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
