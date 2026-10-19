// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_multimodal.hpp"

#include <algorithm>
#include <nlohmann/json.hpp>
#include <sstream>

#include "gguf_frontend_extensions.hpp"
#include "gguf_tokenizer.hpp"
#include "openvino/frontend/gguf/adapt_mmproj_to_genai.hpp"
#include "openvino/frontend/gguf/extension/genai.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/genai/gguf.hpp"

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

AudioEncoderConfig audio_encoder_config(const MmprojMetadata& meta) {
    const auto projector = meta.text("audio.projector");
    AudioEncoderConfig config;
    if (projector == "lfm2a") {
        config.type = AudioEncoderType::LFM2;
    } else if (projector == "gemma4ua") {
        config.type = AudioEncoderType::GEMMA4_UNIFIED;
    } else {
        OPENVINO_ASSERT(projector == "gemma4a", "Unsupported GGUF audio projector: ", projector);
        config.type = AudioEncoderType::GEMMA4;
    }
    if (config.type != AudioEncoderType::GEMMA4_UNIFIED)
        config.num_mel_bins = meta.integer("clip.audio.num_mel_bins");
    return config;
}

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
    if (projector == "lfm2a") {
        c.config["model_type"] = "lfm2_audio";
        c.config["position_ids_offset"] = 0;
        c.image = json::object();
        c.video = json::object();
        return c;
    }
    const auto patch = meta.integer("clip.vision.patch_size");
    c.config["vision_config"]["patch_size"] = patch;
    c.image["patch_size"] = patch;
    c.image["image_mean"] = meta.floats("clip.vision.image_mean");
    c.image["image_std"] = meta.floats("clip.vision.image_std");
    if (projector == "lfm2") {
        c.config["model_type"] = "lfm2_vl";
        c.config["position_ids_offset"] = 0;
        c.image["merge_size"] = meta.integer("vision.merge");
        const auto merged_patch = patch * c.image["merge_size"].get<size_t>();
        c.image["min_pixels"] = 64 * merged_patch * merged_patch;
        c.image["max_pixels"] = 256 * merged_patch * merged_patch;
        c.image["tile_size"] = 512;
        c.image["min_patches"] = 2;
        c.image["max_patches"] = 10;
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

GGUFMultimodalModels read_gguf_multimodal(const std::filesystem::path& language,
                                          const std::filesystem::path& mmproj,
                                          const ov::AnyMap& properties,
                                          const ExtensionList& extensions) {
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
    auto combined = projector_frontend.convert(projector_frontend.load(mmproj.string()));
    const MmprojMetadata meta(combined);
    const auto architecture = result.language->get_rt_info<std::string>("gguf_architecture");
    const bool lfm_audio =
        architecture == "lfm2" && meta.has("audio.projector") && meta.text("audio.projector") == "lfm2a";
    const auto projector = lfm_audio ? meta.text("audio.projector") : meta.text("vision.projector");
    const bool gemma4 = architecture == "gemma4" && (projector == "gemma4v" || projector == "gemma4uv");
    OPENVINO_ASSERT((architecture == "gemma3" && projector == "gemma3") ||
                        ((architecture == "qwen35" || architecture == "qwen35moe") && projector == "qwen3vl_merger") ||
                        (architecture == "muse-glimmer" && projector == "muse-glimmer") || gemma4 || lfm_audio ||
                        (architecture == "lfm2" && projector == "lfm2"),
                    "Unsupported GGUF language/projector pair: ",
                    architecture,
                    " / ",
                    projector);

    auto& vlm = result.vlm;
    using Adapter = gguf::pass::AdaptMmprojToGenAI;
    Adapter vision_adapter(Adapter::Modality::VISION, Adapter::Layout::VISION_ENCODERS);
    if (!lfm_audio) {
        auto vision = combined->clone();
        vision_adapter.run_on_model(vision);
        vlm.models = vision_adapter.get_vision_models();
    }
    if ((gemma4 || lfm_audio) && meta.has("audio.projector")) {
        auto audio = combined->clone();
        Adapter(Adapter::Modality::AUDIO).run_on_model(audio);
        vlm.models["audio_embeddings"] = audio;
        vlm.audio_encoder = audio_encoder_config(meta);
    }
    vlm.models["text_embeddings"] = genai->get_embedding_model();
    if (const auto& per_layer = genai->get_per_layer_embedding_model())
        vlm.models["text_embeddings_per_layer"] = per_layer;

    const auto configs = gguf_configs(architecture, projector, meta, result);
    vlm.config = VLMConfig(configs.config);
    vlm.processor_config = ProcessorConfig(configs.image);
    vlm.video_processor_config = VideoProcessorConfig(configs.video);
    if (architecture == "gemma3") {
        // GGUF Gemma3 vocabularies lack <image_soft_token>; <pad> only marks image positions.
        vlm.config.image_soft_token = "<pad>";
        const auto placeholder = result.tokenizer.encode(vlm.config.image_soft_token, add_special_tokens(false));
        OPENVINO_ASSERT(placeholder.input_ids.get_size() == 1,
                        "GGUF Gemma3 requires a single-token padding placeholder for image assembly");
    }
    return result;
}
}  // namespace ov::genai
