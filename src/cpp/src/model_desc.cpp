// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "model_desc.hpp"

#include "gguf_utils/gguf_tokenizer.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {
namespace {

std::filesystem::path get_draft_model_path(const ov::Any& value) {
    if (value.is<std::filesystem::path>())
        return value.as<std::filesystem::path>();
    return value.as<std::string>();
}

bool has_mtp_mode(const ov::genai::ModelDesc& draft) {
    const auto mtp_mode = draft.properties.find("mtp_mode");
    return mtp_mode != draft.properties.end() && mtp_mode->second.as<bool>();
}

void validate_mtp_target(const std::filesystem::path& models_path, const ov::genai::ModelDesc& draft) {
    OPENVINO_ASSERT(!has_mtp_mode(draft) || !is_gguf_model(models_path),
                    "MTP drafts do not support raw GGUF target models; use a converted OpenVINO target model.");
}

}  // namespace
ov::genai::ModelDesc get_draft_model_from_config(const ov::AnyMap& config) {
    ov::genai::ModelDesc draft_model;
    if (config.find(utils::DRAFT_MODEL_ARG_NAME) != config.end()) {
        draft_model = config.at(utils::DRAFT_MODEL_ARG_NAME).as<ov::genai::ModelDesc>();
    }
    return draft_model;
}

ov::genai::ModelDesc extract_draft_model_from_config(ov::AnyMap& config) {
    ov::genai::ModelDesc draft_model;
    if (config.find(ov::genai::utils::DRAFT_MODEL_ARG_NAME) != config.end()) {
        draft_model = config.at(ov::genai::utils::DRAFT_MODEL_ARG_NAME).as<ov::genai::ModelDesc>();
        config.erase(ov::genai::utils::DRAFT_MODEL_ARG_NAME);
    }
    return draft_model;
}

ov::AnyMap resolve_draft_model_request(const std::filesystem::path& models_path,
                                       const std::string& device,
                                       const ov::AnyMap& properties,
                                       bool allow_mtp) {
    ov::AnyMap resolved = properties;
    std::optional<std::string> speculative_type;
    std::optional<std::filesystem::path> draft_path;
    if (const auto it = resolved.find(ov::genai::speculative_type.name()); it != resolved.end()) {
        speculative_type = it->second.as<std::string>();
        resolved.erase(it);
    }
    if (const auto it = resolved.find(ov::genai::draft_model_path.name()); it != resolved.end()) {
        draft_path = get_draft_model_path(it->second);
        resolved.erase(it);
    }
    if (!speculative_type && !draft_path) {
        const auto draft_it = resolved.find(utils::DRAFT_MODEL_ARG_NAME);
        if (draft_it != resolved.end()) {
            const auto& draft = draft_it->second.as<ov::genai::ModelDesc>();
            OPENVINO_ASSERT(allow_mtp || !draft.is_gguf_mtp,
                            "GGUF MTP drafts are supported only by ContinuousBatchingPipeline.");
            validate_mtp_target(models_path, draft);
        }
        return resolved;
    }

    OPENVINO_ASSERT(resolved.find(utils::DRAFT_MODEL_ARG_NAME) == resolved.end(),
                    "draft_model cannot be combined with speculative_type or draft_model_path.");
    const auto prompt_lookup_it = resolved.find(ov::genai::prompt_lookup.name());
    OPENVINO_ASSERT(prompt_lookup_it == resolved.end() || !prompt_lookup_it->second.as<bool>(),
                    "Speculative decoding and prompt lookup decoding are mutually exclusive.");

    const auto source = draft_path.value_or(models_path);
    if (speculative_type) {
        OPENVINO_ASSERT(*speculative_type == "mtp",
                        "Unsupported speculative_type '",
                        *speculative_type,
                        "'. Supported value: 'mtp'.");
    }
    if (is_gguf_model(source)) {
        OPENVINO_ASSERT(allow_mtp, "GGUF MTP drafts are supported only by ContinuousBatchingPipeline.");
        OPENVINO_ASSERT(!is_gguf_model(models_path) && draft_path.has_value(),
                        "GGUF MTP requires a converted VLM model directory and an explicit GGUF "
                        "draft_model_path.");
        OPENVINO_ASSERT(std::filesystem::exists(models_path / "openvino_text_embeddings_model.xml"),
                        "GGUF MTP requires the converted VLM directory to contain "
                        "openvino_text_embeddings_model.xml.");
    } else if (speculative_type) {
        OPENVINO_ASSERT(std::filesystem::exists(source / "openvino_mtp_model.xml"),
                        "speculative_type='mtp' requires MTP layers, but none were found in '",
                        source.string(),
                        "'. An IR directory must contain openvino_mtp_model.xml.");
    }

    // The draft is read here, so extensions must be registered first.
    utils::extract_extensions_to_core(resolved);
    ov::AnyMap draft_properties = resolved;
    for (const auto& name : {ov::genai::scheduler_config.name(),
                             ov::genai::prompt_lookup.name(),
                             utils::CONFIG_ARG_NAME.c_str(),
                             utils::STREAMER_ARG_NAME.c_str(),
                             "ATTENTION_BACKEND"}) {
        draft_properties.erase(name);
    }
    // A GGUF draft validates the reader options itself; an IR draft must not receive them.
    if (!is_gguf_model(source)) {
        draft_properties.erase(ov::genai::gguf_reader.name());
        draft_properties.erase(ov::genai::enable_save_ov_model.name());
    }
    const auto draft = ov::genai::draft_model(source, device, draft_properties);
    const auto& loaded_draft = draft.second.as<ov::genai::ModelDesc>();
    OPENVINO_ASSERT(allow_mtp || !loaded_draft.is_gguf_mtp,
                    "GGUF MTP drafts are supported only by ContinuousBatchingPipeline.");
    validate_mtp_target(models_path, loaded_draft);
    resolved.insert(draft);
    return resolved;
}
}  // namespace genai
}  // namespace ov
