// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "model_desc.hpp"

#include "gguf_utils/gguf_tokenizer.hpp"
#include "gguf_utils/gguf_utils.hpp"
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

bool has_mtp_layers(const std::filesystem::path& models_path) {
    if (is_gguf_model(models_path))
        return gguf_utils::has_gguf_mtp_layers(models_path);
    return std::filesystem::exists(models_path / "openvino_mtp_model.xml");
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
                                       const ov::AnyMap& properties) {
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
    if (!speculative_type && !draft_path)
        return resolved;

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
        OPENVINO_ASSERT(has_mtp_layers(source),
                        "speculative_type='mtp' requires MTP layers, but none were found in '",
                        source.string(),
                        "'. A GGUF file must set <arch>.nextn_predict_layers; an IR directory must contain "
                        "openvino_mtp_model.xml.");
    }

    // The draft is read here, so extensions must be registered first.
    utils::extract_extensions_to_core(resolved);
    ov::AnyMap draft_properties = resolved;
    if (is_gguf_model(models_path) && is_gguf_model(source) &&
        resolved.find(ov::genai::gguf_reader.name()) == resolved.end()) {
        resolved[ov::genai::gguf_reader.name()] = FRONTEND_GGUF_READER;
    }
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
    resolved.insert(ov::genai::draft_model(source, device, draft_properties));
    return resolved;
}
}  // namespace genai
}  // namespace ov
