// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_utils.hpp"

#include <utility>
#include <vector>

#include "model_desc.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "utils.hpp"

#ifdef ENABLE_GGUF
#include "gguf.hpp"
#include "gguf_modeling.hpp"
#endif

namespace ov::genai {

const std::string FRONTEND_GGUF_READER = "FRONTEND";
const std::string LEGACY_GGUF_READER = "LEGACY";

namespace gguf_utils {

const std::string GGUF_MTP_TARGET_PROPERTY = "GGUF_MTP_TARGET";

GGUFProperties extract_gguf_properties(const ov::AnyMap& external_properties) {
    GGUFProperties result;
    auto [rest, enable_save_ov_model] = utils::extract_gguf_properties(external_properties);
    result.rest = std::move(rest);
    result.enable_save_ov_model = enable_save_ov_model;

    auto reader_it = result.rest.find(ov::genai::gguf_reader.name());
    if (reader_it != result.rest.end()) {
        const auto reader = reader_it->second.as<std::string>();
        OPENVINO_ASSERT(reader == FRONTEND_GGUF_READER || reader == LEGACY_GGUF_READER,
                        "GGUF reader must be either '",
                        FRONTEND_GGUF_READER,
                        "' or '",
                        LEGACY_GGUF_READER,
                        "', got '",
                        reader,
                        "'");
        result.legacy_reader = reader == LEGACY_GGUF_READER;
        result.rest.erase(reader_it);
    }
    if (const auto it = result.rest.find(GGUF_MTP_TARGET_PROPERTY); it != result.rest.end()) {
        result.mtp_target = it->second.as<GGUFModelHolder>();
        result.rest.erase(it);
    }

    return result;
}

std::shared_ptr<ov::Model> read_gguf_model(const std::filesystem::path& model_path,
                                         const GGUFProperties& properties) {
#ifdef ENABLE_GGUF
    if (properties.mtp_target && *properties.mtp_target)
        return std::exchange(*properties.mtp_target, nullptr);
    try {
        const auto draft = properties.rest.find(utils::DRAFT_MODEL_ARG_NAME);
        const bool embeddings = draft != properties.rest.end() &&
                                draft->second.as<ModelDesc>().properties.count("mtp_mode") &&
                                draft->second.as<ModelDesc>().properties.at("mtp_mode").as<bool>();
        OPENVINO_ASSERT(!embeddings || !properties.use_legacy_reader(), "GGUF MTP requires GGUF_READER=\"FRONTEND\".");
        OPENVINO_ASSERT(!embeddings || !properties.enable_save_ov_model,
                        "Saving a decomposed GGUF MTP model with ENABLE_SAVE_OV_MODEL is not supported.");
        return create_from_gguf(model_path.string(),
                                properties.enable_save_ov_model,
                                properties.use_legacy_reader(),
                                embeddings);
    } catch (const std::exception& error) {
        if (!properties.legacy_reader.has_value()) {
            OPENVINO_THROW(error.what(),
                           "\nThe default legacy GGUF reader failed to load this model. "
                           "Try GGUF_READER=\"FRONTEND\" (Python) or "
                           "ov::genai::gguf_reader(\"FRONTEND\") (C++). "
                           "The OpenVINO GGUF frontend should provide better model quality and broader "
                           "architecture support, but it is in preview and may have limitations.");
        }
        throw;
    }
#else
    OPENVINO_THROW("GGUF support is switched off. Please, recompile with 'cmake -DENABLE_GGUF=ON'");
#endif
}

GGUFMtpModels read_gguf_mtp_models(const std::filesystem::path& model_path, const GGUFProperties& properties) {
#ifdef ENABLE_GGUF
    OPENVINO_ASSERT(!properties.legacy_reader.value_or(false), "GGUF MTP drafts require GGUF_READER=\"FRONTEND\".");
    OPENVINO_ASSERT(!properties.enable_save_ov_model, "Saving a GGUF MTP draft is not supported.");
    GGUFMtpModels models;
    models.target = convert_gguf_with_frontend(model_path.string(), true, "mtp", &models.draft);
    OPENVINO_ASSERT(models.draft, "The GGUF frontend did not produce an MTP draft for ", model_path.string());
    return models;
#else
    OPENVINO_THROW("GGUF support is switched off. Please, recompile with 'cmake -DENABLE_GGUF=ON'");
#endif
}

bool has_gguf_mtp_layers(const std::filesystem::path& model_path) {
#ifdef ENABLE_GGUF
    const auto metadata = get_gguf_metadata(model_path.string());
    const auto arch_it = metadata.find("general.architecture");
    if (arch_it == metadata.end() || !std::holds_alternative<std::string>(arch_it->second))
        return false;
    const auto layers_it = metadata.find(std::get<std::string>(arch_it->second) + ".nextn_predict_layers");
    if (layers_it == metadata.end())
        return false;
    if (std::holds_alternative<int>(layers_it->second))
        return std::get<int>(layers_it->second) > 0;
    if (!std::holds_alternative<ov::Tensor>(layers_it->second))
        return false;
    const auto& tensor = std::get<ov::Tensor>(layers_it->second);
    if (tensor.get_size() != 1)
        return false;
    switch (tensor.get_element_type()) {
    case ov::element::u8:
        return *tensor.data<const uint8_t>() > 0;
    case ov::element::i8:
        return *tensor.data<const int8_t>() > 0;
    case ov::element::u16:
        return *tensor.data<const uint16_t>() > 0;
    case ov::element::i16:
        return *tensor.data<const int16_t>() > 0;
    case ov::element::u32:
        return *tensor.data<const uint32_t>() > 0;
    case ov::element::i32:
        return *tensor.data<const int32_t>() > 0;
    case ov::element::u64:
        return *tensor.data<const uint64_t>() > 0;
    case ov::element::i64:
        return *tensor.data<const int64_t>() > 0;
    default:
        return false;
    }
#else
    OPENVINO_THROW("GGUF support is switched off. Please, recompile with 'cmake -DENABLE_GGUF=ON'");
#endif
}

std::shared_ptr<ov::Model> take_text_embeddings(const std::shared_ptr<ov::Model>& model) {
    auto& info = model->get_rt_info();
    const auto it = info.find("gguf_text_embeddings");
    if (it == info.end())
        return nullptr;
    auto embeddings = it->second.as<std::shared_ptr<ov::Model>>();
    info.erase(it);
    return embeddings;
}

bool has_gguf_recurrent_states(const ov::Model& model) {
    const auto recurrent_states = model.get_rt_info().find("gguf_recurrent_states");
    return recurrent_states != model.get_rt_info().end() &&
           !recurrent_states->second.as<std::vector<std::string>>().empty();
}

}  // namespace gguf_utils
}  // namespace ov::genai
