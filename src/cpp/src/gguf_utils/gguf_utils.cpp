// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_utils.hpp"

#include <vector>

#include "openvino/genai/llm_pipeline.hpp"
#include "utils.hpp"

#ifdef ENABLE_GGUF
#include "gguf_modeling.hpp"
#endif

namespace ov::genai {

const std::string FRONTEND_GGUF_READER = "FRONTEND";
const std::string LEGACY_GGUF_READER = "LEGACY";

namespace gguf_utils {

GGUFProperties extract_gguf_properties(const ov::AnyMap& external_properties) {
    GGUFProperties result;
    auto [rest, enable_save_ov_model] = utils::extract_gguf_properties(external_properties);
    result.rest = std::move(rest);
    result.enable_save_ov_model = enable_save_ov_model;

    auto reader_it = result.rest.find(ov::genai::gguf_reader.name());
    if (reader_it != result.rest.end()) {
        const auto reader = reader_it->second.as<std::string>();
        if (reader == LEGACY_GGUF_READER) {
            OPENVINO_THROW("The legacy GGUF reader has been removed. "
                           "Please use GGUF_READER=\"FRONTEND\" (Python) or "
                           "ov::genai::gguf_reader(\"FRONTEND\") (C++), "
                           "or omit the property as the OpenVINO GGUF frontend is now the default.");
        }
        OPENVINO_ASSERT(reader == FRONTEND_GGUF_READER,
                        "GGUF reader must be '",
                        FRONTEND_GGUF_READER,
                        "', got '",
                        reader,
                        "'");
        result.rest.erase(reader_it);
    }

    return result;
}

std::shared_ptr<ov::Model> read_gguf_model(const std::filesystem::path& model_path,
                                         const GGUFProperties& properties) {
#ifdef ENABLE_GGUF
    return create_from_gguf(model_path.string(), properties.enable_save_ov_model);
#else
    OPENVINO_THROW("GGUF support is switched off. Please, recompile with 'cmake -DENABLE_GGUF=ON'");
#endif
}

bool has_gguf_recurrent_states(const ov::Model& model) {
    const auto recurrent_states = model.get_rt_info().find("gguf_recurrent_states");
    return recurrent_states != model.get_rt_info().end() &&
           !recurrent_states->second.as<std::vector<std::string>>().empty();
}

}  // namespace gguf_utils
}  // namespace ov::genai
