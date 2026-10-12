// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_utils.hpp"

#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "utils.hpp"

#ifdef ENABLE_GGUF
#include "gguf_modeling.hpp"
#endif

namespace ov::genai {

const std::string FRONTEND_GGUF_READER = "FRONTEND";
const std::string LEGACY_GGUF_READER = "LEGACY";

bool is_gguf_model(const std::filesystem::path& file_path) {
    return file_path.extension() == ".gguf";
}

namespace gguf_utils {

GGUFProperties extract_gguf_properties(const ov::AnyMap& external_properties) {
    GGUFProperties result;
    result.rest = external_properties;

    auto save_it = result.rest.find(ov::genai::enable_save_ov_model.name());
    if (save_it != result.rest.end()) {
        result.enable_save_ov_model = save_it->second.as<bool>();
        result.rest.erase(save_it);
    }

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

    return result;
}

void save_openvino_model(const std::shared_ptr<ov::Model>& model, const std::string& save_path, bool compress_to_fp16) {
    try {
        auto serialize_start_time = std::chrono::high_resolution_clock::now();
        ov::save_model(model, save_path, compress_to_fp16);
        auto serialize_finish_time = std::chrono::high_resolution_clock::now();
        auto serialize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(serialize_finish_time - serialize_start_time).count();
        std::stringstream ss;
        ss << "Save generated OpenVINO model to: " << save_path << " done. Time: " << serialize_duration << " ms";
        print_gguf_debug_info(ss.str());
    }
    catch (const ov::Exception& e) {
        OPENVINO_THROW("Exception during model serialization ", e.what(), ", user can disable it by setting 'ov::genai::enable_save_ov_model' property to false");
    }
}

std::shared_ptr<ov::Model> read_gguf_model(const std::filesystem::path& model_path,
                                         const GGUFProperties& properties) {
#ifdef ENABLE_GGUF
    try {
        return create_from_gguf(model_path.string(), properties.enable_save_ov_model, properties.use_legacy_reader());
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

bool has_gguf_recurrent_states(const ov::Model& model) {
    const auto recurrent_states = model.get_rt_info().find("gguf_recurrent_states");
    return recurrent_states != model.get_rt_info().end() &&
           !recurrent_states->second.as<std::vector<std::string>>().empty();
}

void print_gguf_debug_info(const std::string& debug_info) {
    if (!utils::env_setup_for_print_debug_info()) {
        return;
    }
    std::cout << "[GGUF Reader]: " << debug_info << std::endl;
}

}  // namespace gguf_utils
}  // namespace ov::genai
