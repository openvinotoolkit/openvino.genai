// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <iostream>
#include <chrono>
#include <sstream>

#include <openvino/openvino.hpp>
#include "openvino/frontend/gguf/extension/genai.hpp"
#include "openvino/frontend/gguf/frontend.hpp"

#include "gguf_utils/gguf_modeling.hpp"
#include "gguf_utils/gguf_utils.hpp"
#include "utils.hpp"

std::shared_ptr<ov::Model> convert_gguf_with_frontend(const std::string& model_path) {
    ov::frontend::gguf::FrontEnd frontend;
    frontend.add_extension(std::make_shared<ov::frontend::gguf::GenAIExtension>());
    return frontend.convert(frontend.load(model_path));
}

std::shared_ptr<ov::Model> create_from_gguf(const std::string& model_path,
                                            bool enable_save_ov_model) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::stringstream ss;

    ss << "Loading and converting GGUF model via the OpenVINO GGUF frontend: " << model_path;
    ov::genai::utils::print_gguf_debug_info(ss.str());

    std::shared_ptr<ov::Model> model = convert_gguf_with_frontend(model_path);

    if (enable_save_ov_model) {
        std::filesystem::path save_path = std::filesystem::path(model_path).parent_path() / "openvino_model.xml";
        ov::genai::utils::save_openvino_model(model, save_path.string(), true);
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count();
    ss.str("");
    ss << "GGUF model conversion done. Time: " << duration << "ms";
    ov::genai::utils::print_gguf_debug_info(ss.str());

    return model;
}
