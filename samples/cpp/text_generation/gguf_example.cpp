// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

#include "gguf_modeling.hpp"

#include "openvino/openvino.hpp"

int main(int argc, char* argv[]) {
    std::string models_path = argv[1];
    std::string output_path = argv[2];

    auto model = create_from_gguf(models_path);

    ov::save_model(model, output_path + "/openvino_model.xml", false);
}
