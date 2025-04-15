// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

#include "gguf_modeling.hpp"

#include "openvino/openvino.hpp"

int main(int argc, char* argv[]) {
    // std::string models_path = argv[1];
    // std::string output_path = argv[2];

    auto model = create_from_gguf("/home/chentianmeng/model_file/Qwen-gguf/Qwen2.5-7B-Instruct-q6k/qwen2.5-7b-instruct-q6_k.gguf");

    ov::save_model(model,  "qwen/openvino_model.xml", false);
}
