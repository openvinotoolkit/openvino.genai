
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "model_yaml.hpp"
#include "utils.hpp"

#include <yaml-cpp/yaml.h>

namespace TEST_MODEL {

std::string get_device() {
    const char* device_env = std::getenv("DEVICE");
    if (device_env != nullptr && std::string(device_env) != "") {
        return std::string(device_env);
    }
    return "CPU";
}

std::string Qwen2_5_VL_3B_Instruct_INT4() {
    return get_model_path() + "/Qwen2.5-VL-3B-Instruct/INT4/";
}

std::string ZImage_Turbo_fp16_ov() {
    return get_model_path() + "/Z-Image-Turbo-fp16-ov/";
}

std::string Wan_2_1() {
    return get_model_path() + "/Wan2.1-T2V-1.3B-Diffusers/";
}

std::string Qwen3_5() {
    return get_model_path() + "/Qwen3.5-35B-A3B-Base_VL_OV_IR/";
}

std::string Qwen3_5_0_8B() {
    return get_model_path() + "/Qwen3.5-0.8B/";
}

std::string get_qwen2_5_vl_config_yaml(const std::string& model_path, const std::string& device) {
    YAML::Node config;

    // Based on the provided YAML structure, generate the configuration dynamically.
    config["global_context"]["model_type"] = "qwen2_5_vl";

    YAML::Node pipeline_modules = config["pipeline_modules"];

    YAML::Node pipeline_params;
    pipeline_params["type"] = "ParameterModule";
    YAML::Node param_outputs;
    param_outputs.push_back(output_node("img1", "OVTensor"));
    param_outputs.push_back(output_node("prompts_data", "String"));
    pipeline_params["outputs"] = param_outputs;
    pipeline_modules["pipeline_params"] = pipeline_params;

    YAML::Node image_preprocessor;
    image_preprocessor["type"] = "ImagePreprocessModule";
    image_preprocessor["device"] = device;
    image_preprocessor["description"] = "Image or Video preprocessing.";
    YAML::Node inputs;
    inputs.push_back(input_node("image", "OVTensor", "pipeline_params.img1"));
    image_preprocessor["inputs"] = inputs;
    YAML::Node outputs;
    outputs.push_back(output_node("raw_data", "OVTensor"));
    outputs.push_back(output_node("source_size", "VecInt"));
    image_preprocessor["outputs"] = outputs;
    YAML::Node model_path_node;
    model_path_node["model_path"] = model_path;
    image_preprocessor["params"] = model_path_node;
    pipeline_modules["image_preprocessor"] = image_preprocessor;

    YAML::Node prompt_encoder;
    prompt_encoder["type"] = "TextEncoderModule";
    prompt_encoder["device"] = device;
    YAML::Node pe_inputs;
    pe_inputs.push_back(input_node("prompts", "String", "pipeline_params.prompts_data"));
    pe_inputs.push_back(input_node("encoded_image", "OVTensor", "image_preprocessor.raw_data"));
    pe_inputs.push_back(input_node("source_size", "VecInt", "image_preprocessor.source_size"));
    prompt_encoder["inputs"] = pe_inputs;
    YAML::Node pe_outputs;
    pe_outputs.push_back(output_node("input_ids", "OVTensor"));
    pe_outputs.push_back(output_node("mask", "OVTensor"));
    pe_outputs.push_back(output_node("images_sequence", "VecInt"));
    prompt_encoder["outputs"] = pe_outputs;
    YAML::Node pe_model_path_node;
    pe_model_path_node["model_path"] = model_path;
    prompt_encoder["params"] = pe_model_path_node;
    pipeline_modules["prompt_encoder"] = prompt_encoder;

    YAML::Node text_embedding;
    text_embedding["type"] = "TextEmbeddingModule";
    text_embedding["device"] = device;
    YAML::Node te_inputs;
    te_inputs.push_back(input_node("input_ids", "OVTensor", "prompt_encoder.input_ids"));
    text_embedding["inputs"] = te_inputs;
    YAML::Node te_outputs;
    te_outputs.push_back(output_node("input_embedding", "OVTensor"));
    text_embedding["outputs"] = te_outputs;
    YAML::Node te_model_path_node;
    te_model_path_node["model_path"] = model_path;
    te_model_path_node["scale_emb"] = "1.0";
    text_embedding["params"] = te_model_path_node;
    pipeline_modules["text_embedding"] = text_embedding;

    YAML::Node vision_encoder;
    vision_encoder["type"] = "VisionEncoderModule";
    vision_encoder["device"] = device;
    YAML::Node ve_inputs;
    ve_inputs.push_back(input_node("preprocessed_image", "OVTensor", "image_preprocessor.raw_data"));
    ve_inputs.push_back(input_node("source_size", "VecInt", "image_preprocessor.source_size"));
    ve_inputs.push_back(input_node("images_sequence", "VecInt", "prompt_encoder.images_sequence"));
    ve_inputs.push_back(input_node("input_ids", "OVTensor", "prompt_encoder.input_ids"));
    vision_encoder["inputs"] = ve_inputs;
    YAML::Node ve_outputs;
    ve_outputs.push_back(output_node("image_embedding", "OVTensor"));
    ve_outputs.push_back(output_node("video_embedding", "OVTensor"));
    ve_outputs.push_back(output_node("position_ids", "OVTensor"));
    ve_outputs.push_back(output_node("rope_delta", "Int"));
    vision_encoder["outputs"] = ve_outputs;
    YAML::Node ve_model_path_node;
    ve_model_path_node["model_path"] = model_path;
    ve_model_path_node["vision_start_token_id"] = "151652";
    vision_encoder["params"] = ve_model_path_node;
    pipeline_modules["vision_encoder"] = vision_encoder;

    YAML::Node embedding_merger;
    embedding_merger["type"] = "EmbeddingMergerModule";
    embedding_merger["device"] = device;
    YAML::Node em_inputs;
    em_inputs.push_back(input_node("input_ids", "OVTensor", "prompt_encoder.input_ids"));
    em_inputs.push_back(input_node("input_embedding", "OVTensor", "text_embedding.input_embedding"));
    em_inputs.push_back(input_node("image_embedding", "OVTensor", "vision_encoder.image_embedding"));
    em_inputs.push_back(input_node("video_embedding", "OVTensor", "vision_encoder.video_embedding"));
    embedding_merger["inputs"] = em_inputs;
    YAML::Node em_outputs;
    em_outputs.push_back(output_node("merged_embedding", "OVTensor"));
    embedding_merger["outputs"] = em_outputs;
    YAML::Node em_model_path_node;
    em_model_path_node["model_path"] = model_path;
    embedding_merger["params"] = em_model_path_node;
    pipeline_modules["embedding_merger"] = embedding_merger;

    YAML::Node llm_inference;
    llm_inference["type"] = "LLMInferenceModule";
    llm_inference["description"] = "LLM module for Continuous Batch pipeline";
    llm_inference["device"] = device;
    YAML::Node li_inputs;
    li_inputs.push_back(input_node("embeds", "OVTensor", "embedding_merger.merged_embedding"));
    li_inputs.push_back(input_node("position_ids", "OVTensor", "vision_encoder.position_ids"));
    li_inputs.push_back(input_node("rope_delta", "Int", "vision_encoder.rope_delta"));
    llm_inference["inputs"] = li_inputs;
    YAML::Node li_outputs;
    li_outputs.push_back(output_node("generated_text", "String"));
    llm_inference["outputs"] = li_outputs;
    YAML::Node li_model_path_node;
    li_model_path_node["model_path"] = model_path;
    li_model_path_node["max_new_tokens"] = "16";
    li_model_path_node["do_sample"] = "false";
    li_model_path_node["top_p"] = "1.0";
    li_model_path_node["top_k"] = "50";
    li_model_path_node["temperature"] = "1.0";
    li_model_path_node["repetition_penalty"] = "1.0";
    llm_inference["params"] = li_model_path_node;
    pipeline_modules["llm_inference"] = llm_inference;

    YAML::Node pipeline_result;
    pipeline_result["type"] = "ResultModule";
    pipeline_result["description"] = "Collects final results and formats the output structure.";
    pipeline_result["device"] = device;
    YAML::Node pr_inputs;
    pr_inputs.push_back(input_node("generated_text", "String", "llm_inference.generated_text"));
    pipeline_result["inputs"] = pr_inputs;
    pipeline_modules["pipeline_result"] = pipeline_result;

    return YAML::Dump(config);
}
}  // namespace TEST_MODEL