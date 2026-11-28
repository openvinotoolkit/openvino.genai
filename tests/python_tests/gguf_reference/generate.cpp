// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Reuse the pinned upstream fixture factory, which is internal to its test executable.
#define main llama_arch_test_main
#include "tests/test-llama-archs.cpp"
#undef main

#include "src/llama-model.h"
#include "src/llama-model-saver.h"

// Keep generated weights stable when the upstream fixture factory changes its initializer.
static void reset_fixture_tensor(struct ggml_tensor* tensor) {
    const size_t seed = 1 ^ std::hash<std::string>{}(tensor->name);
    std::mt19937 generator(seed);
    std::normal_distribution<float> distribution(0.0f, 0.01f);
    std::vector<float> values(ggml_nelements(tensor));
    for (auto& value : values)
        value = distribution(generator);
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half_values(values.size());
        std::transform(values.begin(), values.end(), half_values.begin(), ggml_fp32_to_fp16);
        ggml_backend_tensor_set(tensor, half_values.data(), 0, ggml_nbytes(tensor));
    } else {
        throw std::runtime_error("Expected F32 or F16 fixture weights");
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: genai-gguf-generate architecture output-directory dense|moe\n");
        return 2;
    }
    try {
        common_init();
        const auto arch = llm_arch_from_string(argv[1]);
        if (arch == LLM_ARCH_UNKNOWN || !arch_supported(arch) || !llama_arch_supports_save(arch))
            throw std::runtime_error("Architecture is not supported by the pinned fixture factory/saver");
        const std::string variant = argv[3];
        const bool moe = variant == "moe";
        if ((variant != "dense" && variant != "moe") || (moe && !moe_implemented(arch)) ||
            (!moe && moe_mandatory(arch)))
            throw std::runtime_error("Variant is not supported by the pinned fixture factory");
        auto metadata = get_gguf_ctx(arch, moe);
        gguf_set_val_str(metadata.get(), "tokenizer.ggml.model", "none");
        const std::string prefix = std::string(argv[1]) + ".";
        // The generic fixture sets residual/SWA metadata even for architectures
        // that ignore it. MiniCPM has its own backward-compatible scale defaults.
        gguf_remove_key(metadata.get(), (prefix + "residual_scale").c_str());
        if (arch != LLM_ARCH_GEMMA2 && arch != LLM_ARCH_OPENAI_MOE) {
            gguf_remove_key(metadata.get(), (prefix + "attention.sliding_window").c_str());
            gguf_remove_key(metadata.get(), (prefix + "attention.sliding_window_pattern").c_str());
        }
        if (arch != LLM_ARCH_BAILINGMOE2 && arch != LLM_ARCH_ERNIE4_5_MOE)
            gguf_remove_key(metadata.get(), (prefix + "leading_dense_block_count").c_str());
        if (arch != LLM_ARCH_ERNIE4_5_MOE)
            gguf_remove_key(metadata.get(), (prefix + "interleave_moe_layer_step").c_str());
        if (moe && arch != LLM_ARCH_BAILINGMOE2)
            gguf_set_val_u32(metadata.get(), (prefix + "expert_gating_func").c_str(), 1);
        if (moe) {
            gguf_set_val_u32(metadata.get(), (prefix + "expert_count").c_str(), 4);
            gguf_set_val_u32(metadata.get(), (prefix + "expert_used_count").c_str(), 2);
            if (arch == LLM_ARCH_LLAMA || arch == LLM_ARCH_MINICPM || arch == LLM_ARCH_MISTRAL3) {
                gguf_set_val_bool(metadata.get(), (prefix + "expert_weights_norm").c_str(), true);
                gguf_remove_key(metadata.get(), (prefix + "expert_shared_count").c_str());
                gguf_remove_key(metadata.get(), (prefix + "expert_shared_feed_forward_length").c_str());
            }
        }
        if (arch == LLM_ARCH_GEMMA2) {
            gguf_set_val_f32(metadata.get(), (prefix + "attn_logit_softcapping").c_str(), 50.0f);
            gguf_set_val_f32(metadata.get(), (prefix + "final_logit_softcapping").c_str(), 30.0f);
        }
        init_normal_pool(1, 0.01f);
        auto model_and_ctx = get_model_and_ctx(metadata.get(), nullptr, 1, 0.01f, {});
        // add_kv_from_model() emits unused hparam defaults as real metadata, e.g.
        // llama.logit_scale=0. Preserve the factory inputs instead, without its dummy tensors.
        llama_model_saver saver(model_and_ctx.first.get());
        gguf_set_kv(saver.gguf_ctx, metadata.get());
        // The user-model factory creates every optional tensor. Leave optional biases,
        // RoPE factors and the LFM2 audio-detokenizer head out to exercise the ordinary
        // decoder configuration. F32 weights do not need the randomly initialized
        // quantization scale tensors.
        const bool keep_biases = arch == LLM_ARCH_OPENAI_MOE || arch == LLM_ARCH_JAIS2;
        const bool keep_expert_bias =
            arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_MINIMAX_M2;
        for (const auto& [name, tensor] : model_and_ctx.first->tensors_by_name) {
            if ((!keep_biases && name.find("ssm_dt.bias") == std::string::npos &&
                 name.find("ssm_conv1d.bias") == std::string::npos &&
                 !(keep_expert_bias && name.find("exp_probs_b.bias") != std::string::npos) &&
                 name.find(".bias") != std::string::npos) ||
                name.find("rope_freqs") != std::string::npos || name.find("rope_factors") != std::string::npos ||
                name.find(".scale") != std::string::npos || name.find(".input_scale") != std::string::npos ||
                name.rfind("dense_2.", 0) == 0 || name.rfind("cls.output.", 0) == 0 ||
                gguf_find_tensor(saver.gguf_ctx, name.c_str()) >= 0)
                continue;
            reset_fixture_tensor(tensor);
            if (name.find("norm.weight") != std::string::npos && arch != LLM_ARCH_GEMMA && arch != LLM_ARCH_GEMMA2) {
                // Non-Gemma norms need unit-centered weights so the residual does not
                // overwhelm attention/FFN. Gemma applies its own +1 in the graph.
                if (tensor->type != GGML_TYPE_F32)
                    throw std::runtime_error("Expected F32 normalization weights");
                std::vector<float> values(ggml_nelements(tensor));
                ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
                for (auto& value : values)
                    value += 1.0f;
                ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
            }
            saver.add_tensor(tensor);
        }
        const std::string path = std::string(argv[2]) + "/" + argv[1] + (moe ? "-moe.gguf" : "-dense.gguf");
        if (!gguf_write_to_file(saver.gguf_ctx, path.c_str(), false))
            throw std::runtime_error("Cannot write generated GGUF");
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
